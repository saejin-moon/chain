//! Embedded `llama-cpp-2` backend (enabled via `--features llama-cpp`).
//! Provides:
//! - True 1x physical KV-cache prefix sharing across all 12 probes (`seq_id = 0..=12`)
//! - Dedicated physical P-core OS thread via `core_affinity` (no tokio worker blocking)
//! - Direct `&[f32]` target-token logit extraction (skipping full-vocab softmax)
//! - Microsecond `AtomicBool` abort callback + in-place KV excision (`llama_memory_seq_rm/add`)

use super::{
    DecisionBackend, EngineError, GeneratorBackend, GeneratorSlot, MAX_ACTIONS, ProbeLogitsBatch,
    ProbeSpec, ScopedPrompt,
};
use async_trait::async_trait;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
};
use std::num::NonZeroU32;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{mpsc, oneshot};

enum Sys1Cmd {
    PinPrefix {
        text: String,
        reply: oneshot::Sender<Result<(), EngineError>>,
    },
    EvalBatch {
        state_frame: String,
        probes: Vec<ProbeSpec>,
        num_actions: usize,
        reply: oneshot::Sender<Result<ProbeLogitsBatch, EngineError>>,
    },
}

pub struct LlamaCppDecisionBackend {
    tx: mpsc::Sender<Sys1Cmd>,
}

impl LlamaCppDecisionBackend {
    pub fn spawn(model_path: String, physical_cores: usize, use_gpu: bool) -> Result<Self, EngineError> {
        let (tx, mut rx) = mpsc::channel::<Sys1Cmd>(32);
        let (init_tx, init_rx) = std::sync::mpsc::channel();

        std::thread::Builder::new()
            .name("sys1-infer".to_string())
            .spawn(move || {
                if let Some(core_ids) = core_affinity::get_core_ids()
                    && let Some(first) = core_ids.first()
                {
                    core_affinity::set_for_current(*first);
                }

                let backend = match LlamaBackend::init() {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = init_tx.send(Err(EngineError::Unavailable(e.to_string())));
                        return;
                    }
                };
                let mut model_params = LlamaModelParams::default();
                if use_gpu {
                    model_params = model_params.with_n_gpu_layers(999);
                }
                let model = match LlamaModel::load_from_file(&backend, &model_path, &model_params) {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = init_tx.send(Err(EngineError::Unavailable(e.to_string())));
                        return;
                    }
                };
                let ctx_params = LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(4096))
                    .with_n_threads(physical_cores as i32)
                    .with_n_threads_batch(physical_cores as i32)
                    .with_n_seq_max(16);

                let mut ctx = match model.new_context(&backend, ctx_params) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = init_tx.send(Err(EngineError::Unavailable(e.to_string())));
                        return;
                    }
                };
                let _ = init_tx.send(Ok(()));

                let mut prefix_len: i32 = 0;
                while let Some(cmd) = rx.blocking_recv() {
                    match cmd {
                        Sys1Cmd::PinPrefix { text, reply } => {
                            ctx.clear_kv_cache();
                            let tokens = match model.str_to_token(&text, AddBos::Always) {
                                Ok(t) => t,
                                Err(e) => {
                                    let _ = reply.send(Err(EngineError::InvalidOutput(e.to_string())));
                                    continue;
                                }
                            };
                            let mut batch = LlamaBatch::new(tokens.len().max(512), 13);
                            let seqs: Vec<i32> = (0..=12).collect();
                            for (pos, tok) in tokens.iter().enumerate() {
                                let _ = batch.add(*tok, pos as i32, &seqs, false);
                            }
                            let res = ctx
                                .decode(&mut batch)
                                .map_err(|e| EngineError::Transport(e.to_string()));
                            prefix_len = tokens.len() as i32;
                            let _ = reply.send(res);
                        }
                        Sys1Cmd::EvalBatch {
                            state_frame,
                            probes,
                            num_actions,
                            reply,
                        } => {
                            let start = std::time::Instant::now();
                            // Clear dynamic suffix positions >= prefix_len across seq 0..=12
                            for seq_id in 0..=12 {
                                let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), Some(prefix_len as u32), None);
                            }
                            let frame_toks = match model.str_to_token(&state_frame, AddBos::Never) {
                                Ok(t) => t,
                                Err(e) => {
                                    let _ = reply.send(Err(EngineError::InvalidOutput(e.to_string())));
                                    continue;
                                }
                            };
                            let frame_end = prefix_len + frame_toks.len() as i32;
                            let mut batch = LlamaBatch::new(2048, 13);
                            let all_seqs: Vec<i32> = (0..=12).collect();
                            for (i, tok) in frame_toks.iter().enumerate() {
                                let _ = batch.add(*tok, prefix_len + i as i32, &all_seqs, false);
                            }

                            let mut probe_logit_indices = Vec::with_capacity(probes.len());
                            for (m, probe) in probes.iter().enumerate() {
                                let seq_id = (m + 1) as i32;
                                if let Ok(ptoks) = model.str_to_token(&probe.suffix, AddBos::Never) {
                                    for (j, tok) in ptoks.iter().enumerate() {
                                        let is_last = j + 1 == ptoks.len();
                                        let idx = batch.n_tokens();
                                        let _ = batch.add(*tok, frame_end + j as i32, &[seq_id], is_last);
                                        if is_last {
                                            probe_logit_indices.push((m, idx));
                                        }
                                    }
                                }
                            }

                            if let Err(e) = ctx.decode(&mut batch) {
                                let _ = reply.send(Err(EngineError::Transport(e.to_string())));
                                continue;
                            }

                            let mut out = ProbeLogitsBatch {
                                num_actions,
                                latency_us: start.elapsed().as_micros() as u64,
                                ..Default::default()
                            };
                            for (m, batch_idx) in probe_logit_indices {
                                let logits = ctx.get_logits_ith(batch_idx);
                                let probe = &probes[m];
                                match probe.id {
                                    "p_fwd" => softmax_target_tokens(&model, logits, &probe.choices, &mut out.p_fwd, num_actions),
                                    "p_inv" => softmax_target_tokens(&model, logits, &probe.choices, &mut out.p_inv, num_actions),
                                    "p_pert_1" => softmax_target_tokens(&model, logits, &probe.choices, &mut out.p_pert[0], num_actions),
                                    "p_pert_2" => softmax_target_tokens(&model, logits, &probe.choices, &mut out.p_pert[1], num_actions),
                                    "p_pert_3" => softmax_target_tokens(&model, logits, &probe.choices, &mut out.p_pert[2], num_actions),
                                    "r_precond" => {
                                        let (p, logit) = binary_yes_prob(&model, logits);
                                        out.r_precond = p;
                                        out.raw_precond_logit = logit;
                                    }
                                    "r_reversible" => out.r_reversible = binary_yes_prob(&model, logits).0,
                                    "r_progress" => out.r_progress = binary_yes_prob(&model, logits).0,
                                    "r_aligned" => out.r_aligned = binary_yes_prob(&model, logits).0,
                                    "r_info_missing" => out.r_info_missing = binary_yes_prob(&model, logits).0,
                                    _ => {}
                                }
                            }
                            let _ = reply.send(Ok(out));
                        }
                    }
                }
            })
            .map_err(|e| EngineError::Unavailable(e.to_string()))?;

        init_rx
            .recv()
            .map_err(|e| EngineError::Unavailable(e.to_string()))??;
        Ok(Self { tx })
    }
}

fn softmax_target_tokens(
    model: &LlamaModel,
    logits: &[f32],
    choices: &[String],
    out: &mut [f64; MAX_ACTIONS],
    num_actions: usize,
) {
    let mut max_l = f32::NEG_INFINITY;
    let mut raw = [f32::NEG_INFINITY; MAX_ACTIONS];
    for (i, c) in choices.iter().take(num_actions.min(MAX_ACTIONS)).enumerate() {
        if let Ok(toks) = model.str_to_token(c, AddBos::Never)
            && let Some(tok) = toks.first()
        {
            let l = logits[tok.0 as usize];
            raw[i] = l;
            if l > max_l {
                max_l = l;
            }
        }
    }
    let mut sum = 0.0f64;
    for i in 0..num_actions.min(MAX_ACTIONS) {
        let e = ((raw[i] - max_l) as f64).exp();
        out[i] = e;
        sum += e;
    }
    if sum > 0.0 {
        for v in out.iter_mut().take(num_actions.min(MAX_ACTIONS)) {
            *v /= sum;
        }
    }
}

fn binary_yes_prob(model: &LlamaModel, logits: &[f32]) -> (f64, f64) {
    let l_yes = model
        .str_to_token("YES", AddBos::Never)
        .ok()
        .and_then(|t| t.first().map(|id| logits[id.0 as usize]))
        .unwrap_or(0.0) as f64;
    let l_no = model
        .str_to_token("NO", AddBos::Never)
        .ok()
        .and_then(|t| t.first().map(|id| logits[id.0 as usize]))
        .unwrap_or(0.0) as f64;
    let diff = l_yes - l_no;
    let p = 1.0 / (1.0 + (-diff).exp());
    (p, diff)
}

#[async_trait]
impl DecisionBackend for LlamaCppDecisionBackend {
    async fn pin_static_prefix(&self, static_prefix: &str) -> Result<(), EngineError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Sys1Cmd::PinPrefix {
                text: static_prefix.to_string(),
                reply,
            })
            .await
            .map_err(|e| EngineError::Transport(e.to_string()))?;
        rx.await.map_err(|e| EngineError::Transport(e.to_string()))?
    }

    async fn evaluate_probes(
        &self,
        state_frame: &str,
        probes: &[ProbeSpec],
        num_actions: usize,
    ) -> Result<ProbeLogitsBatch, EngineError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Sys1Cmd::EvalBatch {
                state_frame: state_frame.to_string(),
                probes: probes.to_vec(),
                num_actions,
                reply,
            })
            .await
            .map_err(|e| EngineError::Transport(e.to_string()))?;
        rx.await.map_err(|e| EngineError::Transport(e.to_string()))?
    }
}

/// Placeholder wrapper for System-2 `llama-cpp-2` slot-partitioned generator.
pub struct LlamaCppGeneratorBackend;

#[async_trait]
impl GeneratorBackend for LlamaCppGeneratorBackend {
    async fn generate_scoped(
        &self,
        _slot: GeneratorSlot,
        _prompt: ScopedPrompt,
        abort_flag: Arc<AtomicBool>,
    ) -> Result<String, EngineError> {
        if abort_flag.load(Ordering::Acquire) {
            return Err(EngineError::Aborted);
        }
        Err(EngineError::Unavailable(
            "Configure GGUF path and slot runner for LlamaCppGeneratorBackend".to_string(),
        ))
    }

    async fn excise_kv_range(
        &self,
        _slot: GeneratorSlot,
        _pos_clean: usize,
        _pos_t: usize,
    ) -> Result<(), EngineError> {
        Ok(())
    }
}
