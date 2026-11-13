use super::{
    DecisionBackend, EngineError, GeneratorBackend, GeneratorSlot, MAX_ACTIONS, ProbeLogitsBatch,
    ProbeSpec, ScopedPrompt,
};
use crate::config::HttpApiFormat;
use async_trait::async_trait;
use reqwest::Client;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

/// HTTP client supporting:
/// - `SystemOne` (`/v1/systemone/batch` for `baselabs/joey-1` sidecar)
/// - `LlamaServer` (`/completion` for `llama-server`)
pub struct HttpDecisionBackend {
    client: Client,
    endpoint: String,
    api_format: HttpApiFormat,
    pinned_prefix: Mutex<String>,
}

impl HttpDecisionBackend {
    pub fn new(endpoint: impl Into<String>, api_format: HttpApiFormat) -> Self {
        Self {
            client: Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            api_format,
            pinned_prefix: Mutex::new(String::new()),
        }
    }
}

#[derive(Serialize)]
struct CompletionReq<'a> {
    prompt: &'a str,
    n_predict: usize,
    n_probs: usize,
    cache_prompt: bool,
    temperature: f32,
}

#[derive(Deserialize)]
struct CompletionResp {
    #[serde(default)]
    completion_probabilities: Vec<TokenProbEntry>,
}

#[derive(Deserialize)]
struct TokenProbEntry {
    #[serde(default)]
    probs: Vec<TopLogprob>,
}

#[derive(Deserialize)]
struct TopLogprob {
    tok_str: String,
    prob: f64,
}

#[derive(Serialize)]
struct SystemOneBatchReq<'a> {
    prefix: &'a str,
    state_frame: &'a str,
    probes: Vec<SystemOneProbeReq<'a>>,
}

#[derive(Serialize)]
struct SystemOneProbeReq<'a> {
    id: &'static str,
    suffix: &'a str,
    choices: &'a [String],
}

#[derive(Deserialize)]
struct SystemOneBatchResp {
    results: FxHashMap<String, Vec<f64>>,
}

#[async_trait]
impl DecisionBackend for HttpDecisionBackend {
    async fn pin_static_prefix(&self, static_prefix: &str) -> Result<(), EngineError> {
        *self.pinned_prefix.lock().unwrap() = static_prefix.to_string();
        Ok(())
    }

    async fn evaluate_probes(
        &self,
        state_frame: &str,
        probes: &[ProbeSpec],
        num_actions: usize,
    ) -> Result<ProbeLogitsBatch, EngineError> {
        let start = Instant::now();
        let prefix = self.pinned_prefix.lock().unwrap().clone();
        let mut batch = ProbeLogitsBatch {
            num_actions,
            ..Default::default()
        };

        if self.api_format == HttpApiFormat::SystemOne {
            let req = SystemOneBatchReq {
                prefix: &prefix,
                state_frame,
                probes: probes
                    .iter()
                    .map(|p| SystemOneProbeReq {
                        id: p.id,
                        suffix: &p.suffix,
                        choices: &p.choices,
                    })
                    .collect(),
            };
            let url = format!("{}/v1/systemone/batch", self.endpoint);
            let resp = self
                .client
                .post(&url)
                .json(&req)
                .send()
                .await
                .map_err(|e| EngineError::Transport(e.to_string()))?;
            let parsed: SystemOneBatchResp = resp
                .json()
                .await
                .map_err(|e| EngineError::InvalidOutput(e.to_string()))?;

            for (id, probs) in parsed.results {
                match id.as_str() {
                    "p_fwd" => copy_slice(&probs, &mut batch.p_fwd, num_actions),
                    "p_inv" => copy_slice(&probs, &mut batch.p_inv, num_actions),
                    "p_pert_1" => copy_slice(&probs, &mut batch.p_pert[0], num_actions),
                    "p_pert_2" => copy_slice(&probs, &mut batch.p_pert[1], num_actions),
                    "p_pert_3" => copy_slice(&probs, &mut batch.p_pert[2], num_actions),
                    "r_precond" => {
                        batch.r_precond = probs.first().copied().unwrap_or(0.5);
                        let p = batch.r_precond.clamp(1e-4, 1.0 - 1e-4);
                        batch.raw_precond_logit = (p / (1.0 - p)).ln();
                    }
                    "r_reversible" => batch.r_reversible = probs.first().copied().unwrap_or(0.5),
                    "r_progress" => batch.r_progress = probs.first().copied().unwrap_or(0.5),
                    "r_aligned" => batch.r_aligned = probs.first().copied().unwrap_or(0.5),
                    "r_info_missing" => batch.r_info_missing = probs.first().copied().unwrap_or(0.1),
                    "score_severity" => {
                        let mut w = 0.0;
                        let mut tot = 0.0;
                        for (lvl, &p) in probs.iter().enumerate() {
                            w += (lvl as f64 / (probs.len().max(2) - 1) as f64) * p;
                            tot += p;
                        }
                        batch.severity_score = if tot > 0.0 { w / tot } else { 0.25 };
                    }
                    _ => {}
                }
            }
            batch.latency_us = start.elapsed().as_micros() as u64;
            return Ok(batch);
        }

        let futs = probes.iter().map(|probe| {
            let full_prompt = format!("{prefix}\n{state_frame}\n{}", probe.suffix);
            let url = format!("{}/completion", self.endpoint);
            let client = self.client.clone();
            async move {
                let req = CompletionReq {
                    prompt: &full_prompt,
                    n_predict: 1,
                    n_probs: 16,
                    cache_prompt: true,
                    temperature: 0.0,
                };
                let resp = client
                    .post(&url)
                    .json(&req)
                    .send()
                    .await
                    .map_err(|e| EngineError::Transport(e.to_string()))?;
                let parsed: CompletionResp = resp
                    .json()
                    .await
                    .map_err(|e| EngineError::InvalidOutput(e.to_string()))?;
                Ok::<_, EngineError>((probe.id, parsed))
            }
        });

        let mut results = Vec::with_capacity(probes.len());
        for fut in futs {
            results.push(fut.await?);
        }

        for (idx, (id, resp)) in results.into_iter().enumerate() {
            let top = resp
                .completion_probabilities
                .first()
                .map(|e| e.probs.as_slice())
                .unwrap_or(&[]);
            match id {
                "p_fwd" => fill_choice_probs(top, &probes[idx].choices, &mut batch.p_fwd, num_actions),
                "p_inv" => fill_choice_probs(top, &probes[idx].choices, &mut batch.p_inv, num_actions),
                "p_pert_1" => fill_choice_probs(top, &probes[idx].choices, &mut batch.p_pert[0], num_actions),
                "p_pert_2" => fill_choice_probs(top, &probes[idx].choices, &mut batch.p_pert[1], num_actions),
                "p_pert_3" => fill_choice_probs(top, &probes[idx].choices, &mut batch.p_pert[2], num_actions),
                "r_precond" => {
                    batch.r_precond = extract_yes_prob(top);
                    let p = batch.r_precond.clamp(1e-4, 1.0 - 1e-4);
                    batch.raw_precond_logit = (p / (1.0 - p)).ln();
                }
                "r_reversible" => batch.r_reversible = extract_yes_prob(top),
                "r_progress" => batch.r_progress = extract_yes_prob(top),
                "r_aligned" => batch.r_aligned = extract_yes_prob(top),
                "r_info_missing" => batch.r_info_missing = extract_yes_prob(top),
                "score_severity" => batch.severity_score = extract_severity_score(top),
                _ => {}
            }
        }

        batch.latency_us = start.elapsed().as_micros() as u64;
        Ok(batch)
    }
}

fn copy_slice(src: &[f64], dst: &mut [f64; MAX_ACTIONS], num_actions: usize) {
    let n = src.len().min(num_actions).min(MAX_ACTIONS);
    dst[..n].copy_from_slice(&src[..n]);
}

fn fill_choice_probs(
    probs: &[TopLogprob],
    choices: &[String],
    out: &mut [f64; MAX_ACTIONS],
    num_actions: usize,
) {
    let mut sum = 0.0;
    for (i, choice) in choices.iter().take(num_actions.min(MAX_ACTIONS)).enumerate() {
        let p = probs
            .iter()
            .find(|t| t.tok_str.trim().eq_ignore_ascii_case(choice.trim()))
            .map(|t| t.prob)
            .unwrap_or(1e-4);
        out[i] = p;
        sum += p;
    }
    if sum > 0.0 {
        for v in out.iter_mut().take(num_actions.min(MAX_ACTIONS)) {
            *v /= sum;
        }
    }
}

fn extract_yes_prob(probs: &[TopLogprob]) -> f64 {
    let p_yes = probs
        .iter()
        .find(|t| t.tok_str.trim().eq_ignore_ascii_case("YES"))
        .map(|t| t.prob)
        .unwrap_or(1e-4);
    let p_no = probs
        .iter()
        .find(|t| t.tok_str.trim().eq_ignore_ascii_case("NO"))
        .map(|t| t.prob)
        .unwrap_or(1e-4);
    p_yes / (p_yes + p_no)
}

fn extract_severity_score(probs: &[TopLogprob]) -> f64 {
    let mut weighted = 0.0;
    let mut total = 0.0;
    for level in 0..=4 {
        let key = level.to_string();
        if let Some(t) = probs.iter().find(|t| t.tok_str.trim() == key) {
            weighted += (level as f64 / 4.0) * t.prob;
            total += t.prob;
        }
    }
    if total > 0.0 {
        weighted / total
    } else {
        0.25
    }
}

/// HTTP generator client supporting OpenAI `/v1/completions` (vLLM / SGLang)
/// and `llama-server` `/completion`.
pub struct HttpGeneratorBackend {
    client: Client,
    endpoint: String,
    model_name: String,
    api_format: HttpApiFormat,
}

impl HttpGeneratorBackend {
    pub fn new(
        endpoint: impl Into<String>,
        model_name: impl Into<String>,
        api_format: HttpApiFormat,
    ) -> Self {
        Self {
            client: Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            model_name: model_name.into(),
            api_format,
        }
    }
}

#[derive(Serialize)]
struct GenCompletionReq<'a> {
    prompt: &'a str,
    n_predict: usize,
    id_slot: usize,
    cache_prompt: bool,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    grammar: Option<&'a str>,
}

#[derive(Deserialize)]
struct GenCompletionResp {
    content: String,
}

#[derive(Serialize)]
struct OpenAiCompletionReq<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: usize,
    temperature: f32,
}

#[derive(Deserialize)]
struct OpenAiCompletionResp {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    text: String,
}

#[async_trait]
impl GeneratorBackend for HttpGeneratorBackend {
    async fn generate_scoped(
        &self,
        slot: GeneratorSlot,
        prompt: ScopedPrompt,
        abort_flag: Arc<AtomicBool>,
    ) -> Result<String, EngineError> {
        if abort_flag.load(Ordering::Acquire) {
            return Err(EngineError::Aborted);
        }
        let full_prompt = format!(
            "{}\n{}\n{}",
            prompt.tier0_static, prompt.tier1_working_state, prompt.tier2_task
        );

        if self.api_format == HttpApiFormat::OpenAi {
            let req = OpenAiCompletionReq {
                model: &self.model_name,
                prompt: &full_prompt,
                max_tokens: prompt.max_tokens,
                temperature: 0.1,
            };
            let url = format!("{}/v1/completions", self.endpoint);
            let resp = self
                .client
                .post(&url)
                .json(&req)
                .send()
                .await
                .map_err(|e| EngineError::Transport(e.to_string()))?;
            if abort_flag.load(Ordering::Acquire) {
                return Err(EngineError::Aborted);
            }
            let parsed: OpenAiCompletionResp = resp
                .json()
                .await
                .map_err(|e| EngineError::InvalidOutput(e.to_string()))?;
            return Ok(parsed
                .choices
                .into_iter()
                .next()
                .map(|c| c.text)
                .unwrap_or_default());
        }

        let req = GenCompletionReq {
            prompt: &full_prompt,
            n_predict: prompt.max_tokens,
            id_slot: slot as usize,
            cache_prompt: true,
            temperature: 0.1,
            grammar: prompt.gbnf_grammar.as_deref(),
        };
        let url = format!("{}/completion", self.endpoint);
        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| EngineError::Transport(e.to_string()))?;

        if abort_flag.load(Ordering::Acquire) {
            return Err(EngineError::Aborted);
        }
        let parsed: GenCompletionResp = resp
            .json()
            .await
            .map_err(|e| EngineError::InvalidOutput(e.to_string()))?;
        Ok(parsed.content)
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
