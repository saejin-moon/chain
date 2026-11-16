use super::{
    DecisionBackend, EngineError, GeneratorBackend, GeneratorSlot, MAX_ACTIONS, ProbeLogitsBatch,
    ProbeSpec, ScopedPrompt,
};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

/// Deterministic mock System-1 + System-2 backend for unit tests, simulation,
/// and calibration harness verification without needing local GGUF weights.
#[derive(Debug, Default)]
pub struct MockDecisionBackend {
    pub pinned_prefix: Mutex<Option<String>>,
    pub scripted_batches: Mutex<VecDeque<ProbeLogitsBatch>>,
}

impl MockDecisionBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_batch(&self, batch: ProbeLogitsBatch) {
        self.scripted_batches.lock().unwrap().push_back(batch);
    }

    /// Build a clean, high-confidence batch favoring `top_idx`.
    pub fn confident_batch(num_actions: usize, top_idx: usize, reversible: bool) -> ProbeLogitsBatch {
        let mut b = ProbeLogitsBatch {
            num_actions,
            r_precond: 0.94,
            r_reversible: if reversible { 0.96 } else { 0.05 },
            r_progress: 0.90,
            r_aligned: 0.95,
            r_info_missing: 0.03,
            severity_score: if reversible { 0.1 } else { 0.8 },
            raw_precond_logit: 2.75,
            latency_us: 18_000,
            ..Default::default()
        };
        let rem = if num_actions > 1 {
            0.06 / (num_actions - 1) as f64
        } else {
            0.0
        };
        for i in 0..num_actions.min(MAX_ACTIONS) {
            let p = if i == top_idx { 0.94 } else { rem };
            b.p_fwd[i] = p;
            for k in 0..3 {
                b.p_pert[k][i] = p;
            }
            b.p_inv[i] = if i == top_idx {
                0.02
            } else {
                0.98 / (num_actions.max(2) - 1) as f64
            };
        }
        b
    }

    /// Build an uncertain/ambiguous batch split across top 2 actions.
    pub fn ambiguous_batch(num_actions: usize, reversible: bool) -> ProbeLogitsBatch {
        let mut b = Self::confident_batch(num_actions, 0, reversible);
        if num_actions >= 2 {
            b.p_fwd[0] = 0.52;
            b.p_fwd[1] = 0.44;
            for k in 0..3 {
                b.p_pert[k][0] = 0.48 + (k as f64) * 0.02;
                b.p_pert[k][1] = 0.46 - (k as f64) * 0.02;
            }
        }
        b.r_precond = 0.88;
        b
    }

    /// Build a drifting / OOD batch to trigger BOCPD or System-2 escalation.
    pub fn drifting_batch(num_actions: usize) -> ProbeLogitsBatch {
        let mut b = Self::confident_batch(num_actions, 0, false);
        for i in 0..num_actions.min(MAX_ACTIONS) {
            let u = 1.0 / num_actions.max(1) as f64;
            b.p_fwd[i] = u;
            b.p_pert[0][i] = if i == 0 { 0.8 } else { 0.2 / (num_actions.max(2) - 1) as f64 };
            b.p_pert[1][i] = if i + 1 == num_actions { 0.8 } else { 0.2 / (num_actions.max(2) - 1) as f64 };
            b.p_pert[2][i] = u;
            b.p_inv[i] = if i == 0 { 0.7 } else { 0.3 / (num_actions.max(2) - 1) as f64 };
        }
        b.r_precond = 0.35;
        b.r_progress = 0.15;
        b.r_aligned = 0.25;
        b.r_info_missing = 0.45;
        b
    }
}

#[async_trait]
impl DecisionBackend for MockDecisionBackend {
    async fn pin_static_prefix(&self, static_prefix: &str) -> Result<(), EngineError> {
        *self.pinned_prefix.lock().unwrap() = Some(static_prefix.to_string());
        Ok(())
    }

    async fn evaluate_probes(
        &self,
        _state_frame: &str,
        _probes: &[ProbeSpec],
        num_actions: usize,
    ) -> Result<ProbeLogitsBatch, EngineError> {
        if let Some(mut batch) = self.scripted_batches.lock().unwrap().pop_front() {
            batch.num_actions = num_actions.max(1);
            return Ok(batch);
        }
        Ok(Self::confident_batch(num_actions.max(1), 0, true))
    }
}

#[derive(Debug, Default)]
pub struct MockGeneratorBackend {
    pub scripted_responses: Mutex<VecDeque<String>>,
    pub excised_ranges: Mutex<Vec<(GeneratorSlot, usize, usize)>>,
}

impl MockGeneratorBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_response(&self, resp: impl Into<String>) {
        self.scripted_responses
            .lock()
            .unwrap()
            .push_back(resp.into());
    }
}

#[async_trait]
impl GeneratorBackend for MockGeneratorBackend {
    async fn generate_scoped(
        &self,
        _slot: GeneratorSlot,
        _prompt: ScopedPrompt,
        abort_flag: Arc<AtomicBool>,
    ) -> Result<String, EngineError> {
        if abort_flag.load(Ordering::Acquire) {
            return Err(EngineError::Aborted);
        }
        if let Some(resp) = self.scripted_responses.lock().unwrap().pop_front() {
            return Ok(resp);
        }
        Ok(r#"{"verdict":"COMMIT","alt_choice":null,"reason":"verified"}"#.to_string())
    }

    async fn excise_kv_range(
        &self,
        slot: GeneratorSlot,
        pos_clean: usize,
        pos_t: usize,
    ) -> Result<(), EngineError> {
        self.excised_ranges
            .lock()
            .unwrap()
            .push((slot, pos_clean, pos_t));
        Ok(())
    }
}
