pub mod http;
#[cfg(feature = "llama-cpp")]
pub mod llama_cpp_backend;
pub mod mock;

use async_trait::async_trait;
use smallvec::SmallVec;
use std::sync::{Arc, atomic::AtomicBool};
use thiserror::Error;

pub const MAX_ACTIONS: usize = 16;
pub const NUM_PROBES: usize = 12;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("inference request aborted")]
    Aborted,
    #[error("backend communication error: {0}")]
    Transport(String),
    #[error("invalid model output: {0}")]
    InvalidOutput(String),
    #[error("backend unavailable: {0}")]
    Unavailable(String),
}

/// Slot partitioning for System-2 KV-cache isolation:
/// Slot 0 = Speculative verification (fast abort never invalidates Slot 1).
/// Slot 1 = Scoped synthesis / escalation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneratorSlot {
    SpeculativeVerify = 0,
    ScopedSynthesis = 1,
}

/// Output of a single batched 12-probe System-1 pass.
/// Stores normalized probability distributions over K <= MAX_ACTIONS choices
/// for forward/inverse/perturbed probes, plus scalar Noul/Score rubric probabilities.
#[derive(Debug, Clone)]
pub struct ProbeLogitsBatch {
    pub num_actions: usize,
    /// P_fwd over actions [0..num_actions]
    pub p_fwd: [f64; MAX_ACTIONS],
    /// P_inv over actions [0..num_actions] (higher = more likely to violate constraints)
    pub p_inv: [f64; MAX_ACTIONS],
    /// 3 perturbed framings P_pert^(1..3)
    pub p_pert: [[f64; MAX_ACTIONS]; 3],
    /// Binary Noul rubric probabilities
    pub r_precond: f64,
    pub r_reversible: f64,
    pub r_progress: f64,
    pub r_aligned: f64,
    pub r_info_missing: f64,
    /// Ordinal Score expectation for blast-radius severity in [0.0, 1.0]
    pub severity_score: f64,
    /// Raw logit for Noul calibration (Platt scaling)
    pub raw_precond_logit: f64,
    /// Measured prefill + probe latency in microseconds
    pub latency_us: u64,
}

impl Default for ProbeLogitsBatch {
    fn default() -> Self {
        Self {
            num_actions: 0,
            p_fwd: [0.0; MAX_ACTIONS],
            p_inv: [0.0; MAX_ACTIONS],
            p_pert: [[0.0; MAX_ACTIONS]; 3],
            r_precond: 0.0,
            r_reversible: 0.0,
            r_progress: 0.0,
            r_aligned: 0.0,
            r_info_missing: 0.0,
            severity_score: 0.0,
            raw_precond_logit: 0.0,
            latency_us: 0,
        }
    }
}

/// A single probe suffix to evaluate against the shared StateFrame prefix.
#[derive(Debug, Clone)]
pub struct ProbeSpec {
    pub id: &'static str,
    pub suffix: String,
    pub choices: SmallVec<[String; MAX_ACTIONS]>,
}

#[async_trait]
pub trait DecisionBackend: Send + Sync {
    /// Pin static prefix once per session (Tier 0).
    async fn pin_static_prefix(&self, static_prefix: &str) -> Result<(), EngineError>;

    /// Evaluate the dynamic StateFrame delta + 12 orthogonal probes in one batch.
    async fn evaluate_probes(
        &self,
        state_frame: &str,
        probes: &[ProbeSpec],
        num_actions: usize,
    ) -> Result<ProbeLogitsBatch, EngineError>;
}

#[derive(Debug, Clone)]
pub struct ScopedPrompt {
    pub tier0_static: Arc<str>,
    pub tier1_working_state: String,
    pub tier2_task: String,
    pub max_tokens: usize,
    pub gbnf_grammar: Option<String>,
}

#[async_trait]
pub trait GeneratorBackend: Send + Sync {
    /// Generate a scoped, grammar-constrained response on a dedicated slot.
    async fn generate_scoped(
        &self,
        slot: GeneratorSlot,
        prompt: ScopedPrompt,
        abort_flag: Arc<AtomicBool>,
    ) -> Result<String, EngineError>;

    /// In-place surgical KV excision for Module E (no-op on stateless HTTP backends).
    async fn excise_kv_range(
        &self,
        slot: GeneratorSlot,
        pos_clean: usize,
        pos_t: usize,
    ) -> Result<(), EngineError>;
}
