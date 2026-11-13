use crate::config::EvsiConfig;
use crate::engine::{MAX_ACTIONS, ProbeLogitsBatch};
use crate::probes::actions::CandidateAction;
use crate::stats::conformal::ConformalResult;
use crate::stats::dirichlet::DirichletSummary;
use rand::Rng;
use rustc_hash::FxHashMap;
use statrs::distribution::Beta;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteDecision {
    FastPath { action_idx: usize },
    Speculative { action_idx: usize },
    System2Escalate { reason: &'static str },
    AskUser { reason: &'static str },
}

/// EVSI / Bayes-Risk router + Beta-Binomial tool reliability + Platt scaling + SPRT + Thompson Sampling.
#[derive(Debug, Clone)]
pub struct BayesRouter {
    pub evsi: EvsiConfig,
    pub tau_ood: f64,
    /// Discounted Beta(alpha, beta) reliability per tool name
    tool_reliability: FxHashMap<String, (f64, f64)>,
    /// Online inverse temperature for Noul Platt scaling (beta = 1/T)
    pub platt_beta: f64,
    /// SPRT log-likelihood ratio for hopeless retry detection
    pub sprt_llr: f64,
    pub sprt_upper: f64,
    pub sprt_lower: f64,
    /// Per-action Beta counts for Thompson Sampling during sandbox recovery
    recovery_betas: [(f64, f64); MAX_ACTIONS],
}

impl BayesRouter {
    pub fn new(evsi: EvsiConfig, tau_ood: f64, sprt_alpha: f64, sprt_beta: f64) -> Self {
        let upper = ((1.0 - sprt_beta) / sprt_alpha.max(1e-6)).ln();
        let lower = (sprt_beta.max(1e-6) / (1.0 - sprt_alpha)).ln();
        Self {
            evsi,
            tau_ood,
            tool_reliability: FxHashMap::default(),
            platt_beta: 1.0,
            sprt_llr: 0.0,
            sprt_upper: upper,
            sprt_lower: lower,
            recovery_betas: [(2.0, 2.0); MAX_ACTIONS],
        }
    }

    pub fn calibrated_prob(&self, raw_logit: f64) -> f64 {
        1.0 / (1.0 + (-self.platt_beta * raw_logit).exp())
    }

    pub fn update_platt(&mut self, raw_logit: f64, outcome_ok: bool, lr: f64) {
        let p = self.calibrated_prob(raw_logit);
        let y = if outcome_ok { 1.0 } else { 0.0 };
        self.platt_beta = (self.platt_beta - lr * (p - y) * raw_logit).clamp(0.33, 2.0);
    }

    pub fn tool_reliability(&self, tool: &str) -> f64 {
        self.tool_reliability
            .get(tool)
            .map(|&(a, b)| a / (a + b))
            .unwrap_or(0.95)
    }

    pub fn update_tool_reliability(&mut self, tool: &str, ok: bool, decay: f64) {
        let entry = self
            .tool_reliability
            .entry(tool.to_string())
            .or_insert((9.5, 0.5));
        entry.0 = decay * entry.0 + if ok { 1.0 } else { 0.0 };
        entry.1 = decay * entry.1 + if ok { 0.0 } else { 1.0 };
    }

    /// Update Wald's SPRT on binary `r_progress > 0.5` signal; returns true if hopeless loop detected.
    pub fn update_sprt_progress(&mut self, made_progress: bool) -> bool {
        let (p1, p0) = (0.80f64, 0.30f64);
        let delta = if made_progress {
            (p1 / p0).ln()
        } else {
            ((1.0 - p1) / (1.0 - p0)).ln()
        };
        self.sprt_llr = (self.sprt_llr + delta).clamp(self.sprt_lower - 1.0, self.sprt_upper + 1.0);
        self.sprt_llr <= self.sprt_lower
    }

    pub fn reset_sprt(&mut self) {
        self.sprt_llr = 0.0;
    }

    /// Choose action via EVSI / Bayes-Risk + Conformal Set C(S).
    pub fn route(
        &self,
        batch: &ProbeLogitsBatch,
        dir: &DirichletSummary,
        conf: &ConformalResult,
        candidates: &[CandidateAction],
    ) -> RouteDecision {
        if dir.contradiction_flagged {
            return RouteDecision::System2Escalate {
                reason: "forward_inverse_contradiction",
            };
        }
        if dir.d_js >= self.tau_ood || conf.drift_alarm {
            return RouteDecision::System2Escalate {
                reason: "ood_or_calibration_drift",
            };
        }

        let top1 = dir.top1_idx;
        let cand = &candidates[top1];
        if cand.requires_system2 {
            return RouteDecision::System2Escalate {
                reason: "synthesis_slot_selected",
            };
        }

        let p_hat = dir.posterior_mean[top1];
        let p_rev = if cand.reversible {
            batch.r_reversible
        } else {
            0.0
        };
        let p_miss = batch.r_info_missing;
        let cal_precond = self.calibrated_prob(batch.raw_precond_logit) * self.tool_reliability(&cand.description);

        // Expected error cost in seconds
        let c_base = 5.0;
        let c_err = c_base
            * (1.0 + self.evsi.lambda_severity * batch.severity_score)
            * (1.0 + self.evsi.lambda_irrev * (1.0 - p_rev));
        let r_exec = c_err * (1.0 - p_hat);

        let net_gain_s2 = self.evsi.q_s2_accuracy * (1.0 - p_miss) * r_exec - self.evsi.cost_s2_seconds;
        let net_gain_ask = p_miss * r_exec - self.evsi.cost_ask_seconds;

        if net_gain_ask > net_gain_s2 && net_gain_ask > 0.0 {
            return RouteDecision::AskUser {
                reason: "evsi_missing_user_intent",
            };
        }

        let set_len = conf.prediction_set.len();
        if set_len == 1 && cal_precond > 0.80 && net_gain_s2 <= 0.0 {
            return RouteDecision::FastPath { action_idx: top1 };
        }

        if (2..=3).contains(&set_len) && p_rev > 0.90 && cand.reversible {
            return RouteDecision::Speculative { action_idx: top1 };
        }

        RouteDecision::System2Escalate {
            reason: "evsi_or_large_conformal_set",
        }
    }

    /// Thompson Sampling over candidate indices in C(S) during reversible sandbox recovery.
    pub fn thompson_pick(&self, set: &[usize]) -> usize {
        let mut rng = rand::thread_rng();
        let mut best_idx = set[0];
        let mut best_sample = f64::NEG_INFINITY;
        for &idx in set {
            let (a, b) = self.recovery_betas[idx.min(MAX_ACTIONS - 1)];
            let sample = Beta::new(a.max(0.1), b.max(0.1))
                .map(|d| rng.sample(d))
                .unwrap_or(0.5);
            if sample > best_sample {
                best_sample = sample;
                best_idx = idx;
            }
        }
        best_idx
    }

    pub fn update_thompson(&mut self, idx: usize, ok: bool) {
        let slot = &mut self.recovery_betas[idx.min(MAX_ACTIONS - 1)];
        slot.0 = 0.9 * slot.0 + if ok { 1.0 } else { 0.0 };
        slot.1 = 0.9 * slot.1 + if ok { 0.0 } else { 1.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CalibrationData;
    use crate::engine::mock::MockDecisionBackend;
    use crate::probes::actions::ActionKind;
    use crate::sandbox::traits::RiskGroup;
    use crate::stats::conformal::ConformalEstimator;
    use crate::stats::dirichlet::DirichletEstimator;

    #[test]
    fn routes_confident_discrete_to_fastpath() {
        let dir_est = DirichletEstimator::new(1.5);
        let cal = CalibrationData::default();
        let mut conf_est = ConformalEstimator::from_calibration(&cal, 0.10, 0.03, 0.01, 0.40);
        let router = BayesRouter::new(EvsiConfig {
            cost_s2_seconds: 1.2,
            cost_ask_seconds: 15.0,
            q_s2_accuracy: 0.85,
            lambda_severity: 1.0,
            lambda_irrev: 2.0,
        }, 0.25, 0.01, 0.01);

        let batch = MockDecisionBackend::confident_batch(2, 0, true);
        let cands = vec![
            CandidateAction {
                label: "A".to_string(),
                description: "read_file(src/lib.rs)".to_string(),
                kind: ActionKind::EntityBound {
                    tool: "read_file".to_string(),
                    target: "src/lib.rs".to_string(),
                },
                risk_group: RiskGroup::PureRead,
                reversible: true,
                requires_system2: false,
            },
            CandidateAction {
                label: "B".to_string(),
                description: "DONE".to_string(),
                kind: ActionKind::MetaDone,
                risk_group: RiskGroup::PureRead,
                reversible: true,
                requires_system2: false,
            },
        ];
        let dir = dir_est.evaluate(&batch);
        let conf = conf_est.build_prediction_set(&dir.posterior_mean[..2], &cands, dir.d_js, 0.25);
        let decision = router.route(&batch, &dir, &conf, &cands);
        assert_eq!(decision, RouteDecision::FastPath { action_idx: 0 });
    }
}
