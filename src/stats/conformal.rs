use crate::config::CalibrationData;
use crate::engine::MAX_ACTIONS;
use crate::probes::actions::CandidateAction;
use crate::sandbox::traits::RiskGroup;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use statrs::distribution::{ChiSquared, ContinuousCDF};

pub const ACI_WINDOW: usize = 200;

/// Mondrian + Adaptive Conformal Inference (ACI) + Capped Martingale / Fisher drift detector.
#[derive(Debug, Clone)]
pub struct ConformalEstimator {
    pub target_alpha: f64,
    pub alpha_t: f64,
    pub gamma: f64,
    pub alpha_min: f64,
    pub alpha_max: f64,
    pub shrinkage_m: usize,
    pub global_scores: Vec<f64>,
    pub group_scores: FxHashMap<RiskGroup, Vec<f64>>,
    /// Pre-allocated ring buffer for online ACI quantile tracking (zero alloc)
    online_window: [f64; ACI_WINDOW],
    online_len: usize,
    online_cursor: usize,
    /// Capped power martingale & sliding p-value window for calibration drift
    pub martingale: f64,
    p_val_window: [f64; 16],
    p_val_len: usize,
    p_val_cursor: usize,
}

#[derive(Debug, Clone)]
pub struct ConformalResult {
    pub prediction_set: SmallVec<[usize; MAX_ACTIONS]>,
    pub q_hat_used: f64,
    pub conformal_p_value: f64,
    pub drift_alarm: bool,
}

impl ConformalEstimator {
    pub fn from_calibration(
        cal: &CalibrationData,
        target_alpha: f64,
        gamma: f64,
        alpha_min: f64,
        alpha_max: f64,
    ) -> Self {
        let mut global_scores = cal.global_scores.clone();
        global_scores.sort_by(|a, b| a.total_cmp(b));

        let mut group_scores = FxHashMap::default();
        for (k, v) in &cal.group_scores {
            let group = match k.as_str() {
                "pure_read" => RiskGroup::PureRead,
                "fs_mutate" => RiskGroup::FsMutate,
                "shell" => RiskGroup::Shell,
                "network_api" => RiskGroup::NetworkApi,
                _ => continue,
            };
            let mut sorted = v.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            group_scores.insert(group, sorted);
        }

        Self {
            target_alpha,
            alpha_t: target_alpha,
            gamma,
            alpha_min,
            alpha_max,
            shrinkage_m: cal.shrinkage_m.max(1),
            global_scores,
            group_scores,
            online_window: [0.0; ACI_WINDOW],
            online_len: 0,
            online_cursor: 0,
            martingale: 1.0,
            p_val_window: [1.0; 16],
            p_val_len: 0,
            p_val_cursor: 0,
        }
    }

    /// Compute Mondrian group-conditional quantile with shrinkage toward global quantile.
    pub fn quantile_for_group(&self, group: RiskGroup) -> f64 {
        let q_global = sorted_quantile(&self.global_scores, 1.0 - self.alpha_t);
        if let Some(gs) = self.group_scores.get(&group) {
            let n_g = gs.len();
            if n_g == 0 {
                return q_global;
            }
            let q_g = sorted_quantile(gs, 1.0 - self.alpha_t);
            let m = self.shrinkage_m as f64;
            let n = n_g as f64;
            (n / (n + m)) * q_g + (m / (n + m)) * q_global
        } else {
            q_global
        }
    }

    /// Build split/Mondrian conformal prediction set C(S) and compute smoothed conformal p-value.
    pub fn build_prediction_set(
        &mut self,
        posterior_mean: &[f64],
        candidates: &[CandidateAction],
        d_js: f64,
        tau_ood: f64,
    ) -> ConformalResult {
        let k = posterior_mean.len().min(candidates.len()).min(MAX_ACTIONS);
        let mut set = SmallVec::<[usize; MAX_ACTIONS]>::new();
        let mut max_q = 0.0f64;

        let mut top1_prob = 1e-9f64;
        let mut second_prob = 0.0f64;
        for &p in posterior_mean.iter().take(k) {
            if p > top1_prob {
                second_prob = top1_prob;
                top1_prob = p;
            } else if p > second_prob {
                second_prob = p;
            }
        }
        // Non-conformity of top-1 is the ratio of runner-up to top-1 (0 = total certainty, 1 = tie)
        let top1_score = if k > 1 {
            (second_prob / top1_prob).clamp(0.0, 1.0)
        } else {
            0.01
        };

        for i in 0..k {
            // Relative non-conformity 1 - (p_i / p_top1) is scale-invariant across K=2..16 candidates:
            // - Distinct winner => |C(S)| == 1 (FastPath)
            // - 2-3 close contenders => |C(S)| in {2, 3} (Speculative)
            // - Flat/uncertain => |C(S)| > 3 (System2Escalate)
            let rel_nonconformity = 1.0 - (posterior_mean[i] / top1_prob);
            let q_hat = self.quantile_for_group(candidates[i].risk_group);
            max_q = max_q.max(q_hat);
            if d_js < tau_ood && rel_nonconformity <= q_hat {
                set.push(i);
            }
        }

        // Compute smoothed conformal p-value against global calibration buffer
        let p_val = self.smoothed_p_value(top1_score);
        self.p_val_window[self.p_val_cursor] = p_val.max(1e-6);
        self.p_val_cursor = (self.p_val_cursor + 1) % 16;
        if self.p_val_len < 16 {
            self.p_val_len += 1;
        }

        // Capped power martingale (epsilon = 0.7)
        let eps = 0.7;
        self.martingale = (self.martingale * eps * p_val.max(1e-6).powf(eps - 1.0)).max(1.0);

        let fisher_alarm = if self.p_val_len >= 8 {
            let chi_stat: f64 = -2.0
                * self.p_val_window[..self.p_val_len]
                    .iter()
                    .map(|p| p.ln())
                    .sum::<f64>();
            let df = (2 * self.p_val_len) as f64;
            ChiSquared::new(df)
                .map(|d| d.sf(chi_stat) < 0.01)
                .unwrap_or(false)
        } else {
            false
        };

        let drift_alarm = self.martingale > 20.0 || fisher_alarm;

        ConformalResult {
            prediction_set: set,
            q_hat_used: max_q,
            conformal_p_value: p_val,
            drift_alarm,
        }
    }

    /// Adaptive Conformal Inference (Gibbs & Candes, 2021) update after verifiable outcome.
    pub fn aci_update(&mut self, y_star_in_set: bool, realized_nonconformity: f64) {
        let err_t = if y_star_in_set { 0.0 } else { 1.0 };
        self.alpha_t = (self.alpha_t + self.gamma * (self.target_alpha - err_t))
            .clamp(self.alpha_min, self.alpha_max);

        self.online_window[self.online_cursor] = realized_nonconformity;
        self.online_cursor = (self.online_cursor + 1) % ACI_WINDOW;
        if self.online_len < ACI_WINDOW {
            self.online_len += 1;
        }
    }

    /// Zero-alloc online quantile over recent window using `select_nth_unstable_by`.
    pub fn online_window_quantile(&self) -> Option<f64> {
        if self.online_len < 10 {
            return None;
        }
        let mut scratch = [0.0f64; ACI_WINDOW];
        scratch[..self.online_len].copy_from_slice(&self.online_window[..self.online_len]);
        let idx = (((1.0 - self.alpha_t) * self.online_len as f64).ceil() as usize)
            .saturating_sub(1)
            .min(self.online_len - 1);
        let (_, val, _) = scratch[..self.online_len].select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
        Some(*val)
    }

    pub fn reset_episode_drift(&mut self) {
        self.martingale = 1.0;
        self.p_val_len = 0;
        self.p_val_cursor = 0;
    }

    fn smoothed_p_value(&self, s_t: f64) -> f64 {
        let n = self.global_scores.len();
        if n == 0 {
            return 0.5;
        }
        let greater = self.global_scores.iter().filter(|&&s| s > s_t).count();
        let equal = self.global_scores.iter().filter(|&&s| (s - s_t).abs() < 1e-12).count();
        (greater as f64 + 0.5 * (1 + equal) as f64) / (n + 1) as f64
    }
}

fn sorted_quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 1.0;
    }
    let n = sorted.len();
    let idx = ((q * (n + 1) as f64).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probes::actions::ActionKind;

    #[test]
    fn aci_adapts_alpha_on_miscoverage() {
        let cal = CalibrationData::default();
        let mut conf = ConformalEstimator::from_calibration(&cal, 0.10, 0.05, 0.01, 0.40);
        let init_alpha = conf.alpha_t;
        conf.aci_update(false, 0.90);
        assert!(conf.alpha_t < init_alpha);
        for _ in 0..5 {
            conf.aci_update(true, 0.05);
        }
        assert!(conf.alpha_t > init_alpha - 0.05);
    }

    #[test]
    fn mondrian_differentiates_read_vs_shell() {
        let cal = CalibrationData::default();
        let conf = ConformalEstimator::from_calibration(&cal, 0.10, 0.03, 0.01, 0.40);
        let q_read = conf.quantile_for_group(RiskGroup::PureRead);
        let q_shell = conf.quantile_for_group(RiskGroup::Shell);
        assert!(q_read <= q_shell);

        let mut conf_mut = conf;
        let cands = vec![CandidateAction {
            label: "A".to_string(),
            description: "read".to_string(),
            kind: ActionKind::MetaDone,
            risk_group: RiskGroup::PureRead,
            reversible: true,
            requires_system2: false,
        }];
        let res = conf_mut.build_prediction_set(&[0.95], &cands, 0.01, 0.25);
        assert_eq!(res.prediction_set.as_slice(), &[0]);
    }
}
