use crate::engine::{MAX_ACTIONS, ProbeLogitsBatch};

/// Zero-allocation Dirichlet & divergence estimator over stack arrays.
#[derive(Debug, Clone)]
pub struct DirichletEstimator {
    /// Weights for [p_fwd, p_pert_1, p_pert_2, p_pert_3, p_inv_comp]
    pub probe_weights: [f64; 5],
    pub w_pert_max: f64,
}

#[derive(Debug, Clone)]
pub struct DirichletSummary {
    pub num_actions: usize,
    pub d_js: f64,
    pub kappa_consistency: f64,
    pub contradiction_flagged: bool,
    pub posterior_mean: [f64; MAX_ACTIONS],
    pub epistemic_uncertainty: f64,
    pub top1_idx: usize,
}

impl DirichletEstimator {
    pub fn new(w_pert_max: f64) -> Self {
        Self {
            probe_weights: [1.5, 0.6, 0.6, 0.6, 0.7],
            w_pert_max,
        }
    }

    /// Evaluate JSD, forward-inverse consistency, and Dirichlet posterior in < 1 us.
    pub fn evaluate(&self, batch: &ProbeLogitsBatch) -> DirichletSummary {
        let k = batch.num_actions.clamp(1, MAX_ACTIONS);

        // 1. Mean JSD between p_fwd and p_pert[0..3]
        let mut jsd_sum = 0.0;
        for m in 0..3 {
            jsd_sum += jensen_shannon(&batch.p_fwd[..k], &batch.p_pert[m][..k]);
        }
        let d_js = jsd_sum / 3.0;

        // 2. Forward-inverse consistency: align p_fwd with normalized (1 - p_inv)
        let mut p_inv_comp = [0.0f64; MAX_ACTIONS];
        let mut inv_sum = 0.0;
        for i in 0..k {
            let v = (1.0 - batch.p_inv[i]).max(1e-6);
            p_inv_comp[i] = v;
            inv_sum += v;
        }
        if inv_sum > 0.0 {
            for v in p_inv_comp.iter_mut().take(k) {
                *v /= inv_sum;
            }
        }
        let kappa_consistency = 1.0 - jensen_shannon(&batch.p_fwd[..k], &p_inv_comp[..k]);
        let argmax_fwd = argmax(&batch.p_fwd[..k]);
        let argmax_inv = argmax(&batch.p_inv[..k]);
        let contradiction_flagged = k > 1 && argmax_fwd == argmax_inv && batch.p_inv[argmax_inv] > 0.35;

        // 3. Enforce correlated-probe weight cap on perturbed framings [1..=3]
        let mut weights = self.probe_weights;
        let pert_sum = weights[1] + weights[2] + weights[3];
        if pert_sum > self.w_pert_max && pert_sum > 0.0 {
            let scale = self.w_pert_max / pert_sum;
            weights[1] *= scale;
            weights[2] *= scale;
            weights[3] *= scale;
        }

        // 4. Dirichlet pseudo-counts alpha_i = 0.05 + sum_m w_m P^(m)(a_i)
        let mut alpha = [0.0f64; MAX_ACTIONS];
        let mut alpha_sum = 0.0;
        for i in 0..k {
            let a = 0.05
                + weights[0] * batch.p_fwd[i]
                + weights[1] * batch.p_pert[0][i]
                + weights[2] * batch.p_pert[1][i]
                + weights[3] * batch.p_pert[2][i]
                + weights[4] * p_inv_comp[i];
            alpha[i] = a;
            alpha_sum += a;
        }

        let mut posterior_mean = [0.0f64; MAX_ACTIONS];
        for i in 0..k {
            posterior_mean[i] = alpha[i] / alpha_sum;
        }

        let epistemic_uncertainty = (k as f64) / alpha_sum;
        let top1_idx = argmax(&posterior_mean[..k]);

        DirichletSummary {
            num_actions: k,
            d_js,
            kappa_consistency,
            contradiction_flagged,
            posterior_mean,
            epistemic_uncertainty,
            top1_idx,
        }
    }

    /// Online Hedge / multiplicative-weights update when true action `y_star` is observed.
    pub fn hedge_update(&mut self, batch: &ProbeLogitsBatch, y_star: usize, eta: f64) {
        if y_star >= batch.num_actions.min(MAX_ACTIONS) {
            return;
        }
        let inv_comp = (1.0 - batch.p_inv[y_star]).max(1e-4);
        let probs = [
            batch.p_fwd[y_star],
            batch.p_pert[0][y_star],
            batch.p_pert[1][y_star],
            batch.p_pert[2][y_star],
            inv_comp,
        ];
        let mut sum = 0.0;
        for m in 0..5 {
            self.probe_weights[m] *= (eta * probs[m].max(1e-4).ln()).exp();
            sum += self.probe_weights[m];
        }
        if sum > 0.0 {
            let target_sum = 4.0;
            for w in &mut self.probe_weights {
                *w = (*w / sum) * target_sum;
            }
        }
    }
}

pub fn jensen_shannon(p: &[f64], q: &[f64]) -> f64 {
    let mut jsd = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let p_c = pi.max(1e-12);
        let q_c = qi.max(1e-12);
        let m = 0.5 * (p_c + q_c);
        jsd += 0.5 * (p_c * (p_c / m).ln() + q_c * (q_c / m).ln());
    }
    jsd.max(0.0)
}

pub fn argmax(xs: &[f64]) -> usize {
    let mut best_i = 0;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mock::MockDecisionBackend;

    #[test]
    fn detects_ood_and_contradiction() {
        let est = DirichletEstimator::new(1.5);
        let clean = MockDecisionBackend::confident_batch(4, 0, true);
        let clean_res = est.evaluate(&clean);
        assert!(clean_res.d_js < 0.01);
        assert!(!clean_res.contradiction_flagged);
        assert_eq!(clean_res.top1_idx, 0);

        let drift = MockDecisionBackend::drifting_batch(4);
        let drift_res = est.evaluate(&drift);
        assert!(drift_res.d_js > 0.10);
        assert!(drift_res.contradiction_flagged);
    }
}
