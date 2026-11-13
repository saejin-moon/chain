use statrs::distribution::{Continuous, StudentsT};

const MAX_RUN_LEN: usize = 64;

#[derive(Debug, Clone)]
struct StudentTParams {
    mu: f64,
    kappa: f64,
    alpha: f64,
    beta: f64,
}

impl Default for StudentTParams {
    fn default() -> Self {
        Self {
            mu: 0.0,
            kappa: 1.0,
            alpha: 1.0,
            beta: 0.02,
        }
    }
}

impl StudentTParams {
    fn predictive_pdf(&self, x: f64) -> f64 {
        let df = 2.0 * self.alpha;
        let scale = (self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa)).sqrt();
        StudentsT::new(self.mu, scale.max(1e-4), df)
            .map(|d| d.pdf(x).max(1e-12))
            .unwrap_or(1e-6)
    }

    fn update(&self, x: f64) -> Self {
        let kappa_n = self.kappa + 1.0;
        let mu_n = (self.kappa * self.mu + x) / kappa_n;
        let alpha_n = self.alpha + 0.5;
        let beta_n = self.beta + 0.5 * (self.kappa * (x - self.mu).powi(2)) / kappa_n;
        Self {
            mu: mu_n,
            kappa: kappa_n,
            alpha: alpha_n,
            beta: beta_n,
        }
    }
}

/// Bayesian Online Changepoint Detection (Adams & MacKay, Student-t conjugate prior)
/// over 3D telemetry: [D_JS, 1 - r_progress, 1 - r_aligned].
#[derive(Debug, Clone)]
pub struct BocpdDetector {
    pub hazard: f64,
    pub trigger_prob: f64,
    run_probs: Vec<f64>,
    hypotheses: Vec<[StudentTParams; 3]>,
}

#[derive(Debug, Clone)]
pub struct BocpdUpdate {
    pub short_run_prob: f64,
    pub changepoint_triggered: bool,
    pub map_run_length: usize,
}

impl BocpdDetector {
    pub fn new(hazard: f64, trigger_prob: f64) -> Self {
        Self {
            hazard,
            trigger_prob,
            run_probs: vec![1.0],
            hypotheses: vec![[
                StudentTParams::default(),
                StudentTParams::default(),
                StudentTParams::default(),
            ]],
        }
    }

    pub fn reset(&mut self) {
        self.run_probs.clear();
        self.run_probs.push(1.0);
        self.hypotheses.clear();
        self.hypotheses.push([
            StudentTParams::default(),
            StudentTParams::default(),
            StudentTParams::default(),
        ]);
    }

    pub fn step(&mut self, x: [f64; 3]) -> BocpdUpdate {
        let n = self.run_probs.len();
        let mut pred_probs = Vec::with_capacity(n);
        for h in &self.hypotheses {
            let p = h[0].predictive_pdf(x[0]) * h[1].predictive_pdf(x[1]) * h[2].predictive_pdf(x[2]);
            pred_probs.push(p);
        }

        let mut next_probs = vec![0.0; (n + 1).min(MAX_RUN_LEN)];
        let mut growth_mass = 0.0;
        let mut cp_mass = 0.0;

        for r in 0..n {
            let joint = self.run_probs[r] * pred_probs[r];
            let g = joint * (1.0 - self.hazard);
            let c = joint * self.hazard;
            cp_mass += c;
            if r + 1 < next_probs.len() {
                next_probs[r + 1] += g;
            } else {
                next_probs[MAX_RUN_LEN - 1] += g;
            }
            growth_mass += g;
        }
        next_probs[0] = cp_mass;

        let total = growth_mass + cp_mass;
        if total > 0.0 {
            for p in &mut next_probs {
                *p /= total;
            }
        } else {
            next_probs[0] = 1.0;
        }

        let mut next_hyps = Vec::with_capacity(next_probs.len());
        next_hyps.push([
            StudentTParams::default().update(x[0]),
            StudentTParams::default().update(x[1]),
            StudentTParams::default().update(x[2]),
        ]);
        for h in self.hypotheses.iter().take(next_probs.len() - 1) {
            next_hyps.push([h[0].update(x[0]), h[1].update(x[1]), h[2].update(x[2])]);
        }

        self.run_probs = next_probs;
        self.hypotheses = next_hyps;

        // P(r_t < 3) indicates abrupt regime shift into drift/looping
        let short_run_prob: f64 = self.run_probs.iter().take(3).sum();
        let map_run_length = self
            .run_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);

        BocpdUpdate {
            short_run_prob,
            changepoint_triggered: self.run_probs.len() > 3 && short_run_prob > self.trigger_prob,
            map_run_length,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bocpd_triggers_on_abrupt_drift_sequence() {
        let mut det = BocpdDetector::new(0.05, 0.75);
        for _ in 0..8 {
            let u = det.step([0.02, 0.05, 0.03]);
            assert!(!u.changepoint_triggered);
        }
        // Inject abrupt regime shift (high D_JS, stalled progress, misaligned)
        let mut triggered = false;
        for _ in 0..3 {
            let u = det.step([0.65, 0.90, 0.85]);
            if u.changepoint_triggered {
                triggered = true;
            }
        }
        assert!(triggered, "BOCPD should detect abrupt context-rot regime shift");
    }
}
