use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("I/O error reading config: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub server: ServerConfig,
    pub decision_model: DecisionModelConfig,
    pub generator_model: GeneratorModelConfig,
    pub stats: StatsConfig,
    pub evsi: EvsiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub bind_addr: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    LlamaCpp,
    Http,
    Mock,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeviceTarget {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionModelConfig {
    pub backend: BackendKind,
    pub model_path: String,
    pub endpoint: String,
    pub device: DeviceTarget,
    pub physical_cores: usize,
    pub kv_cache_type: String,
    pub max_state_frame_tokens: usize,
    pub max_digest_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorModelConfig {
    pub backend: BackendKind,
    pub model_path: String,
    pub endpoint: String,
    pub device: DeviceTarget,
    pub kv_cache_type: String,
    pub max_verify_tokens: usize,
    pub max_synthesis_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsConfig {
    pub target_alpha: f64,
    pub aci_gamma: f64,
    pub aci_alpha_min: f64,
    pub aci_alpha_max: f64,
    pub tau_ood: f64,
    pub tau_clean: f64,
    pub bocpd_hazard: f64,
    pub bocpd_trigger_prob: f64,
    pub w_pert_max: f64,
    pub sprt_alpha: f64,
    pub sprt_beta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvsiConfig {
    pub cost_s2_seconds: f64,
    pub cost_ask_seconds: f64,
    pub q_s2_accuracy: f64,
    pub lambda_severity: f64,
    pub lambda_irrev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    pub global_scores: Vec<f64>,
    pub group_scores: FxHashMap<String, Vec<f64>>,
    pub shrinkage_m: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                bind_addr: "127.0.0.1:8080".to_string(),
            },
            decision_model: DecisionModelConfig {
                backend: BackendKind::Mock,
                model_path: "models/AgentDecision-4B-Q4_K_M.gguf".to_string(),
                endpoint: "http://127.0.0.1:8081".to_string(),
                device: DeviceTarget::Cpu,
                physical_cores: 8,
                kv_cache_type: "q8_0".to_string(),
                max_state_frame_tokens: 500,
                max_digest_tokens: 120,
            },
            generator_model: GeneratorModelConfig {
                backend: BackendKind::Mock,
                model_path: "models/Qwen3-8B-Q4_K_M.gguf".to_string(),
                endpoint: "http://127.0.0.1:8082".to_string(),
                device: DeviceTarget::Gpu,
                kv_cache_type: "q8_0".to_string(),
                max_verify_tokens: 64,
                max_synthesis_tokens: 512,
            },
            stats: StatsConfig {
                target_alpha: 0.10,
                aci_gamma: 0.03,
                aci_alpha_min: 0.01,
                aci_alpha_max: 0.40,
                tau_ood: 0.25,
                tau_clean: 0.10,
                bocpd_hazard: 0.05,
                bocpd_trigger_prob: 0.75,
                w_pert_max: 1.5,
                sprt_alpha: 0.01,
                sprt_beta: 0.01,
            },
            evsi: EvsiConfig {
                cost_s2_seconds: 1.2,
                cost_ask_seconds: 15.0,
                q_s2_accuracy: 0.85,
                lambda_severity: 1.0,
                lambda_irrev: 2.0,
            },
        }
    }
}

impl Default for CalibrationData {
    fn default() -> Self {
        let mut group_scores = FxHashMap::default();
        group_scores.insert(
            "pure_read".to_string(),
            vec![0.01, 0.05, 0.10, 0.19, 0.30, 0.42],
        );
        group_scores.insert(
            "fs_mutate".to_string(),
            vec![0.03, 0.09, 0.16, 0.27, 0.48, 0.65],
        );
        group_scores.insert(
            "shell".to_string(),
            vec![0.05, 0.14, 0.25, 0.40, 0.68, 0.82],
        );
        group_scores.insert(
            "network_api".to_string(),
            vec![0.04, 0.13, 0.24, 0.39, 0.64, 0.79],
        );
        Self {
            global_scores: vec![
                0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.31, 0.40, 0.52, 0.72, 0.85,
            ],
            group_scores,
            shrinkage_m: 20,
        }
    }
}

impl RuntimeConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let raw = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&raw)?)
    }
}

impl CalibrationData {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let raw = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&raw)?)
    }
}
