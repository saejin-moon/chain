use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use chain::config::{BackendKind, CalibrationData, RuntimeConfig};
use chain::engine::{
    DecisionBackend, GeneratorBackend,
    http::{HttpDecisionBackend, HttpGeneratorBackend},
    mock::{MockDecisionBackend, MockGeneratorBackend},
};
use chain::runtime::ChainRuntime;
use chain::sandbox::{
    fs_tools::{ReadFileTool, RunTestTool, WriteFileTool},
    traits::ToolRegistry,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
struct AppState {
    runtime: Arc<Mutex<ChainRuntime>>,
}

#[derive(Deserialize)]
struct ChatCompletionReq {
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default)]
    initial_files: Vec<String>,
    #[serde(default = "default_max_turns")]
    max_turns: usize,
}

fn default_max_turns() -> usize {
    20
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResp {
    id: String,
    object: &'static str,
    completed: bool,
    summary: String,
    turns: usize,
    fast_path_hits: usize,
    speculative_commits: usize,
    speculative_rollbacks: usize,
    system2_escalations: usize,
    bocpd_prunes: usize,
    trace: Vec<chain::runtime::TurnTrace>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("chain=info")
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("calibrate") {
        run_calibration_cli();
        return;
    }

    let config = RuntimeConfig::from_file("config/runtime.toml").unwrap_or_default();
    let calibration = CalibrationData::from_file("config/calibration.json").unwrap_or_default();

    let decision_backend: Arc<dyn DecisionBackend> = match config.decision_model.backend {
        BackendKind::Http => Arc::new(HttpDecisionBackend::new(
            &config.decision_model.endpoint,
            config.decision_model.api_format,
        )),
        BackendKind::Mock => Arc::new(MockDecisionBackend::new()),
        BackendKind::LlamaCpp => {
            #[cfg(feature = "llama-cpp")]
            {
                use chain::config::DeviceTarget;
                use chain::engine::llama_cpp_backend::LlamaCppDecisionBackend;
                Arc::new(
                    LlamaCppDecisionBackend::spawn(
                        config.decision_model.model_path.clone(),
                        config.decision_model.physical_cores,
                        config.decision_model.device == DeviceTarget::Gpu,
                    )
                    .expect("failed to initialize llama-cpp decision backend"),
                )
            }
            #[cfg(not(feature = "llama-cpp"))]
            {
                tracing::warn!("llama-cpp feature not enabled; falling back to MockDecisionBackend");
                Arc::new(MockDecisionBackend::new())
            }
        }
    };

    let generator_backend: Arc<dyn GeneratorBackend> = match config.generator_model.backend {
        BackendKind::Http => Arc::new(HttpGeneratorBackend::new(
            &config.generator_model.endpoint,
            &config.generator_model.model_path,
            config.generator_model.api_format,
        )),
        BackendKind::Mock => Arc::new(MockGeneratorBackend::new()),
        BackendKind::LlamaCpp => {
            #[cfg(feature = "llama-cpp")]
            {
                Arc::new(chain::engine::llama_cpp_backend::LlamaCppGeneratorBackend)
            }
            #[cfg(not(feature = "llama-cpp"))]
            {
                tracing::warn!("llama-cpp feature not enabled; falling back to MockGeneratorBackend");
                Arc::new(MockGeneratorBackend::new())
            }
        }
    };

    let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(ReadFileTool {
        workspace_root: workspace.clone(),
        max_digest_tokens: config.decision_model.max_digest_tokens,
    }));
    tools.register(Arc::new(WriteFileTool::new(
        workspace.clone(),
        config.decision_model.max_digest_tokens,
    )));
    tools.register(Arc::new(RunTestTool {
        workspace_root: workspace,
        max_digest_tokens: config.decision_model.max_digest_tokens,
    }));

    let bind_addr = config.server.bind_addr.clone();
    let runtime = Arc::new(Mutex::new(ChainRuntime::new(
        config,
        &calibration,
        decision_backend,
        generator_backend,
        tools,
    )));

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(AppState { runtime });

    tracing::info!("CHAIN proxy listening on http://{bind_addr}");
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .expect("failed to bind TCP listener");
    axum::serve(listener, app).await.expect("server error");
}

async fn handle_chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionReq>,
) -> Json<ChatCompletionResp> {
    let goal = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_else(|| "Complete user request".to_string());

    let mut rt = state.runtime.lock().await;
    let res = rt
        .run_episode(&goal, req.initial_files, req.max_turns)
        .await;

    Json(ChatCompletionResp {
        id: uuid::Uuid::now_v7().to_string(),
        object: "chain.episode.completion",
        completed: res.completed,
        summary: res.summary,
        turns: res.metrics.turns,
        fast_path_hits: res.metrics.fast_path_hits,
        speculative_commits: res.metrics.speculative_commits,
        speculative_rollbacks: res.metrics.speculative_rollbacks,
        system2_escalations: res.metrics.system2_escalations,
        bocpd_prunes: res.metrics.bocpd_prunes,
        trace: res.metrics.trace,
    })
}

fn run_calibration_cli() {
    let cal = CalibrationData::default();
    let json = serde_json::to_string_pretty(&cal).unwrap();
    std::fs::create_dir_all("config").unwrap();
    std::fs::write("config/calibration.json", json).unwrap();
    println!("Wrote calibrated quantiles to config/calibration.json");
}
