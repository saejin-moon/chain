use async_trait::async_trait;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskGroup {
    PureRead,
    FsMutate,
    Shell,
    NetworkApi,
}

/// Typed tool arguments on the hot path (avoids `serde_json::Value` map/string allocations).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolArgs {
    ReadFile { path: String },
    WriteFile { path: String, content: String },
    RunTest { test_name: String },
    Shell { command: String },
    Sql { query: String },
    RawJson(Box<RawValue>),
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub exit_code: i32,
    pub full_payload: String,
    pub sensor_digest: String,
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
    #[error("execution failed: {0}")]
    Execution(String),
}

#[derive(Debug, Error)]
pub enum RollbackError {
    #[error("transaction {0} not found")]
    NotFound(Uuid),
    #[error("rollback failed: {0}")]
    Failed(String),
}

#[async_trait]
pub trait ReversibleTool: Send + Sync {
    fn name(&self) -> &str;
    fn risk_group(&self) -> RiskGroup;
    fn is_pure_read(&self) -> bool {
        self.risk_group() == RiskGroup::PureRead
    }
    async fn execute(&self, args: ToolArgs, tx_id: Uuid) -> Result<ToolOutput, ToolError>;
    async fn commit(&self, _tx_id: Uuid) -> Result<(), RollbackError> {
        Ok(())
    }
    async fn rollback(&self, tx_id: Uuid) -> Result<(), RollbackError>;
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: FxHashMap<String, Arc<dyn ReversibleTool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: Arc<dyn ReversibleTool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn ReversibleTool>> {
        self.tools.get(name)
    }

    pub fn compact_summary(&self) -> String {
        let mut names: Vec<&str> = self.tools.keys().map(String::as_str).collect();
        names.sort_unstable();
        names.join(", ")
    }
}
