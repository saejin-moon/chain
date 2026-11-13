use super::traits::{ReversibleTool, RiskGroup, RollbackError, ToolArgs, ToolError, ToolOutput};
use crate::probes::digest::compress_tool_output;
use async_trait::async_trait;
use rusqlite::Connection;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

enum SqlCmd {
    Exec {
        tx_id: Uuid,
        query: String,
        reply: oneshot::Sender<Result<String, String>>,
    },
    Commit {
        tx_id: Uuid,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Rollback {
        tx_id: Uuid,
        reply: oneshot::Sender<Result<(), String>>,
    },
}

/// Dedicated single-writer SQLite WAL actor supporting transactional SAVEPOINT rollback.
pub struct SqliteTool {
    tx: mpsc::Sender<SqlCmd>,
    max_digest_tokens: usize,
}

impl SqliteTool {
    pub fn open(db_path: &str, max_digest_tokens: usize) -> Result<Self, ToolError> {
        let conn = Connection::open(db_path).map_err(|e| ToolError::Execution(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            .map_err(|e| ToolError::Execution(e.to_string()))?;

        let (tx, mut rx) = mpsc::channel::<SqlCmd>(64);
        std::thread::Builder::new()
            .name("sqlite-actor".to_string())
            .spawn(move || {
                while let Some(cmd) = rx.blocking_recv() {
                    match cmd {
                        SqlCmd::Exec {
                            tx_id,
                            query,
                            reply,
                        } => {
                            let sp = format!("tx_{}", tx_id.simple());
                            let res = (|| -> Result<String, rusqlite::Error> {
                                conn.execute_batch(&format!("SAVEPOINT {sp};"))?;
                                let trimmed = query.trim_start().to_ascii_uppercase();
                                if trimmed.starts_with("SELECT") || trimmed.starts_with("PRAGMA") {
                                    let mut stmt = conn.prepare(&query)?;
                                    let cols: Vec<String> = stmt
                                        .column_names()
                                        .iter()
                                        .map(|s| s.to_string())
                                        .collect();
                                    let mut rows = stmt.query([])?;
                                    let mut out = vec![cols.join(" | ")];
                                    while let Some(r) = rows.next()? {
                                        let mut vals = Vec::with_capacity(cols.len());
                                        for i in 0..cols.len() {
                                            let v: rusqlite::types::Value = r.get(i)?;
                                            vals.push(format!("{v:?}"));
                                        }
                                        out.push(vals.join(" | "));
                                    }
                                    Ok(out.join("\n"))
                                } else {
                                    let changed = conn.execute(&query, [])?;
                                    Ok(format!("rows_affected={changed}"))
                                }
                            })();
                            let _ = reply.send(res.map_err(|e| e.to_string()));
                        }
                        SqlCmd::Commit { tx_id, reply } => {
                            let sp = format!("tx_{}", tx_id.simple());
                            let res = conn
                                .execute_batch(&format!("RELEASE SAVEPOINT {sp};"))
                                .map_err(|e| e.to_string());
                            let _ = reply.send(res);
                        }
                        SqlCmd::Rollback { tx_id, reply } => {
                            let sp = format!("tx_{}", tx_id.simple());
                            let res = conn
                                .execute_batch(&format!(
                                    "ROLLBACK TO SAVEPOINT {sp}; RELEASE SAVEPOINT {sp};"
                                ))
                                .map_err(|e| e.to_string());
                            let _ = reply.send(res);
                        }
                    }
                }
            })
            .map_err(|e| ToolError::Execution(e.to_string()))?;

        Ok(Self {
            tx,
            max_digest_tokens,
        })
    }
}

#[async_trait]
impl ReversibleTool for SqliteTool {
    fn name(&self) -> &str {
        "sql_exec"
    }

    fn risk_group(&self) -> RiskGroup {
        RiskGroup::FsMutate
    }

    async fn execute(&self, args: ToolArgs, tx_id: Uuid) -> Result<ToolOutput, ToolError> {
        let query = match args {
            ToolArgs::Sql { query } => query,
            _ => return Err(ToolError::InvalidArgs("expected Sql".to_string())),
        };
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(SqlCmd::Exec {
                tx_id,
                query,
                reply,
            })
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        let full = rx
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?
            .map_err(ToolError::Execution)?;
        let digest = compress_tool_output("sql_exec", 0, &full, self.max_digest_tokens);
        Ok(ToolOutput {
            exit_code: 0,
            full_payload: full,
            sensor_digest: digest,
        })
    }

    async fn commit(&self, tx_id: Uuid) -> Result<(), RollbackError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(SqlCmd::Commit { tx_id, reply })
            .await
            .map_err(|e| RollbackError::Failed(e.to_string()))?;
        rx.await
            .map_err(|e| RollbackError::Failed(e.to_string()))?
            .map_err(RollbackError::Failed)
    }

    async fn rollback(&self, tx_id: Uuid) -> Result<(), RollbackError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(SqlCmd::Rollback { tx_id, reply })
            .await
            .map_err(|e| RollbackError::Failed(e.to_string()))?;
        rx.await
            .map_err(|e| RollbackError::Failed(e.to_string()))?
            .map_err(RollbackError::Failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sqlite_savepoint_rolls_back() {
        let tool = SqliteTool::open(":memory:", 120).unwrap();
        let tx0 = Uuid::now_v7();
        tool.execute(
            ToolArgs::Sql {
                query: "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT);".to_string(),
            },
            tx0,
        )
        .await
        .unwrap();
        tool.commit(tx0).await.unwrap();

        let tx1 = Uuid::now_v7();
        tool.execute(
            ToolArgs::Sql {
                query: "INSERT INTO items (id, name) VALUES (1, 'speculative');".to_string(),
            },
            tx1,
        )
        .await
        .unwrap();
        tool.rollback(tx1).await.unwrap();

        let tx2 = Uuid::now_v7();
        let out = tool
            .execute(
                ToolArgs::Sql {
                    query: "SELECT COUNT(*) FROM items;".to_string(),
                },
                tx2,
            )
            .await
            .unwrap();
        assert!(out.full_payload.contains("0"));
    }
}
