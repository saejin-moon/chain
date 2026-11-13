use super::traits::{ReversibleTool, RiskGroup, RollbackError, ToolArgs, ToolError, ToolOutput};
use crate::probes::digest::compress_tool_output;
use async_trait::async_trait;
use rustc_hash::FxHashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use uuid::Uuid;

pub struct ReadFileTool {
    pub workspace_root: PathBuf,
    pub max_digest_tokens: usize,
}

#[async_trait]
impl ReversibleTool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn risk_group(&self) -> RiskGroup {
        RiskGroup::PureRead
    }

    async fn execute(&self, args: ToolArgs, _tx_id: Uuid) -> Result<ToolOutput, ToolError> {
        let path = match args {
            ToolArgs::ReadFile { path } => path,
            _ => return Err(ToolError::InvalidArgs("expected ReadFile".to_string())),
        };
        let target = resolve_safe(&self.workspace_root, &path)?;
        let content = tokio::fs::read_to_string(&target).await?;
        let digest = compress_tool_output("read_file", 0, &content, self.max_digest_tokens);
        Ok(ToolOutput {
            exit_code: 0,
            full_payload: content,
            sensor_digest: digest,
        })
    }

    async fn rollback(&self, _tx_id: Uuid) -> Result<(), RollbackError> {
        Ok(())
    }
}

#[derive(Debug)]
struct FsBackup {
    target_path: PathBuf,
    backup_path: Option<PathBuf>,
}

/// Shadow-copy filesystem writer using `reflink_copy::reflink_or_copy` (CoW O(1) on
/// Btrfs/XFS/APFS, fast copy fallback on ext4) and atomic cleanup on commit/rollback.
pub struct WriteFileTool {
    pub workspace_root: PathBuf,
    pub shadow_dir: PathBuf,
    pub max_digest_tokens: usize,
    backups: Mutex<FxHashMap<Uuid, FsBackup>>,
}

impl WriteFileTool {
    pub fn new(workspace_root: PathBuf, max_digest_tokens: usize) -> Self {
        let shadow_dir = workspace_root.join(".chain_tx");
        Self {
            workspace_root,
            shadow_dir,
            max_digest_tokens,
            backups: Mutex::new(FxHashMap::default()),
        }
    }
}

#[async_trait]
impl ReversibleTool for WriteFileTool {
    fn name(&self) -> &str {
        "patch_file"
    }

    fn risk_group(&self) -> RiskGroup {
        RiskGroup::FsMutate
    }

    async fn execute(&self, args: ToolArgs, tx_id: Uuid) -> Result<ToolOutput, ToolError> {
        let (path, content) = match args {
            ToolArgs::WriteFile { path, content } => (path, content),
            _ => return Err(ToolError::InvalidArgs("expected WriteFile".to_string())),
        };
        let target = resolve_safe(&self.workspace_root, &path)?;
        let backup_path = if target.exists() {
            let tx_dir = self.shadow_dir.join(tx_id.to_string());
            tokio::fs::create_dir_all(&tx_dir).await?;
            let bk = tx_dir.join("backup.bin");
            let t_clone = target.clone();
            let b_clone = bk.clone();
            tokio::task::spawn_blocking(move || reflink_copy::reflink_or_copy(&t_clone, &b_clone))
                .await
                .map_err(|e| ToolError::Execution(e.to_string()))??;
            Some(bk)
        } else {
            if let Some(parent) = target.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            None
        };

        tokio::fs::write(&target, &content).await?;
        self.backups.lock().unwrap().insert(
            tx_id,
            FsBackup {
                target_path: target,
                backup_path,
            },
        );

        let full = format!("wrote {} bytes to {}", content.len(), path);
        let digest = compress_tool_output("patch_file", 0, &full, self.max_digest_tokens);
        Ok(ToolOutput {
            exit_code: 0,
            full_payload: full,
            sensor_digest: digest,
        })
    }

    async fn commit(&self, tx_id: Uuid) -> Result<(), RollbackError> {
        let bk = self.backups.lock().unwrap().remove(&tx_id);
        if let Some(bk) = bk
            && let Some(bp) = bk.backup_path
            && let Some(dir) = bp.parent()
        {
            let _ = tokio::fs::remove_dir_all(dir).await;
        }
        Ok(())
    }

    async fn rollback(&self, tx_id: Uuid) -> Result<(), RollbackError> {
        let bk = self
            .backups
            .lock()
            .unwrap()
            .remove(&tx_id)
            .ok_or(RollbackError::NotFound(tx_id))?;

        match bk.backup_path {
            Some(bp) => {
                tokio::fs::rename(&bp, &bk.target_path)
                    .await
                    .map_err(|e| RollbackError::Failed(e.to_string()))?;
                if let Some(dir) = bp.parent() {
                    let _ = tokio::fs::remove_dir_all(dir).await;
                }
            }
            None => {
                if bk.target_path.exists() {
                    tokio::fs::remove_file(&bk.target_path)
                        .await
                        .map_err(|e| RollbackError::Failed(e.to_string()))?;
                }
            }
        }
        Ok(())
    }
}

pub struct RunTestTool {
    pub workspace_root: PathBuf,
    pub max_digest_tokens: usize,
}

#[async_trait]
impl ReversibleTool for RunTestTool {
    fn name(&self) -> &str {
        "run_test"
    }

    fn risk_group(&self) -> RiskGroup {
        RiskGroup::Shell
    }

    async fn execute(&self, args: ToolArgs, _tx_id: Uuid) -> Result<ToolOutput, ToolError> {
        let cmd_str = match args {
            ToolArgs::RunTest { test_name } => format!("cargo test {test_name}"),
            ToolArgs::Shell { command } => command,
            _ => return Err(ToolError::InvalidArgs("expected RunTest or Shell".to_string())),
        };
        let out = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(&cmd_str)
            .current_dir(&self.workspace_root)
            .output()
            .await?;
        let code = out.status.code().unwrap_or(-1);
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let digest = compress_tool_output("run_test", code, &combined, self.max_digest_tokens);
        Ok(ToolOutput {
            exit_code: code,
            full_payload: combined,
            sensor_digest: digest,
        })
    }

    async fn rollback(&self, _tx_id: Uuid) -> Result<(), RollbackError> {
        Ok(())
    }
}

fn resolve_safe(root: &Path, rel: &str) -> Result<PathBuf, ToolError> {
    let p = Path::new(rel);
    if p.is_absolute() || rel.contains("..") {
        return Err(ToolError::InvalidArgs(format!(
            "path escapes workspace: {rel}"
        )));
    }
    Ok(root.join(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn write_file_tool_rolls_back_cleanly() {
        let tmp = std::env::temp_dir().join(format!("chain_fs_test_{}", Uuid::now_v7()));
        tokio::fs::create_dir_all(&tmp).await.unwrap();
        let file = tmp.join("sample.txt");
        tokio::fs::write(&file, "original").await.unwrap();

        let tool = WriteFileTool::new(tmp.clone(), 120);
        let tx = Uuid::now_v7();
        tool.execute(
            ToolArgs::WriteFile {
                path: "sample.txt".to_string(),
                content: "corrupted_speculative_edit".to_string(),
            },
            tx,
        )
        .await
        .unwrap();
        assert_eq!(
            tokio::fs::read_to_string(&file).await.unwrap(),
            "corrupted_speculative_edit"
        );

        tool.rollback(tx).await.unwrap();
        assert_eq!(tokio::fs::read_to_string(&file).await.unwrap(), "original");
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }
}
