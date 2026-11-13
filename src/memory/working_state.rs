use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactEntry {
    pub path: String,
    pub sha_short: String,
    pub line_count: usize,
    pub last_tx: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadEndEntry {
    pub action_sig: String,
    pub target: String,
    pub error_sig: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EphemeralTurn {
    pub turn: usize,
    pub action: String,
    pub output_digest: String,
}

/// Tri-Partite Memory: Deterministic ground truth + DeadEndLedger + k=2 Ring Buffer.
/// Eliminates temporal shadowing and LLM summarization loss.
#[derive(Debug, Clone)]
pub struct TriPartiteMemory {
    pub goal: String,
    pub invariants: Vec<String>,
    /// Overwriting artifact index (prevents stale file versions from shadowing live disk state)
    pub artifact_index: FxHashMap<String, ArtifactEntry>,
    /// Verified assertions keyed by tx_id
    pub verified_facts: Vec<(Uuid, String)>,
    /// Latest active stderr / compiler blocker (cleared on pass)
    pub active_blocker: Option<String>,
    /// Bounded negative constraints (last 5 failed attempts)
    pub dead_end_ledger: VecDeque<DeadEndEntry>,
    /// Ephemeral turn ring buffer (k = 2)
    pub ephemeral_turns: VecDeque<EphemeralTurn>,
    /// Last high-trust clean checkpoint
    pub clean_checkpoint: Option<CleanCheckpoint>,
}

#[derive(Debug, Clone)]
pub struct CleanCheckpoint {
    pub turn: usize,
    pub tx_id: Option<Uuid>,
    pub kv_pos: usize,
    pub snapshot_facts_len: usize,
}

impl TriPartiteMemory {
    pub fn new(goal: impl Into<String>, invariants: Vec<String>) -> Self {
        Self {
            goal: goal.into(),
            invariants,
            artifact_index: FxHashMap::default(),
            verified_facts: Vec::new(),
            active_blocker: None,
            dead_end_ledger: VecDeque::with_capacity(5),
            ephemeral_turns: VecDeque::with_capacity(2),
            clean_checkpoint: None,
        }
    }

    pub fn record_artifact(&mut self, path: String, sha_short: String, line_count: usize, tx_id: Uuid) {
        self.artifact_index.insert(
            path.clone(),
            ArtifactEntry {
                path,
                sha_short,
                line_count,
                last_tx: tx_id,
            },
        );
    }

    pub fn record_turn(&mut self, turn: usize, action: String, output_digest: String) {
        if self.ephemeral_turns.len() >= 2 {
            self.ephemeral_turns.pop_front();
        }
        self.ephemeral_turns.push_back(EphemeralTurn {
            turn,
            action,
            output_digest,
        });
    }

    pub fn record_dead_end(&mut self, action_sig: &str, target: &str, error_sig: &str) -> usize {
        if let Some(existing) = self
            .dead_end_ledger
            .iter_mut()
            .find(|e| e.action_sig == action_sig && e.target == target && e.error_sig == error_sig)
        {
            existing.count += 1;
            return existing.count;
        }
        if self.dead_end_ledger.len() >= 5 {
            self.dead_end_ledger.pop_front();
        }
        self.dead_end_ledger.push_back(DeadEndEntry {
            action_sig: action_sig.to_string(),
            target: target.to_string(),
            error_sig: error_sig.to_string(),
            count: 1,
        });
        1
    }

    pub fn save_clean_checkpoint(&mut self, turn: usize, tx_id: Option<Uuid>, kv_pos: usize) {
        self.clean_checkpoint = Some(CleanCheckpoint {
            turn,
            tx_id,
            kv_pos,
            snapshot_facts_len: self.verified_facts.len(),
        });
    }

    /// Render Tier-1 deterministic WorkingState for System-2 (~400-900 tokens max).
    pub fn render_working_state(&self) -> String {
        let mut s = format!("GOAL: {}\n", self.goal);
        if !self.invariants.is_empty() {
            s.push_str(&format!("INVARIANTS: {}\n", self.invariants.join(" | ")));
        }
        if !self.artifact_index.is_empty() {
            s.push_str("ARTIFACTS:\n");
            for a in self.artifact_index.values() {
                s.push_str(&format!("- {} (lines={}, sha={})\n", a.path, a.line_count, a.sha_short));
            }
        }
        if !self.verified_facts.is_empty() {
            s.push_str("VERIFIED_FACTS:\n");
            for (_, fact) in self.verified_facts.iter().rev().take(6).rev() {
                s.push_str(&format!("- {fact}\n"));
            }
        }
        if let Some(err) = &self.active_blocker {
            s.push_str(&format!("ACTIVE_BLOCKER: {err}\n"));
        }
        if !self.dead_end_ledger.is_empty() {
            s.push_str("DO_NOT_REPEAT:\n");
            for d in &self.dead_end_ledger {
                s.push_str(&format!(
                    "- DO NOT {} on {} (failed {}x: {})\n",
                    d.action_sig, d.target, d.count, d.error_sig
                ));
            }
        }
        s
    }
}
