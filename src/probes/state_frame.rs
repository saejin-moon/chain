use super::actions::CandidateAction;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct ActionOutcomeDigest {
    pub turn: usize,
    pub action: String,
    pub sensor_digest: String,
}

/// Compact token-budgeted representation (~300-500 tokens total) fed to System-1.
/// Never contains raw chat history or full file contents.
#[derive(Debug, Clone)]
pub struct StateFrame {
    pub turn: usize,
    pub active_tx: Option<String>,
    pub dirty_files: Vec<String>,
    pub last_test_status: String,
    pub recent_history: VecDeque<ActionOutcomeDigest>,
    pub candidates: Vec<CandidateAction>,
}

impl StateFrame {
    pub fn build_tier0_static(goal: &str, tool_summary: &str) -> String {
        format!(
            "[CHAIN_SYS1]\nGOAL: {goal}\nTOOLS: {tool_summary}\nRespond strictly with the requested token."
        )
    }

    pub fn render_delta(&self, max_tokens: usize) -> String {
        let mut s = format!(
            "[STATE t={} tx={} test={} dirty=[{}]]\n",
            self.turn,
            self.active_tx.as_deref().unwrap_or("none"),
            self.last_test_status,
            self.dirty_files.join(",")
        );
        s.push_str("[RECENT]\n");
        for item in self.recent_history.iter().rev().take(3).rev() {
            s.push_str(&format!(
                "t{}: {} => {}\n",
                item.turn, item.action, item.sensor_digest
            ));
        }
        s.push_str("[CANDIDATES]\n");
        for c in &self.candidates {
            s.push_str(&format!("{}: {}\n", c.label, c.description));
        }
        let max_chars = max_tokens.saturating_mul(4);
        if s.len() > max_chars {
            let mut cut = max_chars;
            while !s.is_char_boundary(cut) && cut > 0 {
                cut -= 1;
            }
            s.truncate(cut);
        }
        s
    }
}
