use crate::engine::MAX_ACTIONS;
use crate::sandbox::traits::RiskGroup;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionKind {
    /// Control-flow meta-actions (zero System-2 tokens)
    MetaDone,
    MetaRollbackLast,
    MetaPruneNow,
    MetaAskUser,
    /// Discrete entity-bound tool call (zero System-2 tokens)
    EntityBound {
        tool: String,
        target: String,
    },
    /// Scoped System-2 synthesis slot (requires <10B generator)
    SynthesisSlot {
        tool: String,
        target: Option<String>,
    },
    /// Explicit abstention / escalation option
    EscalateSystem2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateAction {
    pub label: String,
    pub description: String,
    pub kind: ActionKind,
    pub risk_group: RiskGroup,
    pub reversible: bool,
    pub requires_system2: bool,
}

/// Build a dynamic discrete candidate set `A_t` (capped at MAX_ACTIONS) from
/// observed file paths, failing tests, and registered tools.
pub fn build_candidate_actions(
    observed_files: &[String],
    failing_tests: &[String],
    has_active_tx: bool,
    read_only_mode: bool,
) -> SmallVec<[CandidateAction; MAX_ACTIONS]> {
    let mut actions = SmallVec::<[CandidateAction; MAX_ACTIONS]>::new();
    let labels = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
    ];
    let mut push_act = |desc: String, kind: ActionKind, risk: RiskGroup, rev: bool, s2: bool| {
        if actions.len() < MAX_ACTIONS {
            let label = labels[actions.len()].to_string();
            actions.push(CandidateAction {
                label,
                description: desc,
                kind,
                risk_group: risk,
                reversible: rev,
                requires_system2: s2,
            });
        }
    };

    // 1. Entity-bound reads & tests (zero System-2 tokens)
    for f in observed_files.iter().take(3) {
        push_act(
            format!("read_file({f})"),
            ActionKind::EntityBound {
                tool: "read_file".to_string(),
                target: f.clone(),
            },
            RiskGroup::PureRead,
            true,
            false,
        );
    }
    for t in failing_tests.iter().take(2) {
        push_act(
            format!("run_test({t})"),
            ActionKind::EntityBound {
                tool: "run_test".to_string(),
                target: t.clone(),
            },
            RiskGroup::Shell,
            true,
            false,
        );
    }

    // 2. Scoped synthesis slots (only if not restricted to read-only recovery)
    if !read_only_mode {
        if let Some(f) = observed_files.first() {
            push_act(
                format!("PATCH_FILE({f})"),
                ActionKind::SynthesisSlot {
                    tool: "patch_file".to_string(),
                    target: Some(f.clone()),
                },
                RiskGroup::FsMutate,
                true,
                true,
            );
        }
        push_act(
            "WRITE_SHELL_CMD".to_string(),
            ActionKind::SynthesisSlot {
                tool: "shell_exec".to_string(),
                target: None,
            },
            RiskGroup::Shell,
            false,
            true,
        );
    }

    // 3. Control-flow meta-actions
    if has_active_tx {
        push_act(
            "ROLLBACK_LAST".to_string(),
            ActionKind::MetaRollbackLast,
            RiskGroup::PureRead,
            true,
            false,
        );
    }
    if observed_files.is_empty() {
        push_act(
            "DONE".to_string(),
            ActionKind::MetaDone,
            RiskGroup::PureRead,
            true,
            false,
        );
    }
    push_act(
        "ESCALATE_SYSTEM2".to_string(),
        ActionKind::EscalateSystem2,
        RiskGroup::PureRead,
        true,
        true,
    );

    actions
}

/// Deterministically extract file paths (e.g. `src/foo.rs`) from a text snippet.
pub fn extract_file_paths(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for tok in text.split_whitespace() {
        let cleaned = tok.trim_matches(|c: char| matches!(c, ',' | ';' | ':' | '(' | ')' | '[' | ']' | '\'' | '"'));
        let base = cleaned.split(':').next().unwrap_or(cleaned);
        if (base.contains('/') || base.ends_with(".rs") || base.ends_with(".py") || base.ends_with(".toml"))
            && base.len() > 3
            && !out.iter().any(|x| x == base)
        {
            out.push(base.to_string());
        }
    }
    out
}
