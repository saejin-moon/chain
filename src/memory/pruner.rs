use crate::engine::ScopedPrompt;
use crate::memory::working_state::TriPartiteMemory;
use crate::probes::actions::CandidateAction;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryPlaybook {
    SoftStateDiffReset,
    RollbackWithCounterfactual { rollback_tx: Uuid, hint: String },
    ReadOnlyScopeNarrowing,
    GoalReAnchoring,
}

/// Deterministically select a recovery playbook when BOCPD or System-1 drift triggers.
pub fn select_recovery_playbook(
    memory: &TriPartiteMemory,
    contradiction_flagged: bool,
    r_aligned: f64,
    repeated_error_count: usize,
    d_js: f64,
    tau_ood: f64,
) -> RecoveryPlaybook {
    if contradiction_flagged || r_aligned < 0.50 {
        return RecoveryPlaybook::GoalReAnchoring;
    }
    if repeated_error_count >= 2
        && let Some(cp) = &memory.clean_checkpoint
        && let Some(tx_id) = cp.tx_id
    {
        let last_err = memory
            .active_blocker
            .as_deref()
            .unwrap_or("repeated failure");
        return RecoveryPlaybook::RollbackWithCounterfactual {
            rollback_tx: tx_id,
            hint: format!(
                "Rolled back to clean turn {}. Avoid repeating failed pattern ({last_err}); choose a structurally different approach.",
                cp.turn
            ),
        };
    }
    if d_js >= tau_ood {
        return RecoveryPlaybook::ReadOnlyScopeNarrowing;
    }
    RecoveryPlaybook::SoftStateDiffReset
}

/// Deterministic StateDiffSummary computed in Rust (zero LLM summarization call).
pub fn build_state_diff_summary(memory: &mut TriPartiteMemory, current_turn: usize) -> String {
    let clean_turn = memory
        .clean_checkpoint
        .as_ref()
        .map(|c| c.turn)
        .unwrap_or(0);
    memory.ephemeral_turns.clear();
    format!(
        "[STATE_DIFF_SUMMARY pruned_turns={}..{}]\n{}",
        clean_turn,
        current_turn,
        memory.render_working_state()
    )
}

/// Scoped System-2 prompt builders (Slot 0 = verify, Slot 1 = synthesis/escalate).
pub fn build_speculative_verify_prompt(
    tier0: Arc<str>,
    memory: &TriPartiteMemory,
    top_action: &CandidateAction,
    conformal_subset: &[&CandidateAction],
) -> ScopedPrompt {
    let cand_list: Vec<String> = conformal_subset
        .iter()
        .map(|c| format!("{}: {}", c.label, c.description))
        .collect();
    let task = format!(
        "[SPECULATIVE_VERIFY]\nProposed: {} ({})\nConformal Set C(S): [{}]\nRespond JSON: {{\"verdict\":\"COMMIT\"|\"ABORT\",\"alt_choice\":null|\"<label>\",\"reason\":\"<1-line>\"}}",
        top_action.label,
        top_action.description,
        cand_list.join(", ")
    );
    ScopedPrompt {
        tier0_static: tier0,
        tier1_working_state: memory.render_working_state(),
        tier2_task: task,
        max_tokens: 64,
        gbnf_grammar: Some(
            r##"root ::= "{\"verdict\":" ("\"COMMIT\"" | "\"ABORT\"") ",\"alt_choice\":" ("null" | "\"" [A-P] "\"") ",\"reason\":\"" [^"]* "\"}"##
                .to_string(),
        ),
    }
}

pub fn build_scoped_synthesis_prompt(
    tier0: Arc<str>,
    memory: &TriPartiteMemory,
    selected: &CandidateAction,
    counterfactual_hint: Option<&str>,
) -> ScopedPrompt {
    let hint_line = counterfactual_hint
        .map(|h| format!("\nHINT: {h}"))
        .unwrap_or_default();
    let task = format!(
        "[SCOPED_SYNTHESIS]\nSelected Slot: {} ({}){}\nEmit only the exact arguments or unified patch required.",
        selected.label, selected.description, hint_line
    );
    ScopedPrompt {
        tier0_static: tier0,
        tier1_working_state: memory.render_working_state(),
        tier2_task: task,
        max_tokens: 512,
        gbnf_grammar: None,
    }
}
