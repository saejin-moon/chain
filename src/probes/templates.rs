use super::actions::CandidateAction;
use crate::engine::{MAX_ACTIONS, ProbeSpec};
use smallvec::SmallVec;

/// Construct the 12 orthogonal probes sharing the StateFrame KV prefix:
/// - 1x Direct Selection (P_fwd)
/// - 1x Inverse Exclusion (P_inv)
/// - 3x Perturbed Framing (P_pert_1..3)
/// - 5x Binary Noul Rubrics (r_precond, r_reversible, r_progress, r_aligned, r_info_missing)
/// - 1x Ordinal Score Severity (score_severity: 0..4)
/// - 1x Tie-breaker/sanity check (r_safe_fastpath)
pub fn build_orthogonal_probes(candidates: &[CandidateAction]) -> Vec<ProbeSpec> {
    let choice_labels: SmallVec<[String; MAX_ACTIONS]> =
        candidates.iter().map(|c| c.label.clone()).collect();
    let yes_no: SmallVec<[String; MAX_ACTIONS]> =
        smallvec::smallvec!["YES".to_string(), "NO".to_string()];
    let scores: SmallVec<[String; MAX_ACTIONS]> = (0..=4).map(|i| i.to_string()).collect();

    let top_hint = candidates
        .first()
        .map(|c| c.description.as_str())
        .unwrap_or("none");

    vec![
        ProbeSpec {
            id: "p_fwd",
            suffix: "\n[PROBE:FWD] Optimal next action label:".to_string(),
            choices: choice_labels.clone(),
        },
        ProbeSpec {
            id: "p_inv",
            suffix: "\n[PROBE:INV] Which action label violates constraints or is counterproductive?:"
                .to_string(),
            choices: choice_labels.clone(),
        },
        ProbeSpec {
            id: "p_pert_1",
            suffix: "\n[PROBE:P1] Best immediate step toward GOAL (label only):".to_string(),
            choices: choice_labels.clone(),
        },
        ProbeSpec {
            id: "p_pert_2",
            suffix: "\n[PROBE:P2] Considering RECENT outcomes, select next action label:".to_string(),
            choices: choice_labels.clone(),
        },
        ProbeSpec {
            id: "p_pert_3",
            suffix: "\n[PROBE:P3] Pick the single highest-utility candidate label:".to_string(),
            choices: choice_labels,
        },
        ProbeSpec {
            id: "r_precond",
            suffix: format!(
                "\n[NOUL:PRECOND] Are all preconditions/arguments present for '{top_hint}'? (YES/NO):"
            ),
            choices: yes_no.clone(),
        },
        ProbeSpec {
            id: "r_reversible",
            suffix: format!(
                "\n[NOUL:REV] Can '{top_hint}' be cleanly rolled back with no external side effects? (YES/NO):"
            ),
            choices: yes_no.clone(),
        },
        ProbeSpec {
            id: "r_progress",
            suffix: "\n[NOUL:PROG] Does RECENT show measurable forward progress toward GOAL? (YES/NO):"
                .to_string(),
            choices: yes_no.clone(),
        },
        ProbeSpec {
            id: "r_aligned",
            suffix: "\n[NOUL:ALIGN] Is the current trajectory still aligned with original GOAL? (YES/NO):"
                .to_string(),
            choices: yes_no.clone(),
        },
        ProbeSpec {
            id: "r_info_missing",
            suffix: "\n[NOUL:MISSING] Is required user intent or secret missing from STATE? (YES/NO):"
                .to_string(),
            choices: yes_no.clone(),
        },
        ProbeSpec {
            id: "score_severity",
            suffix: format!(
                "\n[SCORE:BLAST] Rate blast-radius severity of '{top_hint}' from 0 (none) to 4 (critical):"
            ),
            choices: scores,
        },
        ProbeSpec {
            id: "r_safe_fastpath",
            suffix: "\n[NOUL:FAST] Is deterministic execution safe without System-2 verification? (YES/NO):"
                .to_string(),
            choices: yes_no,
        },
    ]
}
