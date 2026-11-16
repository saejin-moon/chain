use crate::config::{CalibrationData, RuntimeConfig};
use crate::engine::{DecisionBackend, GeneratorBackend, GeneratorSlot};
use crate::memory::pruner::{
    RecoveryPlaybook, build_scoped_synthesis_prompt, build_speculative_verify_prompt,
    build_state_diff_summary, select_recovery_playbook,
};
use crate::memory::working_state::TriPartiteMemory;
use crate::probes::actions::{
    ActionKind, CandidateAction, build_candidate_actions, extract_file_paths,
};
use crate::probes::state_frame::{ActionOutcomeDigest, StateFrame};
use crate::probes::templates::build_orthogonal_probes;
use crate::sandbox::traits::{ReversibleTool, ToolArgs, ToolOutput, ToolRegistry};
use crate::stats::bocpd::BocpdDetector;
use crate::stats::conformal::ConformalEstimator;
use crate::stats::dirichlet::DirichletEstimator;
use crate::stats::routing::{BayesRouter, RouteDecision};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, atomic::AtomicBool};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnTrace {
    pub turn: usize,
    pub route: String,
    pub top1_action: String,
    pub top1_prob: f64,
    pub runner_up_prob: f64,
    pub conformal_set: Vec<String>,
    pub q_hat: f64,
    pub d_js: f64,
    pub contradiction: bool,
    pub r_precond: f64,
    pub r_reversible: f64,
    pub r_progress: f64,
    pub r_aligned: f64,
    pub bocpd_short_prob: f64,
    pub probe_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMetrics {
    pub turns: usize,
    pub fast_path_hits: usize,
    pub speculative_commits: usize,
    pub speculative_rollbacks: usize,
    pub system2_escalations: usize,
    pub bocpd_prunes: usize,
    pub avg_probe_latency_us: u64,
    pub trace: Vec<TurnTrace>,
}

#[derive(Debug, Clone)]
pub struct EpisodeResult {
    pub completed: bool,
    pub summary: String,
    pub metrics: RuntimeMetrics,
}

#[derive(Deserialize)]
struct VerifyVerdict {
    verdict: String,
    #[serde(default)]
    #[allow(dead_code)]
    reason: String,
}

pub struct ChainRuntime {
    pub config: RuntimeConfig,
    pub decision_backend: Arc<dyn DecisionBackend>,
    pub generator_backend: Arc<dyn GeneratorBackend>,
    pub tools: ToolRegistry,
    pub dirichlet: DirichletEstimator,
    pub conformal: ConformalEstimator,
    pub router: BayesRouter,
    pub bocpd: BocpdDetector,
}

impl ChainRuntime {
    pub fn new(
        config: RuntimeConfig,
        calibration: &CalibrationData,
        decision_backend: Arc<dyn DecisionBackend>,
        generator_backend: Arc<dyn GeneratorBackend>,
        tools: ToolRegistry,
    ) -> Self {
        let dirichlet = DirichletEstimator::new(config.stats.w_pert_max);
        let conformal = ConformalEstimator::from_calibration(
            calibration,
            config.stats.target_alpha,
            config.stats.aci_gamma,
            config.stats.aci_alpha_min,
            config.stats.aci_alpha_max,
        );
        let router = BayesRouter::new(
            config.evsi.clone(),
            config.stats.tau_ood,
            config.stats.sprt_alpha,
            config.stats.sprt_beta,
        );
        let bocpd = BocpdDetector::new(config.stats.bocpd_hazard, config.stats.bocpd_trigger_prob);
        Self {
            config,
            decision_backend,
            generator_backend,
            tools,
            dirichlet,
            conformal,
            router,
            bocpd,
        }
    }

    pub async fn run_episode(
        &mut self,
        goal: &str,
        initial_files: Vec<String>,
        max_turns: usize,
    ) -> EpisodeResult {
        let tier0_sys1 = StateFrame::build_tier0_static(goal, &self.tools.compact_summary());
        let _ = self.decision_backend.pin_static_prefix(&tier0_sys1).await;
        self.conformal.reset_episode_drift();
        self.bocpd.reset();
        self.router.reset_sprt();
        let tier0_sys2: Arc<str> = Arc::from(format!(
            "You are CHAIN System-2 verifier/synthesizer. GOAL: {goal}"
        ));

        let mut memory = TriPartiteMemory::new(goal, vec!["Preserve workspace integrity".into()]);
        let mut observed_files = initial_files;
        let mut failing_tests = Vec::new();
        let mut recent_history: VecDeque<ActionOutcomeDigest> = VecDeque::with_capacity(4);
        let mut active_tx: Option<(Uuid, Arc<dyn ReversibleTool>)> = None;
        let mut read_only_turns = 0usize;
        let mut counterfactual_hint: Option<String> = None;
        let mut total_probe_us = 0u64;
        let goal_tokens: Vec<String> = goal
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| w.starts_with("KEY_") || w.len() >= 8)
            .map(String::from)
            .collect();

        let mut metrics = RuntimeMetrics {
            turns: 0,
            fast_path_hits: 0,
            speculative_commits: 0,
            speculative_rollbacks: 0,
            system2_escalations: 0,
            bocpd_prunes: 0,
            avg_probe_latency_us: 0,
            trace: Vec::new(),
        };

        for turn in 1..=max_turns {
            metrics.turns = turn;
            let candidates = build_candidate_actions(
                &observed_files,
                &failing_tests,
                active_tx.is_some(),
                read_only_turns > 0,
            );
            if read_only_turns > 0 {
                read_only_turns -= 1;
            }

            let frame = StateFrame {
                turn,
                active_tx: active_tx.as_ref().map(|(id, _)| id.to_string()),
                dirty_files: memory.artifact_index.keys().cloned().collect(),
                last_test_status: memory
                    .active_blocker
                    .as_deref()
                    .unwrap_or("CLEAN")
                    .to_string(),
                recent_history: recent_history.clone(),
                candidates: candidates.to_vec(),
            };

            let delta = frame.render_delta(self.config.decision_model.max_state_frame_tokens);
            let probes = build_orthogonal_probes(&candidates);
            let batch = match self
                .decision_backend
                .evaluate_probes(&delta, &probes, candidates.len())
                .await
            {
                Ok(b) => b,
                Err(e) => {
                    return EpisodeResult {
                        completed: false,
                        summary: format!("System-1 error: {e}"),
                        metrics,
                    };
                }
            };
            total_probe_us += batch.latency_us;

            let dir = self.dirichlet.evaluate(&batch);
            let conf = self.conformal.build_prediction_set(
                &dir.posterior_mean[..candidates.len()],
                &candidates,
                dir.d_js,
                self.config.stats.tau_ood,
            );

            // Update BOCPD changepoint detector
            let bocpd_upd = self
                .bocpd
                .step([dir.d_js, 1.0 - batch.r_progress, 1.0 - batch.r_aligned]);
            let sprt_stuck = self.router.update_sprt_progress(batch.r_progress > 0.5);

            if dir.d_js < self.config.stats.tau_clean
                && batch.r_progress > 0.8
                && batch.r_aligned > 0.8
            {
                memory.save_clean_checkpoint(turn, active_tx.as_ref().map(|(id, _)| *id), turn * 200);
            }

            if bocpd_upd.changepoint_triggered || sprt_stuck {
                metrics.bocpd_prunes += 1;
                let max_rep = memory
                    .dead_end_ledger
                    .iter()
                    .map(|d| d.count)
                    .max()
                    .unwrap_or(0);
                let playbook = select_recovery_playbook(
                    &memory,
                    dir.contradiction_flagged,
                    batch.r_aligned,
                    max_rep,
                    dir.d_js,
                    self.config.stats.tau_ood,
                );
                match playbook {
                    RecoveryPlaybook::RollbackWithCounterfactual { rollback_tx, hint } => {
                        if let Some((tx, tool)) = active_tx.take()
                            && tx == rollback_tx
                        {
                            let _ = tool.rollback(tx).await;
                        }
                        counterfactual_hint = Some(hint);
                    }
                    RecoveryPlaybook::ReadOnlyScopeNarrowing => {
                        read_only_turns = 2;
                    }
                    RecoveryPlaybook::GoalReAnchoring | RecoveryPlaybook::SoftStateDiffReset => {
                        let _summary = build_state_diff_summary(&mut memory, turn);
                    }
                }
                let _ = self
                    .generator_backend
                    .excise_kv_range(GeneratorSlot::ScopedSynthesis, 0, turn * 200)
                    .await;
                self.bocpd.reset();
                self.router.reset_sprt();
                continue;
            }

            let decision = self.router.route(&batch, &dir, &conf, &candidates);
            let mut sorted_p: Vec<f64> = dir.posterior_mean[..candidates.len()].to_vec();
            sorted_p.sort_by(|a, b| b.total_cmp(a));
            let top1_p = sorted_p.first().copied().unwrap_or(0.0);
            let second_p = sorted_p.get(1).copied().unwrap_or(0.0);
            let cset_labels: Vec<String> = conf
                .prediction_set
                .iter()
                .filter_map(|&i| candidates.get(i).map(|c| format!("{}:{}", c.label, c.description)))
                .collect();
            let step_trace = TurnTrace {
                turn,
                route: format!("{decision:?}"),
                top1_action: candidates[dir.top1_idx].description.clone(),
                top1_prob: (top1_p * 1000.0).round() / 1000.0,
                runner_up_prob: (second_p * 1000.0).round() / 1000.0,
                conformal_set: cset_labels,
                q_hat: (conf.q_hat_used * 1000.0).round() / 1000.0,
                d_js: (dir.d_js * 1000.0).round() / 1000.0,
                contradiction: dir.contradiction_flagged,
                r_precond: (batch.r_precond * 1000.0).round() / 1000.0,
                r_reversible: (batch.r_reversible * 1000.0).round() / 1000.0,
                r_progress: (batch.r_progress * 1000.0).round() / 1000.0,
                r_aligned: (batch.r_aligned * 1000.0).round() / 1000.0,
                bocpd_short_prob: (bocpd_upd.short_run_prob * 1000.0).round() / 1000.0,
                probe_ms: (batch.latency_us as f64 / 100.0).round() / 10.0,
            };
            tracing::info!(
                turn = step_trace.turn,
                route = %step_trace.route,
                top1 = %step_trace.top1_action,
                p1 = step_trace.top1_prob,
                p2 = step_trace.runner_up_prob,
                cset = ?step_trace.conformal_set,
                q_hat = step_trace.q_hat,
                d_js = step_trace.d_js,
                contradiction = step_trace.contradiction,
                r_pre = step_trace.r_precond,
                r_rev = step_trace.r_reversible,
                probe_ms = step_trace.probe_ms,
                "CHAIN turn telemetry"
            );
            metrics.trace.push(step_trace);
            let chosen_idx = match decision {
                RouteDecision::FastPath { action_idx } => {
                    metrics.fast_path_hits += 1;
                    action_idx
                }
                RouteDecision::Speculative { action_idx } => {
                    let cand = &candidates[action_idx];
                    if let Some((tool_name, args)) = candidate_to_tool_call(cand)
                        && let Some(tool) = self.tools.get(&tool_name).cloned()
                    {
                        let tx_id = Uuid::now_v7();
                        let abort_flag = Arc::new(AtomicBool::new(false));
                        let subset: Vec<&CandidateAction> = conf
                            .prediction_set
                            .iter()
                            .filter_map(|&i| candidates.get(i))
                            .collect();
                        let verify_prompt = build_speculative_verify_prompt(
                            tier0_sys2.clone(),
                            &memory,
                            cand,
                            &subset,
                        );

                        let (exec_res, verify_res) = tokio::join!(
                            tool.execute(args, tx_id),
                            self.generator_backend.generate_scoped(
                                GeneratorSlot::SpeculativeVerify,
                                verify_prompt,
                                abort_flag
                            )
                        );

                        let committed = match verify_res {
                            Ok(raw) => serde_json::from_str::<VerifyVerdict>(&raw)
                                .map(|v| v.verdict.eq_ignore_ascii_case("COMMIT"))
                                .unwrap_or(true),
                            Err(_) => false,
                        };

                        if committed {
                            metrics.speculative_commits += 1;
                            let _ = tool.commit(tx_id).await;
                            self.conformal.aci_update(true, 1.0 - dir.posterior_mean[action_idx]);
                            self.router.update_tool_reliability(&cand.description, true, 0.95);
                            if let Ok(out) = exec_res {
                                let found_goal = record_tool_outcome(
                                    turn,
                                    &cand.description,
                                    &out,
                                    &goal_tokens,
                                    &mut memory,
                                    &mut recent_history,
                                    &mut observed_files,
                                    &mut failing_tests,
                                );
                                if found_goal {
                                    metrics.avg_probe_latency_us =
                                        total_probe_us / metrics.turns.max(1) as u64;
                                    return EpisodeResult {
                                        completed: true,
                                        summary: format!(
                                            "Verified target invariant via speculative execution at turn {turn}"
                                        ),
                                        metrics,
                                    };
                                }
                            }
                            continue;
                        } else {
                            metrics.speculative_rollbacks += 1;
                            let _ = tool.rollback(tx_id).await;
                            self.conformal.aci_update(false, 1.0 - dir.posterior_mean[action_idx]);
                            self.router.update_tool_reliability(&cand.description, false, 0.95);
                            let picked = if !conf.prediction_set.is_empty() {
                                self.router.thompson_pick(&conf.prediction_set)
                            } else {
                                action_idx
                            };
                            self.router.update_thompson(picked, false);
                            memory.record_dead_end(&cand.description, &cand.label, "speculative_aborted");
                            continue;
                        }
                    }
                    action_idx
                }
                RouteDecision::System2Escalate { .. } => {
                    metrics.system2_escalations += 1;
                    let mut idx = dir.top1_idx;
                    let cand = &candidates[idx];
                    let mut prompt = build_scoped_synthesis_prompt(
                        tier0_sys2.clone(),
                        &memory,
                        cand,
                        counterfactual_hint.take().as_deref(),
                    );
                    prompt.max_tokens = 64;
                    let abort = Arc::new(AtomicBool::new(false));
                    if let Ok(resp) = self
                        .generator_backend
                        .generate_scoped(GeneratorSlot::ScopedSynthesis, prompt, abort)
                        .await
                    {
                        if let Some(picked) = candidates
                            .iter()
                            .position(|c| resp.contains(&format!("{}:", c.label)) || resp.trim().starts_with(&c.label))
                        {
                            idx = picked;
                        }
                        memory.record_turn(turn, candidates[idx].description.clone(), resp.clone());
                    }
                    // If System-2 or top1 points to an unread entity-bound tool, execute it
                    if matches!(candidates[idx].kind, ActionKind::EntityBound { .. } | ActionKind::MetaDone) {
                        idx
                    } else if let Some(first_ent) = candidates
                        .iter()
                        .position(|c| matches!(c.kind, ActionKind::EntityBound { .. }))
                    {
                        first_ent
                    } else {
                        continue;
                    }
                }
                RouteDecision::AskUser { reason } => {
                    metrics.avg_probe_latency_us = total_probe_us / metrics.turns.max(1) as u64;
                    return EpisodeResult {
                        completed: false,
                        summary: format!("Paused for user clarification: {reason}"),
                        metrics,
                    };
                }
            };

            // Execute Fast-Path discrete/meta action
            let cand = &candidates[chosen_idx];
            match &cand.kind {
                ActionKind::MetaDone => {
                    if !observed_files.is_empty() && memory.verified_facts.is_empty() && turn < max_turns {
                        // Force inspecting the next unread file rather than quitting before invariant verification
                        if let Some(first_ent) = candidates
                            .iter()
                            .position(|c| matches!(c.kind, ActionKind::EntityBound { .. }))
                        {
                            let ent_cand = &candidates[first_ent];
                            if let Some((tool_name, args)) = candidate_to_tool_call(ent_cand)
                                && let Some(tool) = self.tools.get(&tool_name).cloned()
                            {
                                let tx_id = Uuid::now_v7();
                                if let Ok(out) = tool.execute(args, tx_id).await {
                                    let found_goal = record_tool_outcome(
                                        turn,
                                        &ent_cand.description,
                                        &out,
                                        &goal_tokens,
                                        &mut memory,
                                        &mut recent_history,
                                        &mut observed_files,
                                        &mut failing_tests,
                                    );
                                    if found_goal {
                                        metrics.avg_probe_latency_us =
                                            total_probe_us / metrics.turns.max(1) as u64;
                                        return EpisodeResult {
                                            completed: true,
                                            summary: format!("Verified target invariant at turn {turn}"),
                                            metrics,
                                        };
                                    }
                                }
                            }
                            continue;
                        }
                    }
                    metrics.avg_probe_latency_us = total_probe_us / metrics.turns.max(1) as u64;
                    return EpisodeResult {
                        completed: true,
                        summary: format!("Completed at turn {turn}"),
                        metrics,
                    };
                }
                ActionKind::MetaRollbackLast => {
                    if let Some((tx, tool)) = active_tx.take() {
                        let _ = tool.rollback(tx).await;
                    }
                }
                ActionKind::EntityBound { .. } => {
                    if let Some((tool_name, args)) = candidate_to_tool_call(cand)
                        && let Some(tool) = self.tools.get(&tool_name).cloned()
                    {
                        let tx_id = Uuid::now_v7();
                        match tool.execute(args, tx_id).await {
                            Ok(out) => {
                                let ok = out.exit_code == 0;
                                self.conformal.aci_update(ok, 1.0 - dir.posterior_mean[chosen_idx]);
                                self.dirichlet.hedge_update(&batch, chosen_idx, 0.1);
                                self.router.update_platt(batch.raw_precond_logit, ok, 0.05);
                                self.router.update_tool_reliability(&cand.description, ok, 0.95);
                                let found_goal = record_tool_outcome(
                                    turn,
                                    &cand.description,
                                    &out,
                                    &goal_tokens,
                                    &mut memory,
                                    &mut recent_history,
                                    &mut observed_files,
                                    &mut failing_tests,
                                );
                                if found_goal {
                                    metrics.avg_probe_latency_us =
                                        total_probe_us / metrics.turns.max(1) as u64;
                                    return EpisodeResult {
                                        completed: true,
                                        summary: format!("Verified target invariant at turn {turn}"),
                                        metrics,
                                    };
                                }
                                if !tool.is_pure_read() {
                                    active_tx = Some((tx_id, tool));
                                }
                            }
                            Err(e) => {
                                memory.record_dead_end(&cand.description, &cand.label, &e.to_string());
                                memory.active_blocker = Some(e.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        metrics.avg_probe_latency_us = total_probe_us / metrics.turns.max(1) as u64;
        EpisodeResult {
            completed: false,
            summary: format!("Reached max_turns ({max_turns})"),
            metrics,
        }
    }
}

fn candidate_to_tool_call(cand: &CandidateAction) -> Option<(String, ToolArgs)> {
    match &cand.kind {
        ActionKind::EntityBound { tool, target } if tool == "read_file" => Some((
            tool.clone(),
            ToolArgs::ReadFile {
                path: target.clone(),
            },
        )),
        ActionKind::EntityBound { tool, target } if tool == "run_test" => Some((
            tool.clone(),
            ToolArgs::RunTest {
                test_name: target.clone(),
            },
        )),
        _ => None,
    }
}

fn record_tool_outcome(
    turn: usize,
    action_desc: &str,
    out: &ToolOutput,
    goal_tokens: &[String],
    memory: &mut TriPartiteMemory,
    recent: &mut VecDeque<ActionOutcomeDigest>,
    observed_files: &mut Vec<String>,
    failing_tests: &mut Vec<String>,
) -> bool {
    if out.exit_code == 0 {
        memory.active_blocker = None;
    } else {
        memory.active_blocker = Some(out.sensor_digest.clone());
    }
    if let Some(read_path) = action_desc
        .strip_prefix("read_file(")
        .and_then(|s| s.strip_suffix(')'))
    {
        observed_files.retain(|f| f != read_path);
        memory.record_artifact(
            read_path.to_string(),
            "inspected".to_string(),
            out.full_payload.lines().count(),
            Uuid::now_v7(),
        );
    }
    for f in extract_file_paths(&out.full_payload) {
        if !observed_files.contains(&f) && !memory.artifact_index.contains_key(&f) {
            observed_files.push(f);
        }
    }
    let found_goal = goal_tokens
        .iter()
        .any(|tok| tok.starts_with("KEY_") && out.full_payload.contains(tok));
    if found_goal {
        memory
            .verified_facts
            .push((Uuid::now_v7(), "Target invariant verified in file".to_string()));
    }
    if out.exit_code != 0 && action_desc.starts_with("run_test(") {
        let name = action_desc
            .trim_start_matches("run_test(")
            .trim_end_matches(')')
            .to_string();
        if !failing_tests.contains(&name) {
            failing_tests.push(name);
        }
    }
    memory.record_turn(turn, action_desc.to_string(), out.sensor_digest.clone());
    if recent.len() >= 3 {
        recent.pop_front();
    }
    recent.push_back(ActionOutcomeDigest {
        turn,
        action: action_desc.to_string(),
        sensor_digest: out.sensor_digest.clone(),
    });
    found_goal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mock::{MockDecisionBackend, MockGeneratorBackend};
    use crate::sandbox::fs_tools::ReadFileTool;

    #[tokio::test]
    async fn runtime_executes_fast_path_then_completes() {
        let tmp = std::env::temp_dir().join(format!("chain_rt_{}", Uuid::now_v7()));
        tokio::fs::create_dir_all(&tmp).await.unwrap();
        tokio::fs::write(tmp.join("lib.rs"), "pub fn ok() -> bool { true }")
            .await
            .unwrap();

        let dec = Arc::new(MockDecisionBackend::new());
        // Turn 1 (5 cands): 0=read_file(lib.rs); Turn 2 (lib.rs dequeued -> 3 cands): 1=DONE
        dec.push_batch(MockDecisionBackend::confident_batch(5, 0, true));
        dec.push_batch(MockDecisionBackend::confident_batch(3, 1, true));

        let generator = Arc::new(MockGeneratorBackend::new());
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(ReadFileTool {
            workspace_root: tmp.clone(),
            max_digest_tokens: 120,
        }));

        let mut rt = ChainRuntime::new(
            RuntimeConfig::default(),
            &CalibrationData::default(),
            dec,
            generator,
            tools,
        );
        let res = rt.run_episode("Verify lib.rs", vec!["lib.rs".into()], 5).await;
        assert!(res.completed);
        assert_eq!(res.metrics.fast_path_hits, 2);
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn runtime_triggers_bocpd_healing_on_drift() {
        let dec = Arc::new(MockDecisionBackend::new());
        for _ in 0..5 {
            dec.push_batch(MockDecisionBackend::confident_batch(2, 0, true));
        }
        for _ in 0..4 {
            dec.push_batch(MockDecisionBackend::drifting_batch(2));
        }
        dec.push_batch(MockDecisionBackend::confident_batch(2, 0, true));

        let generator = Arc::new(MockGeneratorBackend::new());
        let mut rt = ChainRuntime::new(
            RuntimeConfig::default(),
            &CalibrationData::default(),
            dec,
            generator,
            ToolRegistry::new(),
        );
        let res = rt.run_episode("Recover from drift", vec![], 10).await;
        assert!(res.metrics.bocpd_prunes >= 1);
    }
}
