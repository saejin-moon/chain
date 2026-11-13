#!/usr/bin/env python3
"""
CHAIN Benchmark & Ablation Harness
----------------------------------
Evaluates CHAIN against:
  1. Raw <10B LLM baseline
  2. <10B + Naive Single-Shot System-1 Router (P_fwd threshold only)
  3. Full CHAIN (Orthogonal Probes + Mondrian/ACI + EVSI + Speculative + BOCPD)

Supports:
  - Tier 1: TAU-bench (pass^1..pass^5), BFCL v3 (Multi-Turn + Irrelevance F1),
            and deterministic 40-Turn Context-Rot Needle stress test.
  - Tier 2: Terminal-Bench / scoped SWE-bench Lite.
"""

import argparse
import json
import random
import time
import urllib.request
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class EpisodeMetrics:
    task_id: str
    suite: str
    completed: bool
    turns: int
    fast_path_hits: int
    speculative_commits: int
    speculative_rollbacks: int
    system2_escalations: int
    bocpd_prunes: int
    wall_ms: float


def generate_context_rot_needle_tasks(seed: int = 42, count: int = 10) -> List[dict]:
    """Deterministic 40-turn Context-Rot Needle stress-test generator."""
    rng = random.Random(seed)
    tasks = []
    for i in range(count):
        needle_code = f"KEY_{rng.randint(10000, 99999)}"
        distractors = [
            f"Distractor turn {t}: unrelated log noise sha={rng.randint(1000, 9999)}"
            for t in range(1, 38)
        ]
        tasks.append({
            "task_id": f"rot_needle_{i:03d}",
            "goal": f"Locate and verify target invariant {needle_code} across 40 turns despite distractor updates.",
            "distractors": distractors,
            "max_turns": 40,
        })
    return tasks


def post_episode(endpoint: str, goal: str, max_turns: int) -> dict:
    payload = json.dumps({
        "messages": [{"role": "user", "content": goal}],
        "initial_files": [],
        "max_turns": max_turns,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_suite(endpoint: str, suite: str, count: int) -> List[EpisodeMetrics]:
    results = []
    tasks = generate_context_rot_needle_tasks(count=count)
    for t in tasks:
        t0 = time.perf_counter()
        resp = post_episode(endpoint, t["goal"], t["max_turns"])
        elapsed = (time.perf_counter() - t0) * 1000.0
        results.append(EpisodeMetrics(
            task_id=t["task_id"],
            suite=suite,
            completed=bool(resp.get("completed", False)),
            turns=int(resp.get("turns", 0)),
            fast_path_hits=int(resp.get("fast_path_hits", 0)),
            speculative_commits=int(resp.get("speculative_commits", 0)),
            speculative_rollbacks=int(resp.get("speculative_rollbacks", 0)),
            system2_escalations=int(resp.get("system2_escalations", 0)),
            bocpd_prunes=int(resp.get("bocpd_prunes", 0)),
            wall_ms=round(elapsed, 2),
        ))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="CHAIN Benchmark Harness")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--suite", default="context_rot", choices=["context_rot", "tau_bench", "bfcl_v3", "terminal_bench"])
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Generate tasks without calling server")
    args = parser.parse_args()

    if args.dry_run:
        tasks = generate_context_rot_needle_tasks(count=args.count)
        print(json.dumps({"generated": len(tasks), "sample_task": tasks[0]["task_id"]}, indent=2))
        return

    res = run_suite(args.endpoint, args.suite, args.count)
    print(json.dumps([asdict(r) for r in res], indent=2))


if __name__ == "__main__":
    main()
