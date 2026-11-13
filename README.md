# CHAIN — Conformal Heterogeneous Agent Inference Network

A self-contained, local-first **Rust** agent runtime pairing a **2B System-1 decision model** (`baselabs/joey-1`) with a **`<10B` System-2 LLM** (`Qwen/Qwen3.5-9B`), connected through a closed-loop **Statistical Sensor Bus**, **Speculative `ReversibleTool` Sandbox**, and **BOCPD Context-Rot State Healing**.

---

## Architecture

1. **Token-Budgeted `StateFrame` (`src/probes/`):**
   - Every tool emits a deterministic `sensor_digest` (capped at `<= 120` tokens, zero LLM calls).
   - System-1 evaluates a compact `StateFrame` (`~300–500` tokens) plus **12 orthogonal probes** (`P_fwd`, `P_inv`, `P_pert_1..3`, `Noul` rubrics, and `Score` severity) in one shared-prefix pass.
   - Dynamic candidate sets $\mathcal{A}_t$ combine control-flow meta-actions (`DONE`, `ROLLBACK_LAST`, `ASK_USER`), entity-bound tools (`read_file(path)`, `run_test(name)`), scoped synthesis slots, and `ESCALATE_SYSTEM2`.

2. **Statistical Sensor Bus (`src/stats/`):**
   - Zero-allocation stack-array math (`< 5 µs` per tick):
     - **Dirichlet & Divergence (`dirichlet.rs`):** Mean $D_{\text{JS}}$ across perturbations, forward-inverse consistency $\kappa$, and Hedge-weighted Dirichlet posterior.
     - **Mondrian + Adaptive Conformal Inference (`conformal.rs`):** Per-tool-group quantiles (`pure_read`, `fs_mutate`, `shell`, `network_api`), online ACI $\alpha_t$ updates, and capped Power Martingale / Fisher $\chi^2$ calibration-drift detection.
     - **EVSI / Bayes-Risk Routing (`routing.rs`):** Routes between **Fast-Path** (0 System-2 tokens), **Speculative Execution** (parallel sandbox + System-2 verify), **System-2 Escalation**, and **Ask User**, alongside Beta-Binomial tool reliability, online Platt scaling, SPRT early stopping, and Thompson Sampling for recovery.

3. **Tri-Partite Memory & BOCPD State Healing (`src/memory/`, `src/stats/bocpd.rs`):**
   - **Zero-LLM-Summarization Memory:** Deterministic `WorkingState` (overwriting `artifact_index`, `tx_id`-keyed facts), `DeadEndLedger` (negative constraints), and $k=2$ ephemeral ring buffer.
   - **BOCPD Changepoint Detection:** Monitors `[D_JS, 1 - r_progress, 1 - r_aligned]` and triggers deterministic recovery playbooks + in-place KV excision.

4. **Speculative `ReversibleTool` Sandbox (`src/sandbox/`):**
   - **Filesystem:** `reflink-copy` CoW snapshots (`WriteFileTool`) with O(1) rollback.
   - **SQLite:** Single-writer WAL actor (`SqliteTool`) with transactional `SAVEPOINT` commit/rollback.

---

## Quick Start (`baselabs/joey-1` + `Qwen/Qwen3.5-9B`, No GGUF Needed)

### 1. Prerequisites
- **Rust:** `rustup` / `cargo` (Edition 2024)
- **Python (for model servers):** `pip install torch transformers fastapi uvicorn accelerate vllm`

### 2. Hardware / VRAM Requirements
| Configuration | System-1 (`baselabs/joey-1` 2B) | System-2 (`Qwen/Qwen3.5-9B`) | Total Hardware |
| :--- | :--- | :--- | :--- |
| **16 GB VRAM GPU** *(Hybrid)* | ~4.5 GB System RAM (`--device cpu`) | ~7.5–9 GB VRAM (`--quantization bitsandbytes` or `fp8`) | **16 GB VRAM GPU + 16–32 GB RAM** |
| **24 GB+ VRAM GPU** *(Co-located)* | ~4.5 GB VRAM (`bf16`) | ~11–13 GB VRAM (`fp8` / `awq`) or ~19 GB (`bf16` on 32GB+) | **24 GB VRAM** (`fp8` System-2) or **32 GB+ VRAM** (`bf16`) |

### 3. Launch Model Servers
**Terminal 1 — System-1 (`baselabs/joey-1` on port `8081`):**
```bash
python3 scripts/serve_joey.py --model baselabs/joey-1 --port 8081 --device auto --dtype bf16
```

**Terminal 2 — System-2 (`Qwen/Qwen3.5-9B` via vLLM on port `8082`):**
```bash
vllm serve Qwen/Qwen3.5-9B \
  --port 8082 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.65 \
  --max-model-len 16384
```
*(On a 24 GB GPU, add `--quantization fp8` or `--quantization bitsandbytes` to leave headroom for `joey-1`.)*

### 4. Configure & Run CHAIN
Update [config/runtime.toml](config/runtime.toml): set `backend = "http"` under both `[decision_model]` and `[generator_model]`.

```bash
# Run unit & integration tests (uses mock backend)
cargo test

# Start the CHAIN proxy on 127.0.0.1:8080
cargo run --release

# Regenerate Mondrian calibration quantiles
cargo run --release -- calibrate
```

### 5. Run Benchmarks
```bash
# Dry-run 40-turn Context-Rot Needle generator
python3 eval/bench_harness.py --dry-run

# Run against live CHAIN proxy
python3 eval/bench_harness.py --endpoint http://127.0.0.1:8080 --suite context_rot --count 10
```
