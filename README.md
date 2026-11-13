# CHAIN — Conformal Heterogeneous Agent Inference Network

A self-contained, local-first **Rust** agent runtime pairing a **4B System-1 decision model** (`vev-4b` / `AgentDecision-4B`) with a **`<10B` System-2 LLM** (`Qwen3.5-9B`, `Qwen3-8B`, or `Llama-3.1-8B`), connected through a closed-loop **Statistical Sensor Bus**, **Speculative `ReversibleTool` Sandbox**, and **BOCPD Context-Rot State Healing**.

---

## Architecture

1. **Token-Budgeted `StateFrame` (`src/probes/`):**
   - Every tool emits a deterministic `sensor_digest` (capped at `<= 120` tokens, zero LLM calls).
   - System-1 evaluates a compact `StateFrame` (`~300–500` tokens) plus **12 orthogonal probes** (`P_fwd`, `P_inv`, `P_pert_1..3`, `Noul` rubrics, and `Score` severity) in a single shared-prefix batch.
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

## Quick Start

### 1. Prerequisites
- **Rust:** `rustup` / `cargo` (Edition 2024)
- **Embedded `llama.cpp` backend (`--features llama-cpp`, optional):** `cmake`, `clang` / `libclang-dev`, and CUDA Toolkit (`nvcc`) for NVIDIA GPUs (or Metal on Apple Silicon).
- **HTTP backend (`backend = "http"`):** No C++ toolchain required; point [config/runtime.toml](config/runtime.toml) at two running `llama-server` / `vLLM` endpoints.

### 2. Place Models
Create a `models/` directory (gitignored) and place your GGUF weights there:

```text
chain/
├── config/runtime.toml
└── models/
    ├── vev-4b-Q4_K_M.gguf       # System-1 (4B decision model)
    └── Qwen3-8B-Q4_K_M.gguf     # System-2 (<10B generator)
```

Update `backend` (`"mock"`, `"http"`, or `"llama_cpp"`) and `model_path` in [config/runtime.toml](config/runtime.toml).

### 3. Hardware Requirements
| Mode | System-1 (4B `Q4_K_M`) | System-2 (8B–9B `Q4_K_M` + `q8_0` KV) | Recommended Hardware |
| :--- | :--- | :--- | :--- |
| **Hybrid CPU + GPU** | ~3.0–3.5 GB RAM | ~6.5–7.5 GB VRAM | 12–16 GB VRAM GPU + 16–32 GB RAM (8+ physical P-cores) |
| **Co-located GPU** | ~3.0–3.5 GB VRAM | ~6.5–7.5 GB VRAM | 24 GB+ VRAM GPU or 24 GB+ Apple Silicon Unified Memory |

### 4. Build, Test, and Run
```bash
# Run unit & integration tests (uses deterministic mock backend)
cargo test

# Run the OpenAI-compatible proxy (127.0.0.1:8080)
cargo run --release

# Or build with embedded llama-cpp-2 support
cargo run --release --features llama-cpp

# Regenerate default Mondrian calibration quantiles
cargo run -- calibrate
```

### 5. Benchmark Harness
```bash
# Dry-run 40-turn Context-Rot Needle generator
python3 eval/bench_harness.py --dry-run

# Run against a live CHAIN proxy
python3 eval/bench_harness.py --endpoint http://127.0.0.1:8080 --suite context_rot --count 10
```
