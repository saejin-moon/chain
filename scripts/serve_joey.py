#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.0.0",
#   "fastapi>=0.115.0",
#   "pydantic>=2.0.0",
#   "torch>=2.5.0",
#   "transformers>=5.0.0",
#   "uvicorn>=0.32.0",
# ]
# ///
"""
Lightweight System-1 HTTP server for `baselabs/joey-1` (2B Qwen3.5-based decision model).
Exposes POST /v1/systemone/batch for CHAIN's 12-probe orthogonal sensor bus.
"""

import argparse
import copy
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForMultimodalLM as JoeyAutoModel
except ImportError:
    try:
        from transformers import AutoModelForImageTextToText as JoeyAutoModel
    except ImportError:
        JoeyAutoModel = AutoModelForCausalLM


class ProbeReq(BaseModel):
    id: str
    suffix: str
    choices: List[str]


class BatchReq(BaseModel):
    prefix: str
    state_frame: str
    probes: List[ProbeReq]


def clone_hybrid_cache(past_kv):
    """Clone Qwen3.5 hybrid Mamba + Attention cache across turns without re-prefilling."""
    return copy.deepcopy(past_kv)


def create_app(model_id: str, device: str, dtype_str: str) -> FastAPI:
    app = FastAPI(title="CHAIN System-1 (joey-1)")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]

    print(f"Loading {model_id} on {device} ({dtype_str})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = JoeyAutoModel.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_id}

    @app.post("/v1/systemone/batch")
    @torch.no_grad()
    def eval_batch(req: BatchReq) -> Dict[str, Dict[str, List[float]]]:
        t0 = time.perf_counter()
        if not req.probes:
            return {"results": {}}

        # 1. Prefill shared prefix + StateFrame ONCE (~300 tokens)
        shared_text = f"{req.prefix}\n{req.state_frame}"
        shared_ids = tokenizer(shared_text, return_tensors="pt").input_ids.to(model.device)
        prefix_out = model(input_ids=shared_ids, use_cache=True)
        base_kv = prefix_out.past_key_values

        raw_logits_map: Dict[str, torch.Tensor] = {}
        # 2. Evaluate each short suffix (~12 tokens) reusing cloned base_kv
        for probe in req.probes:
            suffix_ids = tokenizer(
                probe.suffix, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(model.device)
            kv_copy = clone_hybrid_cache(base_kv)
            out = model(input_ids=suffix_ids, past_key_values=kv_copy, use_cache=False)
            last_logits = out.logits[0, -1, :].float()

            choice_logits = []
            for c in probe.choices:
                c_clean = c.strip()
                ids_no_sp = tokenizer.encode(c_clean, add_special_tokens=False)
                ids_sp = tokenizer.encode(" " + c_clean, add_special_tokens=False)
                candidates_tids = []
                if ids_no_sp:
                    candidates_tids.append(ids_no_sp[0])
                if ids_sp:
                    candidates_tids.append(ids_sp[0])
                if not candidates_tids:
                    candidates_tids.append(tokenizer.unk_token_id or 0)
                best_logit = max(last_logits[tid] for tid in candidates_tids)
                choice_logits.append(best_logit)

            raw_logits_map[probe.id] = torch.stack(choice_logits)

        # Contextual prior calibration across choice probes (removes position-"A" token bias)
        choice_probe_ids = [
            pid for pid in ("p_fwd", "p_inv", "p_pert_1", "p_pert_2", "p_pert_3")
            if pid in raw_logits_map
        ]
        if len(choice_probe_ids) >= 2:
            mean_prior = torch.stack([raw_logits_map[pid] for pid in choice_probe_ids]).mean(dim=0)
            for pid in choice_probe_ids:
                if pid == "p_inv":
                    # Inverse probe: flip deviation from prior so it doesn't falsely mirror p_fwd's argmax
                    raw_logits_map[pid] = -(raw_logits_map[pid] - mean_prior)
                else:
                    raw_logits_map[pid] = raw_logits_map[pid] - 0.75 * mean_prior

        results: Dict[str, List[float]] = {}
        for probe in req.probes:
            logits_vec = raw_logits_map[probe.id]
            temp = 0.25 if probe.id.startswith("p_") else 0.50
            probs = F.softmax(logits_vec / temp, dim=0).tolist()
            results[probe.id] = probs

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        fwd = [round(x, 3) for x in results.get("p_fwd", [])]
        inv = [round(x, 3) for x in results.get("p_inv", [])]
        pre = round(results.get("r_precond", [0.0])[0], 3)
        rev = round(results.get("r_reversible", [0.0])[0], 3)
        prog = round(results.get("r_progress", [0.0])[0], 3)
        print(
            f"[joey-1] {len(req.probes)} probes in {elapsed_ms:.1f}ms | "
            f"p_fwd={fwd} p_inv={inv} r_pre={pre} r_rev={rev} r_prog={prog}",
            flush=True,
        )
        return {"results": results}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve baselabs/joey-1 for CHAIN")
    parser.add_argument("--model", default="baselabs/joey-1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    app = create_app(args.model, args.device, args.dtype)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
