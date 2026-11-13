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

Run in an isolated uv environment (avoids vllm's transformers v4 pin):
  uv run --no-project scripts/serve_joey.py --model baselabs/joey-1 --port 8081 --device cpu --dtype fp32
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


def create_app(model_id: str, device: str, dtype_str: str) -> FastAPI:
    app = FastAPI(title="CHAIN System-1 (joey-1)")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]

    print(f"Loading {model_id} on {device} ({dtype_str})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = JoeyAutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
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

        shared_text = f"{req.prefix}\n{req.state_frame}"
        shared_ids = tokenizer(shared_text, return_tensors="pt").input_ids.to(model.device)
        bsz = len(req.probes)
        pad_id = tokenizer.pad_token_id or 0

        suffix_lists = [
            tokenizer.encode(p.suffix, add_special_tokens=False) for p in req.probes
        ]
        max_suf = max(len(s) for s in suffix_lists)
        suffix_tensor = torch.full((bsz, max_suf), pad_id, dtype=torch.long, device=model.device)
        suffix_mask = torch.zeros((bsz, max_suf), dtype=torch.long, device=model.device)
        last_indices = []
        for i, s in enumerate(suffix_lists):
            slen = len(s)
            suffix_tensor[i, :slen] = torch.tensor(s, dtype=torch.long, device=model.device)
            suffix_mask[i, :slen] = 1
            last_indices.append(max(slen - 1, 0))

        # Try single batched forward pass using shared prefix KV cache; fall back to full padded batch.
        try:
            prefix_out = model(input_ids=shared_ids, use_cache=True)
            past_kv = copy.deepcopy(prefix_out.past_key_values)
            if hasattr(past_kv, "batch_repeat_interleave"):
                past_kv.batch_repeat_interleave(bsz)
            else:
                raise AttributeError("DynamicCache lacks batch_repeat_interleave")
            prefix_mask = torch.ones(
                (bsz, shared_ids.shape[1]), dtype=torch.long, device=model.device
            )
            full_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
            out = model(
                input_ids=suffix_tensor,
                attention_mask=full_mask,
                past_key_values=past_kv,
                use_cache=False,
            )
        except Exception:
            shared_batch = shared_ids.expand(bsz, -1)
            full_ids = torch.cat([shared_batch, suffix_tensor], dim=1)
            prefix_mask = torch.ones_like(shared_batch)
            full_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
            last_indices = [shared_ids.shape[1] + idx for idx in last_indices]
            out = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)

        results: Dict[str, List[float]] = {}
        for i, probe in enumerate(req.probes):
            last_logits = out.logits[i, last_indices[i], :]
            choice_logits = []
            for c in probe.choices:
                tok_ids = tokenizer.encode(c.strip(), add_special_tokens=False)
                tid = tok_ids[0] if tok_ids else tokenizer.unk_token_id
                choice_logits.append(last_logits[tid].float())
            probs = F.softmax(torch.stack(choice_logits), dim=0).tolist()
            results[probe.id] = probs

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[joey-1] batched {bsz} probes in {elapsed_ms:.1f}ms")
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
