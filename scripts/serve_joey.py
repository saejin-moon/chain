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
        shared_text = f"{req.prefix}\n{req.state_frame}"
        shared_ids = tokenizer(shared_text, return_tensors="pt").input_ids.to(model.device)
        prefix_out = model(input_ids=shared_ids, use_cache=True)
        past_kv = prefix_out.past_key_values

        results: Dict[str, List[float]] = {}
        for probe in req.probes:
            suffix_ids = tokenizer(
                probe.suffix, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(model.device)
            out = model(input_ids=suffix_ids, past_key_values=past_kv, use_cache=False)
            last_logits = out.logits[0, -1, :]

            choice_logits = []
            for c in probe.choices:
                tok_ids = tokenizer.encode(c.strip(), add_special_tokens=False)
                tid = tok_ids[0] if tok_ids else tokenizer.unk_token_id
                choice_logits.append(last_logits[tid].float())

            probs = F.softmax(torch.stack(choice_logits), dim=0).tolist()
            results[probe.id] = probs

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
