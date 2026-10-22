#!/usr/bin/env python
"""Teacher-forced quality evaluation of a checkpoint in the pinned runtime (P3-12).

Task: P3-12 — evaluate teacher-forced logits/NLL/perplexity of the converted
checkpoint against the floating-point baseline. The comparison is made inside
the *same* runtime (the pinned ``exl3xpu`` vLLM plugin), so the only
difference between the two runs is the checkpoint itself.

Method
------
1. Read a token-id corpus (held out from the calibration corpus) and cut it
   into fixed-length sequences.
2. Run the runtime with ``prompt_logprobs=1``: for every prompt position the
   runtime returns the log-probability of the actual token (teacher forced)
   and the top-1 log-probability (the runtime's own argmax prediction).
3. Report the teacher-forced NLL and perplexity, and (with
   ``--compare``) the token-level agreement between the two runs plus the
   per-token log-probability difference on the actual tokens.

Usage
-----
    # floating-point baseline
    python scripts/evaluate_perplexity.py --model /path/to/source \
        --eval-tokens /path/to/eval_tokens.json --out baseline.json
    # quantized checkpoint
    python scripts/evaluate_perplexity.py --model /path/to/exported \
        --quantization exl3 --eval-tokens /path/to/eval_tokens.json \
        --compare baseline.json --out quantized.json

Runtime prerequisites
---------------------
Run under the pinned runtime's environment (the exports
``vllm.xpu/scripts/launch_vllm.sh`` performs)::

    export PATH=/opt/intel/oneapi/compiler/latest/bin:$PATH
    export LD_LIBRARY_PATH=<vllm.xpu>/.venv/lib:$LD_LIBRARY_PATH
    export CPATH=<vllm.xpu>/tools/sysroot/usr/include:$CPATH

Without ``LD_LIBRARY_PATH`` the engine core dies at process-group creation
because the oneCCL V2 core ``libccl.so.2`` carries no ``RUNPATH`` and
``dlopen``s its legacy plugin ``libccl.so.1`` by bare name (both live in
``<venv>/lib``)::

    oneCCL could not be initialized! Could not load any plugin.
    ... Engine core initialization failed. Failed core proc(s): {}

Without ``CPATH`` the Triton JIT fails later, during model warm-up, because
the distro level-zero headers do not ship ``level_zero/ze_api.h`` (the
runtime vendors it under ``tools/sysroot``)::

    fatal error: 'level_zero/ze_api.h' file not found

``require_runtime_env`` fails fast with the exact export lines instead.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List


def load_sequences(path: Path, seq_len: int, num_sequences: int) -> List[List[int]]:
    payload = json.loads(path.read_text())
    tokens = payload["tokens"] if isinstance(payload, dict) else payload
    n = len(tokens) // seq_len
    if n == 0:
        raise SystemExit(f"corpus too short for seq_len={seq_len}: {len(tokens)} tokens")
    take = min(n, num_sequences)
    return [tokens[i * seq_len:(i + 1) * seq_len] for i in range(take)]


def require_runtime_env() -> None:
    """Abort early when the pinned runtime's environment exports are missing.

    Both variables are read once by their consumers (the loader reads
    ``LD_LIBRARY_PATH`` at process start, the Triton JIT reads ``CPATH`` when
    it compiles a kernel), so they cannot be repaired from inside this
    process -- report the exact prerequisite instead of letting the engine
    core abort ~20 s into startup.
    """
    import os

    lib_dir = (Path(sys.prefix) / "lib").resolve()
    if not (lib_dir / "libccl.so.1").exists():
        return  # not the XPU/oneCCL virtualenv layout; nothing to check

    problems: List[str] = []
    loader_entries = {
        Path(entry).resolve()
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if entry
    }
    if lib_dir not in loader_entries:
        problems.append(
            "oneCCL's legacy plugin (libccl.so.1) is not on the loader path; "
            "the engine core would abort with 'oneCCL could not be "
            "initialized! Could not load any plugin'.\n"
            f"    export LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH"
        )

    ze_headers = [Path("/usr/include/level_zero/ze_api.h"),
                  Path("/usr/local/include/level_zero/ze_api.h")]
    ze_headers += [
        Path(entry) / "level_zero" / "ze_api.h"
        for entry in os.environ.get("CPATH", "").split(os.pathsep)
        if entry
    ]
    if not any(path.exists() for path in ze_headers):
        sysroot = Path(sys.prefix).parent / "tools" / "sysroot" / "usr" / "include"
        problems.append(
            "level_zero/ze_api.h is not on the include path; the Triton JIT "
            "would fail during model warm-up with \"fatal error: "
            "'level_zero/ze_api.h' file not found\".\n"
            f"    export CPATH={sysroot}:$CPATH"
        )

    if problems:
        raise SystemExit(
            "missing runtime environment prerequisites:\n- "
            + "\n- ".join(problems)
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--quantization", default="",
                    help="runtime quantization method (e.g. 'exl3'); empty = none")
    ap.add_argument("--eval-tokens", required=True, help="token-id corpus JSON")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--num-sequences", type=int, default=32)
    ap.add_argument("--compare", default="", help="reference report JSON")
    ap.add_argument("--out", required=True, help="write the report here")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--language-model-only", action="store_true",
                    help="text-only mode (required for text-tower-only "
                         "evaluation of a multimodal checkpoint)")
    ap.add_argument("--dtype", default="",
                    help="runtime dtype (e.g. float16); empty = checkpoint dtype")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-num-seqs", type=int, default=0,
                    help="0 = runtime default")
    ap.add_argument("--no-prefix-caching", action="store_true")
    args = ap.parse_args()
    require_runtime_env()

    sequences = load_sequences(Path(args.eval_tokens), args.seq_len,
                                args.num_sequences)

    from vllm import LLM, SamplingParams

    kwargs = dict(model=args.model, max_model_len=args.max_model_len,
                  enforce_eager=True, disable_log_stats=True,
                  gpu_memory_utilization=args.gpu_memory_utilization)
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.language_model_only:
        kwargs["language_model_only"] = True
    if args.dtype:
        kwargs["dtype"] = args.dtype
    if args.max_num_seqs:
        kwargs["max_num_seqs"] = args.max_num_seqs
    if args.no_prefix_caching:
        kwargs["enable_prefix_caching"] = False
    llm = LLM(**kwargs)

    prompts = [{"prompt_token_ids": seq} for seq in sequences]
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0

    total_nll = 0.0
    total_tokens = 0
    per_sequence = []
    predictions: List[int] = []
    actuals: List[int] = []
    actual_logprobs: List[float] = []
    for seq, out in zip(sequences, outputs):
        seq_nll = 0.0
        seq_count = 0
        pl = out.prompt_logprobs or []
        for pos, entry in enumerate(pl):
            if not entry or pos == 0:
                continue
            token_id = seq[pos]
            info = entry.get(token_id)
            if info is None:
                continue
            seq_nll += -float(info.logprob)
            seq_count += 1
            top_id = max(entry.items(), key=lambda kv: kv[1].logprob)[0]
            predictions.append(int(top_id))
            actuals.append(int(token_id))
            actual_logprobs.append(float(info.logprob))
        per_sequence.append({"tokens": seq_count, "nll": seq_nll})
        total_nll += seq_nll
        total_tokens += seq_count

    report = {
        "model": args.model,
        "quantization": args.quantization or None,
        "dtype": args.dtype or None,
        "language_model_only": bool(args.language_model_only),
        "sequences": len(sequences),
        "seq_len": args.seq_len,
        "scored_tokens": total_tokens,
        "nll_sum": total_nll,
        "mean_nll": (total_nll / total_tokens) if total_tokens else None,
        "perplexity": (math.exp(total_nll / total_tokens)
                       if total_tokens else None),
        "elapsed_seconds": elapsed,
        "per_sequence": per_sequence,
        "predictions": predictions,
        "actuals": actuals,
        "actual_logprobs": actual_logprobs,
    }

    if args.compare:
        ref = json.loads(Path(args.compare).read_text())
        ref_pred = ref.get("predictions") or []
        ref_lp = ref.get("actual_logprobs") or []
        if len(ref_pred) == len(predictions) and predictions:
            agree = sum(1 for a, b in zip(predictions, ref_pred) if a == b)
            report["token_agreement"] = agree / len(predictions)
            report["compared_tokens"] = len(predictions)
            if len(ref_lp) == len(actual_logprobs):
                diffs = [a - b for a, b in zip(actual_logprobs, ref_lp)]
                report["mean_logprob_delta"] = sum(diffs) / len(diffs)
                report["mean_abs_logprob_delta"] = (
                    sum(abs(d) for d in diffs) / len(diffs))
        else:
            report["comparison_error"] = (
                f"reference has {len(ref_pred)} predictions, "
                f"this run has {len(predictions)}")

    Path(args.out).write_text(json.dumps(report))
    print(f"model={args.model} quantization={report['quantization']}")
    print(f"scored tokens: {total_tokens}")
    print(f"mean NLL: {report['mean_nll']:.6f}")
    print(f"perplexity: {report['perplexity']:.4f}")
    if "token_agreement" in report:
        print(f"token agreement vs {args.compare}: "
              f"{report['token_agreement'] * 100:.2f}%")
    if "mean_logprob_delta" in report:
        print(f"mean logprob delta vs reference: "
              f"{report['mean_logprob_delta']:+.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
