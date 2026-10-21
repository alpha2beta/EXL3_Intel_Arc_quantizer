#!/usr/bin/env python
"""Prepare a pre-tokenized calibration corpus file (P3-09 real-corpus wiring).

Task: P3-09 — the manifest pins the calibration recipe as ``wikitext-2-raw-v1``
(test split), 128 samples x 2048 sequence length (262,144 tokens), seed 42,
"deterministic non-overlapping chunks, left-truncated". This script turns a
local copy of that corpus plus the target model's tokenizer into exactly the
JSON token-ids file that ``ConversionConfig.calibration_token_ids`` consumes.

Method
------
1. Read the corpus from any of: a ``.parquet`` shard (Salesforce/wikitext
   layout, ``text`` column, rows joined with ``"\\n"`` — the rows *include*
   the blank-line entries, so the join reproduces the raw file), a single
   ``.raw``/``.txt`` file, or a directory of such files (sorted, concatenated
   with ``"\\n"``).
2. Tokenize once with the target model's tokenizer
   (``AutoTokenizer.from_pretrained``), ``add_special_tokens=False``.
3. Left-truncate: drop the leading partial chunk so the stream starts on a
   chunk boundary, then partition into deterministic non-overlapping
   ``seq_len`` chunks.
4. Pick a seed-42-driven start offset within the surplus chunks (a corpus
   with no surplus starts at 0), take ``num_rows * batch_size`` consecutive
   chunks, and flatten them into ``num_rows`` rows of
   ``batch_size * seq_len`` tokens each.
5. Write the ``exl3_quant.calibration_token_ids.v1`` JSON file with full
   provenance metadata (corpus/split names, source + tokenizer SHA-256,
   chunk counts, offset, seed).

Criterion: the output validates through ``ConversionConfig.validate()``
(shape, bounds, format stamp) without further edits.

Usage
-----
    python scripts/prepare_calibration_tokens.py \
        --input /path/to/test-00000-of-00001.parquet \
        --tokenizer /path/to/model_dir \
        --out calibration_tokens.json \
        --corpus wikitext-2-raw-v1 --split test \
        --num-rows 128 --seq-len 2048 --batch-size 1 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

FORMAT_STAMP = "exl3_quant.calibration_token_ids.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_corpus_text(input_path: Path) -> tuple[str, str]:
    """Return (text, source_kind). Rows of wikitext parquet include blank
    lines, so '\\n'.join reproduces the raw file exactly."""
    if input_path.is_dir():
        files = sorted(input_path.glob("*.raw")) + sorted(input_path.glob("*.txt"))
        if not files:
            raise SystemExit(f"no .raw/.txt files in directory: {input_path}")
        parts = [p.read_text(encoding="utf-8") for p in files]
        return "\n".join(parts), f"directory({len(files)} files)"
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(input_path)
        if "text" not in df.columns:
            raise SystemExit(f"parquet has no 'text' column: {input_path}")
        return "\n".join(df["text"].tolist()), "parquet"
    if suffix in (".raw", ".txt"):
        return input_path.read_text(encoding="utf-8"), suffix.lstrip(".")
    raise SystemExit(f"unsupported input (want .parquet/.raw/.txt/directory): {input_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="corpus .parquet/.raw/.txt or directory")
    ap.add_argument("--tokenizer", required=True, help="tokenizer source (model dir)")
    ap.add_argument("--out", required=True, help="output token-ids JSON path")
    ap.add_argument("--corpus", default="wikitext-2-raw-v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-rows", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for name, value in (("--num-rows", args.num_rows), ("--seq-len", args.seq_len),
                        ("--batch-size", args.batch_size)):
        if value < 1:
            raise SystemExit(f"{name} must be >= 1")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"input does not exist: {input_path}")

    text, source_kind = read_corpus_text(input_path)
    print(f"corpus text: {len(text)} chars from {input_path.name} ({source_kind})")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    source_tokens = len(ids)
    print(f"tokens: {source_tokens}")

    seq_len, batch, num_rows, seed = args.seq_len, args.batch_size, args.num_rows, args.seed
    needed_chunks = num_rows * batch

    # Left-truncate: drop the leading partial chunk.
    dropped = len(ids) % seq_len
    ids = ids[dropped:]
    n_chunks = len(ids) // seq_len
    if n_chunks < needed_chunks:
        raise SystemExit(
            f"corpus too small: {n_chunks} whole {seq_len}-token chunks, "
            f"need {needed_chunks} (num_rows {num_rows} x batch_size {batch})")
    surplus = n_chunks - needed_chunks
    start = random.Random(seed).randrange(surplus + 1) if surplus else 0
    print(f"chunks: {n_chunks} total, {surplus} surplus; "
          f"left-truncated {dropped} tokens; start offset {start} (seed {seed})")

    chunks = [ids[(start + i) * seq_len:(start + i + 1) * seq_len]
              for i in range(needed_chunks)]
    rows = [[t for c in chunks[r * batch:(r + 1) * batch] for t in c]
            for r in range(num_rows)]
    for r, row in enumerate(rows):
        if len(row) != seq_len * batch:
            raise SystemExit(f"internal error: row {r} has {len(row)} tokens")

    tokenizer_file = Path(args.tokenizer) / "tokenizer.json"
    payload = {
        "format": FORMAT_STAMP,
        "corpus": args.corpus,
        "split": args.split,
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": (sha256_file(tokenizer_file)
                             if tokenizer_file.is_file() else None),
        "source": str(input_path),
        "source_kind": source_kind,
        "source_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "source_chars": len(text),
        "source_tokens": source_tokens,
        "seq_len": seq_len,
        "batch_size": batch,
        "num_rows": num_rows,
        "seed": seed,
        "chunking": (f"deterministic non-overlapping {seq_len}-token chunks, "
                     f"left-truncated by {dropped} tokens, start offset {start} "
                     f"drawn from seed {seed}"),
        "start_chunk": start,
        "chunks_available": n_chunks,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prepared_by": "scripts/prepare_calibration_tokens.py",
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, "
          f"{num_rows} rows x {seq_len * batch} tokens)")
    print(f"sha256: {sha256_file(out_path)}")


if __name__ == "__main__":
    sys.exit(main())
