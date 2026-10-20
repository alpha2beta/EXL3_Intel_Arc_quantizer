#!/usr/bin/env python
"""Run the P0-18 / P0-19 XPU probes and persist results to artifacts.

Usage:
    PYTHONPATH=. /path/to/python scripts/probe_xpu_extension.py

Outputs:
    artifacts/environment/device_capabilities.json  (P0-19)
    artifacts/environment/extension_probe.json      (P0-18 verification + O-6)
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exl3_quant import xpu_probe  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts" / "environment"


def main() -> int:
    results = xpu_probe.run_all()
    if not results.get("xpu_available"):
        print(" !! torch.xpu is not available; nothing probed")
        return 1

    print(json.dumps({
        k: v for k, v in results.items() if k != "capabilities"
    }, indent=2, default=str))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    capabilities = results.get("capabilities", {})
    (ARTIFACTS / "device_capabilities.json").write_text(
        json.dumps(capabilities, indent=2, sort_keys=True) + "\n")
    (ARTIFACTS / "extension_probe.json").write_text(
        json.dumps({k: results[k] for k in (
            "xpu_available", "build_seconds", "device_selection",
            "output_correctness", "queue_ordering", "tensor_lifetime",
            "hfma_o6_probe")}, indent=2, sort_keys=True, default=str) + "\n")
    print(f" << wrote {ARTIFACTS / 'device_capabilities.json'}")
    print(f" << wrote {ARTIFACTS / 'extension_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
