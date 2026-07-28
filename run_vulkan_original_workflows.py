"""Qualify the retained original Slang/Vulkan execution surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
LIVE_CHECKS = (
    (
        "complete_carter_kkt_recurrence_and_vjp",
        "test_full_slang_recurrence.py",
    ),
    ("native_master_reverse_mode", "test_slang_ad.py"),
    ("native_recurrent_bptt", "test_slang_bptt.py"),
    ("theory33_constitutive_reverse", "test_theory33_slang.py"),
    ("qualified_frozen_constitutive", "test_theory33_frozen_slang.py"),
)


def qualify_original_vulkan_workflows() -> dict[str, object]:
    """Execute every retained native Vulkan validation and audit conditioning."""
    results: dict[str, object] = {}
    for name, script in LIVE_CHECKS:
        completed = subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        results[name] = {
            "script": script,
            "return_code": completed.returncode,
            "stdout": completed.stdout.strip().splitlines(),
            "passed": completed.returncode == 0,
        }

    conditioned = json.loads(
        (ROOT / "hybrid_model_conditioned.json").read_text(
            encoding="utf-8"
        )
    )["conditioning"]
    conditioned_passed = (
        int(conditioned["steps"]) >= 128
        and bool(conditioned["kkt_valid"])
        and float(conditioned["min_kkt_margin"]) > 0.0
        and float(conditioned["min_classifier_margin"]) > 0.0
        and float(conditioned["min_parity_margin"]) > 0.0
        and float(conditioned["max_lyapunov_growth"]) <= 0.0
    )
    qualification = {
        "complete_carter_kkt_recurrence_and_vjp": bool(
            results["complete_carter_kkt_recurrence_and_vjp"]["passed"]
        ),
        "native_master_reverse_mode": bool(
            results["native_master_reverse_mode"]["passed"]
        ),
        "native_recurrent_bptt": bool(
            results["native_recurrent_bptt"]["passed"]
        ),
        "theory33_constitutive_reverse": bool(
            results["theory33_constitutive_reverse"]["passed"]
        ),
        "qualified_frozen_constitutive": bool(
            results["qualified_frozen_constitutive"]["passed"]
        ),
        "conditioned_literal_128_step_audit": conditioned_passed,
    }
    return {
        "schema": "vulkan-tesseract.original-workflows.v1",
        "backend": "Slang/Vulkan",
        "live_checks": results,
        "conditioned_literal_audit": conditioned,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_original_workflows_results.json"),
    )
    args = parser.parse_args()
    report = qualify_original_vulkan_workflows()
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["qualification"], indent=2))
    print(f"qualified={report['qualified']}")
    print(f"wrote {args.output}")
    if not report["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
