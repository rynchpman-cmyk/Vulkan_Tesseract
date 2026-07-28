"""Qualify resident evolution of both mixed massive-vector fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesseract_nr.vulkan_validation import (
    qualify_vulkan_evolved_coupled_stack,
    qualify_vulkan_mixed_proca_evolution,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="8,8")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulkan_mixed_proca_results.json"),
    )
    args = parser.parse_args()
    subsystem = qualify_vulkan_mixed_proca_evolution(
        shape=tuple(int(value) for value in args.shape.split(",")),
        steps=args.steps,
        dt=args.dt,
    )
    coupled = qualify_vulkan_evolved_coupled_stack()
    qualification = {
        **{
            f"subsystem_{name}": bool(value)
            for name, value in subsystem["qualification"].items()
        },
        **{
            f"coupled_{name}": bool(value)
            for name, value in coupled["qualification"].items()
        },
    }
    report = {
        "schema": "vulkan-tesseract.mixed-proca-frontier.v1",
        "backend": "Slang/Vulkan",
        "subsystem": subsystem,
        "coupled_stack": coupled,
        "qualification": qualification,
        "qualified": all(qualification.values()),
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["qualification"], indent=2))
    print(f"qualified={report['qualified']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
