"""Live regression for resident multidimensional Carter/CCZ4 evolution."""

from __future__ import annotations

from tesseract_nr.vulkan_validation import (
    qualify_vulkan_multidimensional_full_stack,
)


def main() -> None:
    report = qualify_vulkan_multidimensional_full_stack(steps=1)
    failed = [
        name
        for name, passed in report["qualification"].items()
        if not passed
    ]
    if failed:
        raise RuntimeError(
            f"multidimensional Vulkan failures: {failed}"
        )
    print("curved resident Carter recovery/transport: qualified")
    print("Strang implicit source schedule: qualified")
    print("derivative CCZ4 vacuum/matter evolution: qualified")


if __name__ == "__main__":
    main()
