"""Export a Vulkan Tesseract checkpoint into a .tessviz Observatory scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tesseract_nr.scene import export_evolved_scene, export_puncture_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--observer", type=Path)
    parser.add_argument(
        "--constraints",
        action="store_true",
        help="derive full CPU ADM constraint volumes",
    )
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    if args.levels < 1 or args.levels > 8:
        parser.error("--levels must be between 1 and 8")
    if args.chunk_size < 4 or args.chunk_size > 256:
        parser.error("--chunk-size must be between 4 and 256")
    with np.load(args.checkpoint, allow_pickle=False) as archive:
        schema = str(archive["schema"])
    if schema == "vulkan-tesseract.evolved-checkpoint.v1":
        result = export_evolved_scene(
            args.checkpoint,
            args.output,
            maximum_levels=args.levels,
            chunk_shape=(args.chunk_size,) * 3,
        )
    else:
        result = export_puncture_scene(
            args.checkpoint,
            args.output,
            trajectory_path=args.trajectory,
            configuration_path=args.configuration,
            observer_path=args.observer,
            include_constraints=args.constraints,
            maximum_levels=args.levels,
            chunk_shape=(args.chunk_size,) * 3,
        )
    print(f"scene: {result.path}")
    print(f"fields: {', '.join(result.fields)}")
    print(f"geometry objects: {result.geometry_objects}")


if __name__ == "__main__":
    main()
