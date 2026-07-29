from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tesseract_nr.scene import (
    SCENE_SCHEMA,
    SceneContractError,
    export_puncture_scene,
    export_evolved_scene,
    load_scene,
    validate_scene,
)


class TesseractSceneTests(unittest.TestCase):
    def make_checkpoint(self, root: Path) -> Path:
        shape = (8, 8, 8)
        points = int(np.prod(shape))
        state = np.zeros((points, 31), dtype=np.float32)
        state[:, 0] = 1.0
        state[:, 4] = 1.0
        state[:, 8] = 1.0
        coordinate = np.linspace(-1.0, 1.0, points, dtype=np.float32)
        state[:, 18] = 0.8 + 0.1 * coordinate
        state[:, 19] = 0.02 * coordinate
        state[:, 20] = 0.001 * coordinate
        state[:, 24] = 0.7 + 0.2 * coordinate
        target = root / "checkpoint.npz"
        np.savez_compressed(
            target,
            schema=np.array(
                "vulkan-tesseract.ccz4-vacuum-checkpoint.v1"
            ),
            shape=np.asarray(shape),
            lengths=np.asarray((3.0, 3.0, 3.0)),
            origins=np.asarray((-1.5, -1.5, -1.5)),
            derivative_order=np.array(2),
            ccz4_state=state,
            time=np.array(0.125),
            steps=np.array(4),
        )
        return target

    def test_checkpoint_exports_strict_zarr_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_checkpoint(root)
            output = root / "puncture.tessviz"
            result = export_puncture_scene(
                checkpoint,
                output,
                maximum_levels=2,
                chunk_shape=(4, 4, 4),
            )
            self.assertEqual(result.manifest["schema"], SCENE_SCHEMA)
            self.assertEqual(len(result.fields), 5)
            scene = load_scene(output)
            field = scene["frames"][0]["fields"][0]
            self.assertEqual(field["storage"], "zarr_v3_raw_chunks")
            level = field["multiscales"][0]
            self.assertEqual(level["chunk_counts"], [2, 2, 2])
            metadata = json.loads(
                (
                    output / level["path"] / "zarr.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["zarr_format"], 3)
            self.assertEqual(metadata["data_type"], "float32")
            group_metadata = json.loads(
                (
                    output
                    / Path(level["path"]).parent
                    / "zarr.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(group_metadata["node_type"], "group")
            self.assertTrue(
                (output / level["path"] / "c" / "0" / "0" / "0").is_file()
            )
            self.assertFalse(scene["provenance"]["restart_capable"])

    def test_contract_rejects_missing_coordinates(self) -> None:
        with self.assertRaises(SceneContractError):
            validate_scene(
                {
                    "schema": SCENE_SCHEMA,
                    "format_version": 1,
                    "scene_id": "bad",
                    "title": "bad",
                    "frames": [],
                    "provenance": {},
                }
            )

    def test_full_checkpoint_exports_neural_phase_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (8, 8, 8)
            points = int(np.prod(shape))
            primitive = np.zeros((points, 9), dtype=np.float32)
            primitive[:, 0] = np.log(0.1)
            primitive[:, 1] = 2.0
            primitive[:, 5] = np.log(0.005)
            primitive[: points // 2, 2] = 0.2
            ccz4 = np.zeros((points, 31), dtype=np.float32)
            ccz4[:, 0] = ccz4[:, 4] = ccz4[:, 8] = 1.0
            ccz4[:, 18] = 1.0
            ccz4[:, 24] = 1.0
            checkpoint = root / "evolved.npz"
            np.savez_compressed(
                checkpoint,
                schema=np.array(
                    "vulkan-tesseract.evolved-checkpoint.v1"
                ),
                shape=np.asarray(shape),
                lengths=np.asarray((3.0, 3.0, 3.0)),
                origins=np.asarray((-1.5, -1.5, -1.5)),
                derivative_order=np.array(2),
                constitutive_model=np.array("frozen_neural"),
                reconstruction=np.array("characteristic_weno5z"),
                primitive=primitive,
                matter_state=np.zeros((points, 11), dtype=np.float32),
                ccz4_state=ccz4,
                time=np.array(0.25),
                steps=np.array(12),
            )
            output = root / "full.tessviz"
            result = export_evolved_scene(
                checkpoint,
                output,
                maximum_levels=1,
                chunk_shape=(4, 4, 4),
            )
            self.assertIn("neural.switching_margin", result.fields)
            self.assertIn("neural.phase_two", result.fields)
            self.assertIn(
                "matter.characteristic_drift_magnitude", result.fields
            )
            scene = load_scene(output)
            self.assertEqual(
                scene["provenance"]["constitutive_model"],
                "frozen_neural",
            )
            kinds = {item["kind"] for item in scene["geometry"]}
            self.assertIn("neural_phase_front", kinds)
            self.assertIn("characteristic_geometry", kinds)


if __name__ == "__main__":
    unittest.main()
