from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen

from tesseract_studio import (
    PRESET_SCHEMA,
    ROOT,
    StudioHandler,
    StudioHTTPServer,
    StudioJobManager,
    execute_lab_command,
    load_protocols,
    load_presets,
)


class StudioPresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.presets = load_presets()

    def test_preset_registry_is_friendly_and_allow_listed(self) -> None:
        self.assertGreaterEqual(len(self.presets), 5)
        self.assertIn("puncture-production", self.presets)
        self.assertIn("neural-phase-collision", self.presets)
        for identifier, preset in self.presets.items():
            with self.subTest(preset=identifier):
                self.assertEqual(
                    PRESET_SCHEMA,
                    json.loads(
                        (
                            ROOT / "studio" / "presets" / f"{identifier}.json"
                        ).read_text(encoding="utf-8")
                    )["schema"],
                )
                public = preset.public()
                self.assertTrue(public["title"])
                self.assertTrue(public["what_you_will_see"])
                self.assertGreater(public["estimated_seconds"], 0)
                self.assertNotIn("arguments", public)

    def test_production_preset_has_exact_accessibility_contract(self) -> None:
        preset = self.presets["puncture-production"]
        self.assertEqual(preset.total_steps, 500)
        self.assertEqual(preset.minimum_vram_gib, 3.0)
        self.assertIn("-CheckpointCadence", preset.arguments)
        self.assertNotIn("-SaveSnapshots", preset.arguments)
        self.assertIsNotNone(preset.resume_checkpoint)

    def test_neural_phase_preset_uses_qualified_reference_ladder(self) -> None:
        preset = self.presets["neural-phase-collision"]
        resolutions = preset.arguments[
            preset.arguments.index("-Resolutions") + 1
        ]
        final_time = preset.arguments[
            preset.arguments.index("-FinalTime") + 1
        ]
        self.assertEqual(resolutions, "16,32,64,128")
        self.assertEqual(final_time, "0.02")

    def test_invalid_neural_phase_shortcut_is_rejected(self) -> None:
        source = (
            ROOT / "studio" / "presets" / "neural-phase-collision.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        index = payload["arguments"].index("-Resolutions") + 1
        payload["arguments"][index] = "16,32"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "invalid.json"
            target.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "qualified 16,32,64,128 ladder"
            ):
                type(self.presets["neural-phase-collision"]).load(target)

    def test_laboratory_protocols_are_typed_and_range_checked(self) -> None:
        protocols = load_protocols()
        self.assertEqual(
            set(protocols),
            {"neural-phase", "puncture", "spacetime", "visualizer"},
        )
        spacetime = protocols["spacetime"]
        values = spacetime.normalize({"nx": "12", "ny": 8, "steps": "3"})
        self.assertEqual(values, {"nx": 12, "ny": 8, "steps": 3})
        with self.assertRaisesRegex(ValueError, "at most 128"):
            spacetime.normalize({"nx": 129})
        with self.assertRaisesRegex(ValueError, "unknown"):
            spacetime.normalize({"shell": "whoami"})
        with self.assertRaisesRegex(ValueError, "factor-two"):
            protocols["neural-phase"].normalize(
                {"resolutions": "16,32,96"}
            )
        manager = StudioJobManager(
            self.presets, protocols=protocols, demo=True
        )
        with self.assertRaisesRegex(ValueError, "unknown laboratory command"):
            execute_lab_command(manager, "powershell whoami")

    def test_result_analyzer_reads_qualification_and_ladders(self) -> None:
        manager = StudioJobManager(self.presets, protocols=load_protocols())
        catalog = manager.result_catalog()
        row = next(
            item
            for item in catalog
            if item["file"]
            == "vulkan_original_neural_phase_full_roe_results.json"
        )
        analysis = manager.analyze_result(row["id"])
        self.assertTrue(analysis["qualified"])
        self.assertGreaterEqual(analysis["qualification"]["total"], 20)
        self.assertTrue(
            any(item["axis"] == "points" for item in analysis["series"])
        )
        self.assertEqual(
            analysis["provenance"]["schema"],
            "vulkan-tesseract.full-neural-phase-riemann.v1",
        )

    def test_demo_job_emits_progress_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = StudioJobManager(
                self.presets,
                demo=True,
                run_root=Path(directory),
            )
            started = manager.start("spacetime-smoke")
            self.assertIn(started["status"], {"starting", "running"})
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                snapshot = manager.snapshot()
                if snapshot["status"] == "complete":
                    break
                time.sleep(0.05)
            else:
                self.fail("demo job did not complete")
            self.assertEqual(snapshot["progress"], 1.0)
            self.assertTrue(snapshot["events"])
            self.assertTrue(snapshot["artifacts"])
            self.assertNotIn("command", snapshot)
            self.assertNotIn("arguments", snapshot)
            self.assertTrue(Path(snapshot["artifacts"][0]["path"]).is_file())
            result = next(
                row
                for row in manager.result_catalog()
                if row["file"] == "demo_result.json"
            )
            analysis = manager.analyze_result(result["id"])
            self.assertTrue(analysis["qualified"])
            self.assertEqual(
                analysis["provenance"]["source"], "studio_run"
            )


class StudioHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.manager = StudioJobManager(
            load_presets(),
            demo=True,
            run_root=Path(self.temporary.name),
        )
        self.server = StudioHTTPServer(
            ("127.0.0.1", 0), StudioHandler, self.manager
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        if self.manager.snapshot()["status"] in {"starting", "running"}:
            self.manager.cancel()
            deadline = time.monotonic() + 3.0
            while (
                self.manager.snapshot()["status"] in {"starting", "running"}
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.temporary.cleanup()

    def get(self, path: str) -> tuple[int, bytes, str]:
        with urlopen(self.base + path, timeout=3) as response:
            return (
                response.status,
                response.read(),
                response.headers.get_content_type(),
            )

    def post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        request = Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            return json.loads(response.read())

    def test_static_application_and_api_are_runnable(self) -> None:
        status, body, content_type = self.get("/")
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/html")
        self.assertIn(b"Tesseract Studio", body)
        self.assertIn(b"Advanced mode", body)
        self.assertIn(b"Tesseract Laboratory", body)
        self.assertNotIn(b"Science view", body)
        self.assertEqual(self.get("/static/studio.css")[0], 200)
        self.assertEqual(self.get("/static/studio.js")[0], 200)

        presets = json.loads(self.get("/api/presets")[1])
        self.assertTrue(presets["ok"])
        self.assertGreaterEqual(len(presets["presets"]), 5)
        protocols = json.loads(self.get("/api/protocols")[1])
        self.assertEqual(len(protocols["protocols"]), 4)
        results = json.loads(self.get("/api/results")[1])
        self.assertGreater(len(results["results"]), 20)
        result_id = results["results"][0]["id"]
        analysis = json.loads(self.get(f"/api/results/{result_id}")[1])
        self.assertTrue(analysis["ok"])
        self.assertIn("provenance", analysis["analysis"])

        started = self.post(
            "/api/jobs/start", {"preset_id": "puncture-plan"}
        )
        self.assertTrue(started["ok"])
        self.assertEqual(started["job"]["preset_id"], "puncture-plan")
        time.sleep(0.15)
        status_payload = json.loads(self.get("/api/status")[1])
        self.assertTrue(status_payload["ok"])
        self.assertIn(
            status_payload["job"]["status"],
            {"running", "complete"},
        )

    def test_advanced_launch_and_console_are_runnable(self) -> None:
        started = self.post(
            "/api/jobs/start-advanced",
            {
                "protocol_id": "spacetime",
                "parameters": {"nx": 8, "ny": 8, "steps": 3},
            },
        )
        self.assertTrue(started["ok"])
        job = started["job"]
        self.assertEqual(job["mode"], "advanced")
        self.assertEqual(job["protocol_id"], "spacetime")
        self.assertEqual(job["parameters"]["steps"], 3)
        self.assertIn("run_vulkan_multidimensional.ps1", job["command_preview"])
        self.assertNotIn("powershell", job["command_preview"].lower())

        status = self.post("/api/console", {"command": "status"})
        self.assertTrue(status["ok"])
        self.assertEqual(status["action"], "status")
        help_payload = self.post("/api/console", {"command": "help"})
        self.assertIn("not an operating-system shell", help_payload["lines"][1]["text"])


if __name__ == "__main__":
    unittest.main()
