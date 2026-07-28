"""Local, zero-terminal control surface for Vulkan Tesseract.

Tesseract Studio deliberately exposes only repository-owned preset manifests.
It never accepts an arbitrary executable, script path, or shell fragment from
the browser.  Existing PowerShell runners remain the canonical command-line
surface; Studio adds preflight, process supervision, progress events, and a
friendly local web interface over those same entry points.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4
import webbrowser


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "studio" / "static"
PRESET_ROOT = ROOT / "studio" / "presets"
PROTOCOL_ROOT = ROOT / "studio" / "protocols"
RUN_ROOT = ROOT / "runs" / "studio"
PRESET_SCHEMA = "vulkan-tesseract.studio-preset.v1"
PROTOCOL_SCHEMA = "vulkan-tesseract.lab-protocol.v1"
STATE_SCHEMA = "vulkan-tesseract.studio-job.v1"

RUNNERS = {
    "visualization": ROOT / "run_visualization.ps1",
    "neural_phase": ROOT / "run_vulkan_original_neural_phase_riemann.ps1",
    "multidimensional": ROOT / "run_vulkan_multidimensional.ps1",
    "puncture_mots": ROOT / "run_vulkan_puncture_mots.ps1",
}

STEP_PATTERN = re.compile(r"\bstep=(\d+)\b")
TIME_PATTERN = re.compile(r"\bt=([0-9.eE+-]+)\b")
SAFE_TOKEN_PATTERN = re.compile(r"^[^\r\n\x00]{1,500}$")
PLACEHOLDERS = {"{job_dir}", "{project_root}"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def _safe_child(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(parent.resolve())
    return resolved


@dataclass(frozen=True)
class StudioPreset:
    id: str
    title: str
    eyebrow: str
    summary: str
    what_you_will_see: str
    runner: str
    arguments: tuple[str, ...]
    duration_label: str
    estimated_seconds: int
    minimum_vram_gib: float
    disk_label: str
    tier: str
    visualizer: str
    accent: str
    total_steps: int | None
    result_files: tuple[str, ...]
    resume_checkpoint: str | None = None

    @classmethod
    def load(cls, path: Path) -> "StudioPreset":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != PRESET_SCHEMA:
            raise ValueError(f"{path.name}: unsupported preset schema")
        required = {
            "id",
            "title",
            "eyebrow",
            "summary",
            "what_you_will_see",
            "runner",
            "arguments",
            "duration_label",
            "estimated_seconds",
            "minimum_vram_gib",
            "disk_label",
            "tier",
            "visualizer",
            "accent",
            "total_steps",
            "result_files",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(
                f"{path.name}: missing preset fields {sorted(missing)}"
            )
        identifier = str(payload["id"])
        if not re.fullmatch(r"[a-z][a-z0-9-]{2,60}", identifier):
            raise ValueError(f"{path.name}: invalid preset id")
        runner = str(payload["runner"])
        if runner not in RUNNERS:
            raise ValueError(f"{path.name}: unknown runner {runner!r}")
        arguments = tuple(str(value) for value in payload["arguments"])
        if any(not SAFE_TOKEN_PATTERN.fullmatch(value) for value in arguments):
            raise ValueError(f"{path.name}: unsafe command argument")
        for value in arguments:
            for match in re.findall(r"\{[^{}]+\}", value):
                if match not in PLACEHOLDERS:
                    raise ValueError(
                        f"{path.name}: unsupported placeholder {match}"
                    )
        if runner == "neural_phase":
            try:
                resolution_text = arguments[
                    arguments.index("-Resolutions") + 1
                ]
                final_time_text = arguments[
                    arguments.index("-FinalTime") + 1
                ]
                resolutions = tuple(
                    int(value) for value in resolution_text.split(",")
                )
                final_time = float(final_time_text)
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"{path.name}: invalid neural-phase contract"
                ) from exc
            if (
                resolutions != (16, 32, 64, 128)
                or final_time != 0.02
            ):
                raise ValueError(
                    f"{path.name}: neural-phase preset must use the "
                    "qualified 16,32,64,128 ladder at t=0.02"
                )
        total_steps = payload["total_steps"]
        if total_steps is not None:
            total_steps = int(total_steps)
            if total_steps < 1:
                raise ValueError(f"{path.name}: total_steps must be positive")
        estimated_seconds = int(payload["estimated_seconds"])
        if estimated_seconds < 1:
            raise ValueError(
                f"{path.name}: estimated_seconds must be positive"
            )
        result_files = tuple(str(value) for value in payload["result_files"])
        if any(
            not SAFE_TOKEN_PATTERN.fullmatch(value) for value in result_files
        ):
            raise ValueError(f"{path.name}: unsafe result path")
        checkpoint = payload.get("resume_checkpoint")
        return cls(
            id=identifier,
            title=str(payload["title"]),
            eyebrow=str(payload["eyebrow"]),
            summary=str(payload["summary"]),
            what_you_will_see=str(payload["what_you_will_see"]),
            runner=runner,
            arguments=arguments,
            duration_label=str(payload["duration_label"]),
            estimated_seconds=estimated_seconds,
            minimum_vram_gib=float(payload["minimum_vram_gib"]),
            disk_label=str(payload["disk_label"]),
            tier=str(payload["tier"]),
            visualizer=str(payload["visualizer"]),
            accent=str(payload["accent"]),
            total_steps=total_steps,
            result_files=result_files,
            resume_checkpoint=(
                None if checkpoint is None else str(checkpoint)
            ),
        )

    def checkpoint_path(self) -> Path | None:
        if self.resume_checkpoint is None:
            return None
        return _safe_child(ROOT / self.resume_checkpoint, ROOT)

    def public(self) -> dict[str, Any]:
        checkpoint = self.checkpoint_path()
        return {
            "id": self.id,
            "title": self.title,
            "eyebrow": self.eyebrow,
            "summary": self.summary,
            "what_you_will_see": self.what_you_will_see,
            "duration_label": self.duration_label,
            "estimated_seconds": self.estimated_seconds,
            "minimum_vram_gib": self.minimum_vram_gib,
            "disk_label": self.disk_label,
            "tier": self.tier,
            "visualizer": self.visualizer,
            "accent": self.accent,
            "total_steps": self.total_steps,
            "resume_available": bool(
                checkpoint is not None and checkpoint.is_file()
            ),
        }


def load_presets(path: Path = PRESET_ROOT) -> dict[str, StudioPreset]:
    presets: dict[str, StudioPreset] = {}
    for source in sorted(path.glob("*.json")):
        preset = StudioPreset.load(source)
        if preset.id in presets:
            raise ValueError(f"duplicate preset id {preset.id!r}")
        presets[preset.id] = preset
    if not presets:
        raise ValueError("Tesseract Studio found no preset manifests")
    return presets


@dataclass(frozen=True)
class LabParameter:
    key: str
    label: str
    option: str
    type: str
    default: Any
    help: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    minimum_items: int = 1

    @classmethod
    def load(cls, payload: dict[str, Any], source: Path) -> "LabParameter":
        parameter_type = str(payload.get("type"))
        allowed = {
            "integer",
            "float",
            "boolean",
            "choice",
            "factor_two_csv",
        }
        if parameter_type not in allowed:
            raise ValueError(
                f"{source.name}: unsupported parameter type {parameter_type}"
            )
        key = str(payload.get("key", ""))
        option = str(payload.get("option", ""))
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", key):
            raise ValueError(f"{source.name}: invalid parameter key")
        if not re.fullmatch(r"-[A-Za-z][A-Za-z0-9]+", option):
            raise ValueError(f"{source.name}: invalid runner option")
        choices = tuple(str(value) for value in payload.get("choices", ()))
        parameter = cls(
            key=key,
            label=str(payload.get("label", key)),
            option=option,
            type=parameter_type,
            default=payload.get("default"),
            help=str(payload.get("help", "")),
            minimum=(
                None
                if payload.get("minimum") is None
                else float(payload["minimum"])
            ),
            maximum=(
                None
                if payload.get("maximum") is None
                else float(payload["maximum"])
            ),
            step=(
                None if payload.get("step") is None else float(payload["step"])
            ),
            choices=choices,
            minimum_items=int(payload.get("minimum_items", 1)),
        )
        parameter.coerce(parameter.default)
        return parameter

    def coerce(self, raw: Any) -> Any:
        if self.type == "boolean":
            if isinstance(raw, bool):
                return raw
            value = str(raw).strip().lower()
            if value in {"true", "1", "yes", "on"}:
                return True
            if value in {"false", "0", "no", "off"}:
                return False
            raise ValueError(f"{self.label} must be true or false")
        if self.type == "integer":
            if isinstance(raw, bool):
                raise ValueError(f"{self.label} must be an integer")
            try:
                value = int(str(raw))
            except ValueError as exc:
                raise ValueError(f"{self.label} must be an integer") from exc
            self._range(value)
            return value
        if self.type == "float":
            if isinstance(raw, bool):
                raise ValueError(f"{self.label} must be a number")
            try:
                value = float(str(raw))
            except ValueError as exc:
                raise ValueError(f"{self.label} must be a number") from exc
            if not (float("-inf") < value < float("inf")):
                raise ValueError(f"{self.label} must be finite")
            self._range(value)
            return value
        if self.type == "choice":
            value = str(raw)
            if value not in self.choices:
                raise ValueError(
                    f"{self.label} must be one of {', '.join(self.choices)}"
                )
            return value
        value = str(raw).replace(" ", "")
        try:
            ladder = tuple(int(item) for item in value.split(","))
        except ValueError as exc:
            raise ValueError(
                f"{self.label} must be comma-separated integers"
            ) from exc
        if len(ladder) < self.minimum_items:
            raise ValueError(
                f"{self.label} needs at least {self.minimum_items} levels"
            )
        if any(
            fine != 2 * coarse
            for coarse, fine in zip(ladder, ladder[1:])
        ):
            raise ValueError(f"{self.label} must be a factor-two ladder")
        for item in ladder:
            self._range(item)
        return ",".join(str(item) for item in ladder)

    def _range(self, value: float) -> None:
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.label} must be at least {self.minimum:g}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.label} must be at most {self.maximum:g}")

    def public(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "option": self.option,
            "type": self.type,
            "default": self.default,
            "help": self.help,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "choices": list(self.choices),
            "minimum_items": self.minimum_items,
        }


@dataclass(frozen=True)
class LabProtocol:
    id: str
    title: str
    summary: str
    runner: str
    visualizer: str
    accent: str
    estimated_seconds: int
    minimum_vram_gib: float
    fixed_arguments: tuple[str, ...]
    output_option: str
    output_name: str
    parameters: tuple[LabParameter, ...]
    qualification_note: str
    total_steps_parameter: str | None = None

    @classmethod
    def load(cls, path: Path) -> "LabProtocol":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != PROTOCOL_SCHEMA:
            raise ValueError(f"{path.name}: unsupported protocol schema")
        identifier = str(payload.get("id", ""))
        if not re.fullmatch(r"[a-z][a-z0-9-]{2,60}", identifier):
            raise ValueError(f"{path.name}: invalid protocol id")
        runner = str(payload.get("runner", ""))
        if runner not in RUNNERS:
            raise ValueError(f"{path.name}: unknown runner {runner!r}")
        parameters = tuple(
            LabParameter.load(value, path)
            for value in payload.get("parameters", ())
        )
        if not parameters:
            raise ValueError(f"{path.name}: protocol has no parameters")
        keys = [parameter.key for parameter in parameters]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{path.name}: duplicate parameter key")
        fixed = tuple(str(value) for value in payload.get("fixed_arguments", ()))
        if any(not SAFE_TOKEN_PATTERN.fullmatch(value) for value in fixed):
            raise ValueError(f"{path.name}: unsafe fixed argument")
        output = payload.get("output")
        if not isinstance(output, dict):
            raise ValueError(f"{path.name}: missing output contract")
        output_option = str(output.get("option", ""))
        if output_option not in {"-Output", "-OutputDirectory"}:
            raise ValueError(f"{path.name}: invalid output option")
        output_name = str(output.get("name", ""))
        if (
            not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", output_name)
            or ".." in output_name
        ):
            raise ValueError(f"{path.name}: invalid output name")
        total_steps_parameter = payload.get("total_steps_parameter")
        if (
            total_steps_parameter is not None
            and str(total_steps_parameter) not in keys
        ):
            raise ValueError(f"{path.name}: invalid total-steps parameter")
        return cls(
            id=identifier,
            title=str(payload.get("title", identifier)),
            summary=str(payload.get("summary", "")),
            runner=runner,
            visualizer=str(payload.get("visualizer", "tesseract")),
            accent=str(payload.get("accent", "#67e8f9")),
            estimated_seconds=max(1, int(payload.get("estimated_seconds", 60))),
            minimum_vram_gib=float(payload.get("minimum_vram_gib", 0)),
            fixed_arguments=fixed,
            output_option=output_option,
            output_name=output_name,
            parameters=parameters,
            qualification_note=str(payload.get("qualification_note", "")),
            total_steps_parameter=(
                None
                if total_steps_parameter is None
                else str(total_steps_parameter)
            ),
        )

    def normalize(self, supplied: dict[str, Any] | None) -> dict[str, Any]:
        values = {} if supplied is None else dict(supplied)
        allowed = {parameter.key for parameter in self.parameters}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                f"unknown {self.id} parameter(s): {', '.join(sorted(unknown))}"
            )
        return {
            parameter.key: parameter.coerce(
                values.get(parameter.key, parameter.default)
            )
            for parameter in self.parameters
        }

    def arguments(
        self, values: dict[str, Any], output: str
    ) -> tuple[str, ...]:
        arguments: list[str] = []
        for parameter in self.parameters:
            value = values[parameter.key]
            if parameter.type == "boolean":
                if value:
                    arguments.append(parameter.option)
                continue
            arguments.extend([parameter.option, str(value)])
        arguments.extend(self.fixed_arguments)
        arguments.extend([self.output_option, output])
        return tuple(arguments)

    def total_steps(self, values: dict[str, Any]) -> int | None:
        if self.id == "puncture" and bool(values.get("plan_only")):
            return None
        if self.total_steps_parameter is None:
            return None
        return int(values[self.total_steps_parameter])

    def estimate(self, values: dict[str, Any]) -> int:
        if self.id == "spacetime":
            scale = (
                int(values["nx"])
                * int(values["ny"])
                * int(values["steps"])
                / (8 * 8 * 2)
            )
            return max(10, round(22 * scale))
        if self.id == "puncture":
            if values["plan_only"]:
                return 5
            scale = (
                (int(values["resolution"]) / 120) ** 3
                * int(values["steps"])
                / 500
                * 0.04
                / float(values["cfl"])
            )
            return max(60, round(86400 * scale))
        if self.id == "neural-phase":
            ladder = sum(int(item) for item in values["resolutions"].split(","))
            scale = (
                ladder
                / (16 + 32 + 64 + 128)
                * float(values["final_time"])
                / 0.02
                * 0.12
                / float(values["cfl"])
            )
            return max(15, round(40 * scale))
        if self.id == "visualizer":
            scale = (
                int(values["samples"])
                * int(values["steps"])
                / (320 * 64)
            )
            return max(3, round(8 * scale))
        return self.estimated_seconds

    def result_files(self) -> tuple[str, ...]:
        if self.output_option == "-Output":
            return (self.output_name,)
        return (
            f"{self.output_name}/configuration.json",
            f"{self.output_name}/summary.json",
            f"{self.output_name}/trajectory.json",
            f"{self.output_name}/final_checkpoint.npz",
        )

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "runner": RUNNERS[self.runner].name,
            "visualizer": self.visualizer,
            "accent": self.accent,
            "estimated_seconds": self.estimated_seconds,
            "minimum_vram_gib": self.minimum_vram_gib,
            "parameters": [parameter.public() for parameter in self.parameters],
            "qualification_note": self.qualification_note,
            "command_grammar": (
                f"run {self.id} "
                + " ".join(
                    f"{parameter.key}=<{parameter.type}>"
                    for parameter in self.parameters
                )
            ),
        }


def load_protocols(path: Path = PROTOCOL_ROOT) -> dict[str, LabProtocol]:
    protocols: dict[str, LabProtocol] = {}
    for source in sorted(path.glob("*.json")):
        protocol = LabProtocol.load(source)
        if protocol.id in protocols:
            raise ValueError(f"duplicate protocol id {protocol.id!r}")
        protocols[protocol.id] = protocol
    if not protocols:
        raise ValueError("Tesseract Studio found no laboratory protocols")
    return protocols


def powershell_executable() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def preflight() -> dict[str, Any]:
    python = Path(sys.executable)
    shader = ROOT / "tesseract_nr_gpu.slang"
    environment = ROOT / ".venv-slang" / "Scripts" / "python.exe"
    powershell = powershell_executable()
    disk = shutil.disk_usage(ROOT)
    checks = {
        "python": {
            "ok": python.is_file(),
            "label": f"Python {sys.version_info.major}.{sys.version_info.minor}",
            "detail": str(python),
        },
        "vulkan_environment": {
            "ok": environment.is_file(),
            "label": "Vulkan environment",
            "detail": (
                "Ready"
                if environment.is_file()
                else "Run the environment setup first"
            ),
        },
        "slangpy": {
            "ok": importlib.util.find_spec("slangpy") is not None,
            "label": "SlangPy runtime",
            "detail": (
                "Available"
                if importlib.util.find_spec("slangpy") is not None
                else "Not installed in this Python environment"
            ),
        },
        "shader": {
            "ok": shader.is_file(),
            "label": "Tesseract shader",
            "detail": shader.name,
        },
        "powershell": {
            "ok": powershell is not None,
            "label": "Preset runner",
            "detail": powershell or "PowerShell not found",
        },
        "disk": {
            "ok": disk.free >= 2 * 1024**3,
            "label": "Free disk space",
            "detail": f"{disk.free / 1024**3:.1f} GiB available",
        },
    }
    required = (
        "python",
        "vulkan_environment",
        "slangpy",
        "shader",
        "powershell",
        "disk",
    )
    return {
        "ready": all(bool(checks[name]["ok"]) for name in required),
        "checks": checks,
        "platform": sys.platform,
        "project_root": str(ROOT),
    }


class StudioJobManager:
    """One-job supervisor for allow-listed Tesseract presets."""

    def __init__(
        self,
        presets: dict[str, StudioPreset],
        *,
        protocols: dict[str, LabProtocol] | None = None,
        demo: bool = False,
        run_root: Path = RUN_ROOT,
    ) -> None:
        self.presets = presets
        self.protocols = load_protocols() if protocols is None else protocols
        self.demo = bool(demo)
        self.run_root = run_root
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._worker: threading.Thread | None = None
        self._cancel_requested = False
        self._sequence = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=240)
        self._state: dict[str, Any] = {
            "schema": STATE_SCHEMA,
            "id": None,
            "preset_id": None,
            "protocol_id": None,
            "mode": None,
            "title": None,
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "elapsed_seconds": 0.0,
            "progress": 0.0,
            "progress_source": "none",
            "current_step": None,
            "total_steps": None,
            "simulation_time": None,
            "qualification": None,
            "return_code": None,
            "job_directory": None,
            "output_directory": None,
            "artifacts": [],
            "runner": None,
            "parameters": {},
            "command_preview": None,
            "visualizer": "tesseract",
            "accent": "#67e8f9",
            "eyebrow": "EXPERIMENT",
            "demo": self.demo,
            "message": "Choose an experiment to begin.",
        }

    def _emit(self, kind: str, message: str, **values: Any) -> None:
        self._sequence += 1
        event = {
            "sequence": self._sequence,
            "time": utc_now(),
            "kind": kind,
            "message": message,
        }
        event.update(values)
        self._events.append(event)

    def _persist(self) -> None:
        directory = self._state.get("job_directory")
        if directory:
            payload = dict(self._state)
            payload["events"] = list(self._events)
            atomic_json(Path(directory) / "job.json", payload)

    def _expanded_arguments(
        self, preset: StudioPreset, job_directory: Path, resume: bool
    ) -> list[str]:
        replacements = {
            "{job_dir}": str(job_directory),
            "{project_root}": str(ROOT),
        }
        arguments = []
        for raw in preset.arguments:
            value = raw
            for marker, replacement in replacements.items():
                value = value.replace(marker, replacement)
            if not SAFE_TOKEN_PATTERN.fullmatch(value):
                raise ValueError("expanded preset argument is unsafe")
            arguments.append(value)
        if resume:
            checkpoint = preset.checkpoint_path()
            if checkpoint is None or not checkpoint.is_file():
                raise FileNotFoundError(
                    "this preset has no restart checkpoint"
                )
            arguments.extend(["-RestartFrom", str(checkpoint)])
        return arguments

    @staticmethod
    def _infer_output_directory(
        arguments: list[str], job_directory: Path
    ) -> Path:
        for option in ("-OutputDirectory", "-Output"):
            if option in arguments:
                index = arguments.index(option)
                if index + 1 < len(arguments):
                    value = Path(arguments[index + 1])
                    target = value if value.is_absolute() else ROOT / value
                    return (
                        target
                        if option == "-OutputDirectory"
                        else target.parent
                    ).resolve()
        return job_directory

    def _command(
        self, preset: StudioPreset, job_directory: Path, resume: bool
    ) -> tuple[list[str], Path]:
        powershell = powershell_executable()
        if powershell is None:
            raise RuntimeError("PowerShell is required to run presets")
        script = RUNNERS[preset.runner].resolve()
        if not script.is_file() or script.parent != ROOT:
            raise FileNotFoundError(
                f"allow-listed runner is missing: {script.name}"
            )
        arguments = self._expanded_arguments(
            preset, job_directory, resume
        )
        command = [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *arguments,
        ]
        return command, self._infer_output_directory(
            arguments, job_directory
        )

    def start(self, preset_id: str, *, resume: bool = False) -> dict[str, Any]:
        preset = self.presets.get(preset_id)
        if preset is None:
            raise KeyError(f"unknown preset {preset_id!r}")
        return self._start_experiment(
            preset,
            resume=resume,
            mode="preset",
            protocol_id=None,
            parameters={},
        )

    def start_protocol(
        self,
        protocol_id: str,
        supplied: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        protocol = self.protocols.get(protocol_id)
        if protocol is None:
            raise KeyError(f"unknown protocol {protocol_id!r}")
        values = protocol.normalize(supplied)
        output = f"{{job_dir}}/{protocol.output_name}"
        preset = StudioPreset(
            id=f"lab-{protocol.id}",
            title=protocol.title,
            eyebrow="ADVANCED LAB RUN",
            summary=protocol.summary,
            what_you_will_see=protocol.qualification_note,
            runner=protocol.runner,
            arguments=protocol.arguments(values, output),
            duration_label="Custom laboratory run",
            estimated_seconds=protocol.estimate(values),
            minimum_vram_gib=protocol.minimum_vram_gib,
            disk_label="Managed job workspace",
            tier="laboratory",
            visualizer=protocol.visualizer,
            accent=protocol.accent,
            total_steps=protocol.total_steps(values),
            result_files=protocol.result_files(),
        )
        return self._start_experiment(
            preset,
            resume=False,
            mode="advanced",
            protocol_id=protocol.id,
            parameters=values,
        )

    def _start_experiment(
        self,
        preset: StudioPreset,
        *,
        resume: bool,
        mode: str,
        protocol_id: str | None,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            if self._state["status"] in {"starting", "running", "stopping"}:
                raise RuntimeError("another Tesseract experiment is running")
            identifier = (
                datetime.now().strftime("%Y%m%d-%H%M%S")
                + "-"
                + uuid4().hex[:7]
            )
            job_directory = self.run_root / identifier
            job_directory.mkdir(parents=True, exist_ok=False)
            command, output_directory = self._command(
                preset, job_directory, resume
            )
            self._events.clear()
            self._sequence = 0
            self._cancel_requested = False
            self._state = {
                "schema": STATE_SCHEMA,
                "id": identifier,
                "preset_id": preset.id,
                "protocol_id": protocol_id,
                "mode": mode,
                "title": preset.title,
                "status": "starting",
                "started_at": utc_now(),
                "finished_at": None,
                "elapsed_seconds": 0.0,
                "progress": 0.0,
                "progress_source": (
                    "step_or_estimated"
                    if preset.total_steps is not None
                    else "estimated"
                ),
                "current_step": 0 if preset.total_steps is not None else None,
                "total_steps": preset.total_steps,
                "simulation_time": None,
                "qualification": None,
                "return_code": None,
                "job_directory": str(job_directory),
                "output_directory": str(output_directory),
                "artifacts": [],
                "runner": RUNNERS[preset.runner].name,
                "parameters": parameters,
                "command_preview": self._display_command(command),
                "visualizer": preset.visualizer,
                "accent": preset.accent,
                "eyebrow": preset.eyebrow,
                "demo": self.demo,
                "resumed": bool(resume),
                "estimated_seconds": preset.estimated_seconds,
                "message": (
                    "Restoring the last checkpoint…"
                    if resume
                    else "Preparing the experiment…"
                ),
            }
            self._emit(
                "system",
                (
                    f"Resuming {preset.title}"
                    if resume
                    else f"Starting {preset.title}"
                ),
            )
            target = self._run_demo if self.demo else self._run_process
            self._worker = threading.Thread(
                target=target,
                args=(preset, command, job_directory),
                name=f"tesseract-studio-{identifier}",
                daemon=True,
            )
            self._worker.start()
            self._persist()
            return self.snapshot()

    @staticmethod
    def _display_command(command: list[str]) -> str:
        visible = [Path(command[6]).name, *command[7:]]
        return " ".join(
            (
                "'" + value.replace("'", "''") + "'"
                if re.search(r"\s", value)
                else value
            )
            for value in visible
        )

    def _update_line(self, preset: StudioPreset, line: str) -> None:
        message = line.strip()
        if not message:
            return
        step_match = STEP_PATTERN.search(message)
        time_match = TIME_PATTERN.search(message)
        if step_match and preset.total_steps:
            step = min(int(step_match.group(1)), preset.total_steps)
            self._state["current_step"] = step
            self._state["progress"] = step / preset.total_steps
            self._state["progress_source"] = "step"
        if time_match:
            try:
                self._state["simulation_time"] = float(time_match.group(1))
            except ValueError:
                pass
        qualification_match = re.search(
            r"\bqualified\s*=\s*(true|false)\b",
            message,
            flags=re.IGNORECASE,
        )
        if qualification_match:
            self._state["qualification"] = (
                qualification_match.group(1).lower() == "true"
            )
        if "qualified=" in message.lower():
            kind = "success" if "true" in message.lower() else "warning"
        elif message.lower().startswith(
            ("warning", "warn", "[warn", "error", "[error")
        ):
            kind = "warning"
        elif message.lower().startswith("wrote"):
            kind = "artifact"
        else:
            kind = "output"
        self._state["message"] = message
        self._emit(kind, message)

    def _creation_flags(self) -> int:
        if os.name != "nt":
            return 0
        return (
            subprocess.CREATE_NO_WINDOW
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )

    def _run_process(
        self,
        preset: StudioPreset,
        command: list[str],
        job_directory: Path,
    ) -> None:
        log_path = job_directory / "output.log"
        started = time.perf_counter()
        try:
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONUTF8"] = "1"
            with log_path.open("w", encoding="utf-8") as log:
                with self._lock:
                    if self._cancel_requested:
                        self._state["status"] = "cancelled"
                        self._state["finished_at"] = utc_now()
                        self._state["message"] = (
                            "Run stopped before the simulation launched."
                        )
                        self._emit("warning", self._state["message"])
                        self._persist()
                        return
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                        env=environment,
                        creationflags=self._creation_flags(),
                    )
                    self._process = process
                    self._state["status"] = "running"
                    self._state["message"] = "The simulation is running."
                    self._emit("system", "GPU job launched")
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    with self._lock:
                        self._update_line(preset, line)
                return_code = process.wait()
            with self._lock:
                cancelled = self._cancel_requested
                self._state["return_code"] = return_code
                self._state["finished_at"] = utc_now()
                self._state["elapsed_seconds"] = (
                    time.perf_counter() - started
                )
                if cancelled:
                    self._state["status"] = "cancelled"
                    self._state["message"] = (
                        "Run stopped. The last complete rolling checkpoint "
                        "is still available."
                    )
                    self._emit("warning", self._state["message"])
                elif (
                    return_code == 0
                    and self._state.get("qualification") is False
                ):
                    self._state["status"] = "failed"
                    self._state["progress"] = 1.0
                    self._state["message"] = (
                        "Run completed but failed one or more "
                        "scientific qualification gates."
                    )
                    self._emit("error", self._state["message"])
                elif return_code == 0:
                    self._state["status"] = "complete"
                    self._state["progress"] = 1.0
                    self._state["message"] = "Experiment complete."
                    self._emit("success", "Experiment complete")
                else:
                    self._state["status"] = "failed"
                    self._state["message"] = (
                        f"Experiment exited with code {return_code}."
                    )
                    self._emit("error", self._state["message"])
                self._state["artifacts"] = self._artifacts(
                    preset, job_directory
                )
                self._process = None
                self._persist()
        except Exception as exc:
            logging.exception("Studio job failed")
            with self._lock:
                self._state["status"] = "failed"
                self._state["return_code"] = -1
                self._state["finished_at"] = utc_now()
                self._state["elapsed_seconds"] = (
                    time.perf_counter() - started
                )
                self._state["message"] = str(exc)
                self._emit("error", str(exc))
                self._process = None
                self._persist()

    def _run_demo(
        self,
        preset: StudioPreset,
        command: list[str],
        job_directory: Path,
    ) -> None:
        del command
        started = time.perf_counter()
        with self._lock:
            self._state["status"] = "running"
            self._state["message"] = "Interactive demonstration running."
            self._emit("system", "Demo telemetry connected")
        samples = 48
        for index in range(samples + 1):
            with self._lock:
                if self._cancel_requested:
                    self._state["status"] = "cancelled"
                    self._state["finished_at"] = utc_now()
                    self._state["message"] = "Demonstration stopped."
                    self._emit("warning", "Demonstration stopped")
                    self._persist()
                    return
                progress = index / samples
                step = (
                    int(progress * preset.total_steps)
                    if preset.total_steps
                    else None
                )
                self._state["progress"] = progress
                self._state["current_step"] = step
                self._state["simulation_time"] = 0.5 * progress
                if preset.visualizer == "puncture" and progress > 0.78:
                    message = (
                        f"step={step or index} common_outer + common_inner "
                        "stability branches resolved"
                    )
                elif preset.visualizer == "phase" and progress > 0.48:
                    message = "Neural phase boundary crossed"
                else:
                    message = (
                        f"step={step or index} spacetime state healthy"
                    )
                self._state["message"] = message
                if index % 6 == 0:
                    self._emit("output", message)
            time.sleep(0.075)
        artifact = job_directory / "demo_result.json"
        atomic_json(
            artifact,
            {
                "schema": "vulkan-tesseract.studio-demo.v1",
                "preset": preset.id,
                "qualified": True,
                "note": "UI demonstration; no physics result claimed.",
            },
        )
        with self._lock:
            self._state["status"] = "complete"
            self._state["finished_at"] = utc_now()
            self._state["elapsed_seconds"] = time.perf_counter() - started
            self._state["progress"] = 1.0
            self._state["qualification"] = True
            self._state["message"] = "Demonstration complete."
            self._state["artifacts"] = [
                {
                    "name": artifact.name,
                    "path": str(artifact),
                    "size_bytes": artifact.stat().st_size,
                    "kind": "json",
                }
            ]
            self._emit("success", "Demonstration complete")
            self._persist()

    def _artifacts(
        self, preset: StudioPreset, job_directory: Path
    ) -> list[dict[str, Any]]:
        artifacts = []
        for raw in preset.result_files:
            value = raw.replace("{job_dir}", str(job_directory)).replace(
                "{project_root}", str(ROOT)
            )
            path = Path(value)
            if not path.is_absolute():
                if value.replace("\\", "/").startswith("runs/"):
                    path = ROOT / path
                else:
                    path = job_directory / path
            try:
                path = _safe_child(path, ROOT)
            except ValueError:
                continue
            if path.is_file():
                artifacts.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "kind": path.suffix.lower().lstrip(".") or "file",
                    }
                )
        return artifacts

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._state["status"] not in {"starting", "running"}:
                raise RuntimeError("there is no active experiment")
            self._cancel_requested = True
            self._state["status"] = "stopping"
            self._state["message"] = "Stopping the experiment…"
            self._emit("warning", "Stop requested")
            process = self._process
        if process is not None and process.poll() is None:
            if os.name == "nt":
                try:
                    subprocess.run(
                        [
                            "taskkill",
                            "/PID",
                            str(process.pid),
                            "/T",
                            "/F",
                        ],
                        check=False,
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        timeout=15,
                    )
                except subprocess.TimeoutExpired:
                    process.kill()
            else:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
        return self.snapshot()

    def _elapsed(self) -> float:
        started = self._state.get("started_at")
        if not started:
            return 0.0
        if self._state["status"] in {
            "complete",
            "failed",
            "cancelled",
        }:
            return float(self._state.get("elapsed_seconds") or 0.0)
        try:
            moment = datetime.fromisoformat(started)
            return max(
                0.0,
                (datetime.now(timezone.utc) - moment).total_seconds(),
            )
        except ValueError:
            return 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
            state["events"] = list(self._events)
            state["event_sequence"] = self._sequence
            elapsed = self._elapsed()
            state["elapsed_seconds"] = elapsed
            if state["status"] in {"starting", "running", "stopping"}:
                progress = float(state.get("progress") or 0.0)
                estimated = float(state.get("estimated_seconds") or 0.0)
                if (
                    state["progress_source"]
                    in {"estimated", "step_or_estimated"}
                    and estimated > 0
                ):
                    progress = min(
                        0.94, elapsed / estimated
                    )
                    state["progress"] = progress
                if progress > 0.005:
                    state["eta_seconds"] = max(
                        0.0, elapsed * (1.0 - progress) / progress
                    )
                elif estimated > 0:
                    state["eta_seconds"] = max(
                        0.0, estimated - elapsed
                    )
                else:
                    state["eta_seconds"] = None
            else:
                state["eta_seconds"] = 0.0
            return state

    def open_output(self) -> Path:
        with self._lock:
            raw = self._state.get("output_directory")
        if not raw:
            raise RuntimeError("no experiment output is available")
        directory = _safe_child(Path(raw), ROOT)
        directory.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(directory)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(directory)])
        else:
            subprocess.Popen(["xdg-open", str(directory)])
        return directory

    def result_catalog(self) -> list[dict[str, Any]]:
        candidates: dict[Path, str] = {}
        if self.run_root.is_dir():
            for job_file in self.run_root.glob("*/job.json"):
                try:
                    job = json.loads(job_file.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                for artifact in job.get("artifacts", ()):
                    if not isinstance(artifact, dict):
                        continue
                    raw_path = artifact.get("path")
                    if not isinstance(raw_path, str):
                        continue
                    path = Path(raw_path).resolve()
                    try:
                        _safe_child(path, self.run_root)
                    except ValueError:
                        try:
                            _safe_child(path, ROOT)
                        except ValueError:
                            continue
                    if path.suffix.lower() == ".json" and path.is_file():
                        candidates[path] = "studio_run"
        for pattern in (
            "vulkan_*_results.json",
            "tesseract_*_results.json",
            "theory33_*_results.json",
        ):
            for path in ROOT.glob(pattern):
                candidates.setdefault(path.resolve(), "qualified_reference")
        rows: list[dict[str, Any]] = []
        for path, source in candidates.items():
            try:
                size = path.stat().st_size
                if size > 16 * 1024 * 1024:
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            relative = self._result_label(path)
            identifier = hashlib.sha256(
                str(path).encode("utf-8")
            ).hexdigest()[:20]
            qualified = payload.get("qualified")
            if qualified is None:
                qualified = payload.get("passed")
            rows.append(
                {
                    "id": identifier,
                    "name": path.stem.replace("_", " "),
                    "file": path.name,
                    "relative_path": relative,
                    "source": source,
                    "schema": payload.get("schema")
                    or payload.get("format")
                    or "unversioned-json",
                    "qualified": (
                        qualified if isinstance(qualified, bool) else None
                    ),
                    "modified": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "size_bytes": size,
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                row["source"] == "studio_run",
                str(row["modified"]),
            ),
            reverse=True,
        )

    def _result_label(self, path: Path) -> str:
        for base in (self.run_root.resolve(), ROOT.resolve()):
            try:
                return path.relative_to(base).as_posix()
            except ValueError:
                continue
        return path.name

    def _result_path(self, identifier: str) -> tuple[Path, str]:
        for row in self.result_catalog():
            if row["id"] != identifier:
                continue
            relative = str(row["relative_path"])
            if row["source"] == "studio_run":
                run_path = self.run_root / relative
                root_path = ROOT / relative
                if run_path.is_file():
                    path = _safe_child(run_path, self.run_root)
                else:
                    path = _safe_child(root_path, ROOT)
            else:
                path = _safe_child(ROOT / relative, ROOT)
            return path, str(row["source"])
        raise KeyError(f"unknown result {identifier!r}")

    def analyze_result(self, identifier: str) -> dict[str, Any]:
        path, source = self._result_path(identifier)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("result is too large for the interactive lab")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("result root must be a JSON object")
        qualifications = _qualification_rows(payload)
        metrics = _metric_rows(payload)
        series = _series_candidates(payload)
        qualified = payload.get("qualified")
        if qualified is None:
            qualified = payload.get("passed")
        passed = sum(1 for row in qualifications if row["passed"])
        provenance_keys = (
            "schema",
            "format",
            "backend",
            "precision",
            "problem",
            "reconstruction",
            "constitutive_model",
            "physical_block",
            "path",
            "variant_digest",
            "reference_schema",
        )
        provenance = {
            key: payload[key]
            for key in provenance_keys
            if key in payload
            and isinstance(payload[key], (str, int, float, bool))
        }
        provenance.update(
            {
                "source": source,
                "file": self._result_label(path),
                "size_bytes": path.stat().st_size,
                "modified": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        return {
            "id": identifier,
            "title": path.stem.replace("_", " "),
            "qualified": qualified if isinstance(qualified, bool) else None,
            "qualification": {
                "passed": passed,
                "total": len(qualifications),
                "rows": qualifications,
            },
            "metrics": metrics,
            "series": series,
            "provenance": provenance,
            "raw": payload,
        }


def _qualification_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, bool):
            rows.append(
                {
                    "path": ".".join(path),
                    "label": path[-1].replace("_", " "),
                    "passed": value,
                }
            )
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, str(key)))

    for key, value in payload.items():
        if key == "qualification" or key.endswith("_qualification"):
            visit(value, (key,))
    if not rows:
        for key in ("qualified", "passed"):
            if isinstance(payload.get(key), bool):
                visit(payload[key], (key,))
    rows.sort(key=lambda row: (not row["passed"], row["path"]))
    return rows[:300]


def _metric_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skip = {"qualification", "implementation_sha256", "raw"}

    def visit(value: Any, path: tuple[str, ...], depth: int) -> None:
        if depth > 5 or len(rows) >= 240:
            return
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            if float("-inf") < float(value) < float("inf"):
                rows.append(
                    {
                        "path": ".".join(path),
                        "label": path[-1].replace("_", " "),
                        "value": value,
                    }
                )
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if key not in skip and not key.endswith("sha256"):
                    visit(child, (*path, str(key)), depth + 1)
            return
        if isinstance(value, list) and len(value) <= 12:
            for index, child in enumerate(value):
                if isinstance(child, dict):
                    visit(child, (*path, str(index)), depth + 1)

    visit(payload, (), 0)
    priority_words = (
        "error",
        "residual",
        "margin",
        "order",
        "speed",
        "balance",
        "steps",
        "time",
        "resolution",
        "points",
    )
    rows.sort(
        key=lambda row: (
            not any(word in row["path"].lower() for word in priority_words),
            len(row["path"].split(".")),
            row["path"],
        )
    )
    return rows[:120]


def _series_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def visit(value: Any, path: tuple[str, ...], depth: int) -> None:
        if depth > 5 or len(candidates) >= 16:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, str(key)), depth + 1)
        elif (
            isinstance(value, list)
            and 2 <= len(value) <= 256
            and all(isinstance(item, dict) for item in value)
        ):
            common = set(value[0])
            for item in value[1:]:
                common &= set(item)
            numeric = [
                key
                for key in common
                if all(
                    isinstance(item.get(key), (int, float))
                    and not isinstance(item.get(key), bool)
                    for item in value
                )
            ]
            if len(numeric) >= 2:
                axis = next(
                    (
                        key
                        for key in (
                            "points",
                            "resolution",
                            "steps",
                            "samples",
                            "time",
                            "degree",
                        )
                        if key in numeric
                    ),
                    numeric[0],
                )
                fields = [key for key in numeric if key != axis][:8]
                candidates.append(
                    {
                        "path": ".".join(path),
                        "label": path[-1].replace("_", " ")
                        if path
                        else "series",
                        "axis": axis,
                        "fields": fields,
                        "rows": [
                            {
                                key: item[key]
                                for key in (axis, *fields)
                            }
                            for item in value
                        ],
                    }
                )

    visit(payload, (), 0)
    candidates.sort(
        key=lambda item: (
            item["axis"] not in {"points", "resolution"},
            -len(item["rows"]),
        )
    )
    return candidates


def execute_lab_command(
    manager: StudioJobManager, command: str
) -> dict[str, Any]:
    if "\n" in command or "\r" in command or len(command) > 1200:
        raise ValueError("laboratory command must be one short line")
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError("could not parse laboratory command") from exc
    if not tokens:
        return {"lines": [], "action": None}
    verb = tokens[0].lower()
    if verb in {"help", "?"}:
        return {
            "action": "help",
            "lines": [
                {
                    "kind": "system",
                    "text": (
                        "run <protocol> key=value... | status | protocols | "
                        "results | stop | open-results | clear"
                    ),
                },
                {
                    "kind": "system",
                    "text": (
                        "This console accepts Tesseract laboratory commands; "
                        "it is intentionally not an operating-system shell."
                    ),
                },
            ],
        }
    if verb == "protocols":
        return {
            "action": "protocols",
            "lines": [
                {
                    "kind": "output",
                    "text": (
                        f"{protocol.id}: {protocol.title}"
                    ),
                }
                for protocol in manager.protocols.values()
            ],
        }
    if verb == "status":
        state = manager.snapshot()
        return {
            "action": "status",
            "job": state,
            "lines": [
                {
                    "kind": "output",
                    "text": (
                        f"{state['status']} | "
                        f"{state.get('title') or 'no active run'} | "
                        f"{float(state.get('progress') or 0) * 100:.1f}%"
                    ),
                }
            ],
        }
    if verb == "results":
        catalog = manager.result_catalog()
        return {
            "action": "results",
            "lines": [
                {
                    "kind": "output",
                    "text": f"{len(catalog)} JSON evidence files indexed",
                }
            ],
        }
    if verb == "stop":
        state = manager.cancel()
        return {
            "action": "stop",
            "job": state,
            "lines": [{"kind": "warning", "text": "Stop requested"}],
        }
    if verb == "open-results":
        directory = manager.open_output()
        return {
            "action": "open_results",
            "lines": [
                {"kind": "artifact", "text": f"Opened {directory}"}
            ],
        }
    if verb == "clear":
        return {"action": "clear", "lines": []}
    if verb != "run":
        raise ValueError(
            f"unknown laboratory command {tokens[0]!r}; enter help"
        )
    if len(tokens) < 2:
        raise ValueError("run requires a protocol id")
    protocol_id = tokens[1]
    supplied: dict[str, Any] = {}
    for token in tokens[2:]:
        if "=" not in token:
            raise ValueError(
                f"expected key=value parameter, received {token!r}"
            )
        key, value = token.split("=", 1)
        if not key or not value:
            raise ValueError("laboratory parameters use key=value")
        supplied[key] = value
    state = manager.start_protocol(protocol_id, supplied)
    return {
        "action": "run",
        "job": state,
        "lines": [
            {
                "kind": "system",
                "text": f"Dispatched {state['command_preview']}",
            }
        ],
    }


class StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        manager: StudioJobManager,
    ) -> None:
        super().__init__(address, handler)
        self.manager = manager


class StudioHandler(BaseHTTPRequestHandler):
    server: StudioHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), format % args)

    def _json(
        self, payload: Any, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _error(
        self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST
    ) -> None:
        self._json({"ok": False, "error": message}, status)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        content_type = (
            mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Cache-Control",
            "no-cache" if path.suffix == ".html" else "public, max-age=300",
        )
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/api/presets":
            presets = [
                value.public()
                for value in self.server.manager.presets.values()
            ]
            self._json({"ok": True, "presets": presets})
            return
        if path == "/api/protocols":
            protocols = [
                value.public()
                for value in self.server.manager.protocols.values()
            ]
            self._json({"ok": True, "protocols": protocols})
            return
        if path == "/api/results":
            self._json(
                {
                    "ok": True,
                    "results": self.server.manager.result_catalog(),
                }
            )
            return
        if path.startswith("/api/results/"):
            identifier = path.removeprefix("/api/results/")
            if not re.fullmatch(r"[a-f0-9]{20}", identifier):
                self._error("invalid result id", HTTPStatus.NOT_FOUND)
                return
            try:
                analysis = self.server.manager.analyze_result(identifier)
            except KeyError as exc:
                self._error(str(exc), HTTPStatus.NOT_FOUND)
                return
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self._error(str(exc), HTTPStatus.UNPROCESSABLE_ENTITY)
                return
            self._json({"ok": True, "analysis": analysis})
            return
        if path == "/api/status":
            self._json(
                {
                    "ok": True,
                    "job": self.server.manager.snapshot(),
                    "preflight": preflight(),
                }
            )
            return
        if path == "/api/preflight":
            self._json({"ok": True, **preflight()})
            return
        if path == "/media/dashboard.png":
            self._file(ROOT / "tesseract_full_stack_dashboard.png")
            return
        if path in {"/", "/index.html"}:
            self._file(STATIC_ROOT / "index.html")
            return
        if path.startswith("/static/"):
            relative = path.removeprefix("/static/")
            try:
                target = _safe_child(STATIC_ROOT / relative, STATIC_ROOT)
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._file(target)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _payload(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid request length") from exc
        if length < 0 or length > 32_768:
            raise ValueError("request body is too large")
        if length == 0:
            return {}
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        return value

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._payload()
            if path == "/api/jobs/start":
                preset_id = payload.get("preset_id")
                if not isinstance(preset_id, str):
                    raise ValueError("preset_id is required")
                state = self.server.manager.start(
                    preset_id, resume=bool(payload.get("resume", False))
                )
                self._json(
                    {"ok": True, "job": state}, HTTPStatus.ACCEPTED
                )
                return
            if path == "/api/jobs/start-advanced":
                protocol_id = payload.get("protocol_id")
                parameters = payload.get("parameters", {})
                if not isinstance(protocol_id, str):
                    raise ValueError("protocol_id is required")
                if not isinstance(parameters, dict):
                    raise ValueError("parameters must be an object")
                state = self.server.manager.start_protocol(
                    protocol_id, parameters
                )
                self._json(
                    {"ok": True, "job": state}, HTTPStatus.ACCEPTED
                )
                return
            if path == "/api/console":
                command = payload.get("command")
                if not isinstance(command, str):
                    raise ValueError("command is required")
                result = execute_lab_command(
                    self.server.manager, command
                )
                self._json({"ok": True, **result})
                return
            if path == "/api/jobs/cancel":
                self._json(
                    {"ok": True, "job": self.server.manager.cancel()}
                )
                return
            if path == "/api/jobs/open-output":
                directory = self.server.manager.open_output()
                self._json({"ok": True, "path": str(directory)})
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._error(str(exc), HTTPStatus.NOT_FOUND)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            self._error(str(exc), HTTPStatus.CONFLICT)


def configure_logging() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(
                RUN_ROOT / "studio.log", encoding="utf-8"
            ),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="animate safe fake telemetry without launching physics runners",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="allow binding outside localhost (not recommended)",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    if args.host not in local_hosts and not args.allow_network:
        parser.error("non-local binding requires --allow-network")

    configure_logging()
    presets = load_presets()
    manager = StudioJobManager(presets, demo=args.demo)
    server = StudioHTTPServer(
        (args.host, args.port), StudioHandler, manager
    )
    url = f"http://{args.host}:{args.port}/"
    logging.info("Tesseract Studio ready at %s", url)
    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        logging.info("Tesseract Studio stopping")
        try:
            manager.cancel()
        except RuntimeError:
            pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
