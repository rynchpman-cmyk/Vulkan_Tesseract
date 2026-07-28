# Tesseract Studio

Tesseract Studio is Vulkan Tesseract's local, beginner-facing control room.
It presents qualified simulations as understandable presets instead of
requiring users to compose physics commands.

The interface is deliberately a local application:

- the browser supplies immediate visual feedback and familiar controls;
- the Python controller runs only on loopback by default;
- the existing PowerShell wrappers remain the execution authority; and
- the Vulkan device, checkpoints, and result files remain on the workstation.

No cloud service, account, telemetry, or JavaScript package installation is
required.

## First launch

The easiest path on a fresh checkout is to double-click
`Setup Tesseract Studio.cmd` once. It creates the private environment,
installs the pinned runtime, and verifies the important imports.

The equivalent manual setup is:

```powershell
python -m venv .venv-slang
.\.venv-slang\Scripts\python.exe -m pip install -r requirements-slang.txt
```

Then double-click `Launch Tesseract Studio.cmd`, or run:

```powershell
.\run_tesseract_studio.ps1
```

The launcher starts a hidden local process and opens
`http://127.0.0.1:8765`. Re-running the launcher focuses the same local
instance instead of starting a competing job server.

For development, keep the server visible:

```powershell
.\run_tesseract_studio.ps1 -Foreground
```

To exercise the complete interface without dispatching Vulkan work:

```powershell
.\run_tesseract_studio.ps1 -Demo
```

## Presets

| Preset | Purpose | Typical scale |
|---|---|---:|
| Full-Stack Visual Tour | Render the neural-spacetime overview dashboard | seconds |
| Neural Phase Collision | Run the learned two-phase Carter Riemann ladder | minutes |
| Spacetime Smoke Test | Exercise the coupled multidimensional GPU stack | minutes |
| Puncture Mission Planner | Validate the production black-hole contract without evolving it | seconds |
| Twin Black-Hole Collision | Run the qualified `N=120`, length-3, 500-step puncture/MOTS campaign | about one day |

Time estimates are guidance, not deadlines. Shader compilation, GPU model,
thermals, drivers, and output cadence affect wall-clock time. The dashboard
switches to measured progress and ETA whenever a runner exposes step counts.

## What the controls mean

- **Launch** starts the selected preset.
- **Stop run** terminates only the managed child process. If a rolling
  checkpoint already exists, it remains the preset's resume target.
- **Resume** appears when the preset's checkpoint exists and restarts the same
  qualified runner from that file.
- **Open results** opens the job's output folder in Explorer.
- **Advanced Mode** opens the complete Tesseract Laboratory without
  cluttering the default experience.

Only one simulation can run at a time. This avoids accidental GPU
oversubscription and ambiguous result ownership.

## Tesseract Laboratory

Advanced Mode is an end-to-end research workbench, not merely a result
viewer. It has three connected surfaces.

### Run control

The experiment designer loads typed protocols from `studio/protocols`.
Parameters have explicit types, ranges, choices, defaults, and scientific
help. The control plane validates them again on the server, constructs an
isolated job workspace, previews the exact repository runner command, and
launches one managed process.

The initial protocols cover:

- learned neural-phase Carter Riemann ladders;
- coupled multidimensional Carter/CCZ4 evolution;
- puncture CCZ4 and MOTS campaigns, including plan-only checks; and
- full-stack or literal visualization.

Custom values are experiments, not automatically new qualification claims.
Each protocol displays the boundary of the tracked reference configuration.

### Live terminal

The laboratory console shows the exact runner, raw stdout, warnings,
qualification messages, artifacts, elapsed time, estimated or step-derived
progress, and final status. It also accepts a small Tesseract command grammar:

```text
help
protocols
status
results
run spacetime nx=8 ny=8 steps=2
run neural-phase resolutions=16,32,64,128 final_time=0.02 cfl=0.12
stop
open-results
clear
```

This is intentionally not PowerShell. Unknown verbs, parameters, executable
paths, newlines, and out-of-range values are rejected before process launch.
This keeps an in-browser laboratory terminal from becoming an arbitrary
operating-system shell.

### Evidence lab

Completed laboratory JSON artifacts and tracked repository qualification
records are indexed automatically. Selecting an artifact produces:

- an overall qualification seal and complete gate matrix;
- prioritized numerical metrics;
- detected resolution or sample ladders with interactive metric selection;
- a data table corresponding to every plotted series;
- backend, precision, constitutive, reconstruction, reference, digest, file,
  timestamp, size, and SHA-256 provenance when present; and
- a copyable, unchanged raw JSON view.

The analyzer is schema-aware but not tied to a single result schema. It walks
qualification dictionaries, numerical structures, and list-of-record ladders
so new Tesseract evidence receives a useful dashboard without rewriting the
front end.

## Safety and trust boundary

Preset and protocol files are declarative JSON, but they are not arbitrary
command templates. `tesseract_studio.py` maps each `runner` value to a fixed,
repository-owned PowerShell wrapper. Unknown runners, placeholders, preset
IDs, protocol parameters, and filesystem escapes are rejected.

Advanced job state exposes a sanitized command preview so researchers can
audit exactly what was dispatched. The API never accepts executable or output
paths from the browser. The server binds to `127.0.0.1` unless a developer
explicitly passes `--allow-network`.

Stopping is forceful on Windows so that the runner and all of its child
processes exit together. Avoid pressing Stop while the event stream explicitly
reports a checkpoint write; an operating-system interruption during any file
write can leave that file incomplete.

## Adding a preset

Add a JSON file beneath `studio/presets` using schema
`vulkan-tesseract.studio-preset.v1`. A preset supplies:

- a unique `id`, readable title, summary, and expected visual outcome;
- one allow-listed `runner`;
- fixed wrapper arguments;
- duration, disk, VRAM, tier, and visualizer metadata;
- optional step count, output directory, result files, and resume checkpoint.

Do not expose free-form shell text or user-controlled filesystem commands.
Add or update a `tests/test_studio.py` contract whenever a preset changes the
production run definition.

For a customizable Advanced Mode surface, add a JSON file beneath
`studio/protocols` using schema `vulkan-tesseract.lab-protocol.v1`. Every
parameter must declare a supported type and explicit numerical limits or
choices. Output locations remain controller-generated and must not be exposed
as protocol parameters.

## Validation

The controller, static application, presets, protocols, console grammar,
result analyzer, HTTP API, and demo lifecycle are covered without requiring
Vulkan:

```powershell
python -m pytest tests\test_studio.py -q
```

The real puncture planner provides the shortest end-to-end wrapper check:

```powershell
.\run_vulkan_puncture_mots.ps1 -PlanOnly
```

Live GPU presets retain the same qualification artifacts and acceptance gates
as their PowerShell runners. The Studio interface does not weaken or replace
the numerical evidence.
