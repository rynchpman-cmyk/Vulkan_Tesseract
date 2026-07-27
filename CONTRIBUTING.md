# Contributing

Contributions are welcome when they improve Vulkan execution, physical
fidelity, numerical stability, performance, or independent validation.

## Development principles

1. **Keep hard decisions explicit.** Do not silently smooth constitutive
   classifiers, active sets, or phase transitions in the production path.
2. **Preserve Carter structure.** Learned constitutive behavior should descend
   from one scalar master response unless an experiment is labeled otherwise.
3. **Keep the CPU oracle independent.** Do not make CPU qualification call the
   shader implementation it is intended to audit.
4. **Report physical margins.** Recovery validity alone is insufficient;
   switching, Legendre, thermodynamic, characteristic, and constraint margins
   belong in evidence.
5. **Preserve device residency.** Evolution kernels must not introduce hidden
   `to_numpy` or host synchronization between explicit snapshots.
6. **Separate qualification from claims.** Experimental kernels should not be
   described as production paths until their tracked gates pass.

## Setup

CPU oracle and repository tests:

```powershell
python -m pip install "numpy>=2.3,<3" pytest
```

Shader runtime:

```powershell
python -m venv .venv-slang
.\.venv-slang\Scripts\python.exe -m pip install -r requirements-slang.txt
```

## Before submitting a change

Run the CPU-safe suite:

```powershell
python -m compileall -q .
python -m pytest tests\nr tests\test_artifacts.py -q
```

For resident Carter or CCZ4 changes, run:

```powershell
.\.venv-slang\Scripts\python.exe .\test_vulkan_full_carter_ccz4.py
.\.venv-slang\Scripts\python.exe .\test_vulkan_multidimensional.py
```

Run any earlier live regression whose kernel or ABI was modified. When a
tracked campaign changes, regenerate its `vulkan_*_results.json` file and
update the matching `VULKAN_*_REPORT.md`.

## Pull requests

Describe:

- the mathematical or implementation change;
- affected resident state fields and kernel entry points;
- which hard signatures can change;
- CPU/GPU error, conservation, recovery, and constraint effects;
- device-residency impact;
- validation commands and measured results; and
- generated Slang, frozen-weight, or evidence changes.

Do not commit virtual environments, downloaded compiler tools, SPIR-V
binaries, binary PyTorch checkpoints, or local campaign logs.
