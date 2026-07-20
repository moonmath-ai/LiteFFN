# LiteLinear Wheel Compatibility

This page documents the public wheel contract for the current LiteLinear
release artifacts in this repository.

## Current Public Release

Current release: `v0.3.0+cu128+rocm72`

| Wheel | Python ABI | Platform | Runtime target | Built against |
| --- | --- | --- | --- | --- |
| `lite_linear-0.3.0+cu128-cp310-cp310-linux_x86_64.whl` | CPython 3.10 | Linux x86_64 | NVIDIA CUDA 12.8 | torch 2.11 cu128 |
| `lite_linear-0.3.0+cu128-cp312-cp312-linux_x86_64.whl` | CPython 3.12 | Linux x86_64 | NVIDIA CUDA 12.8 | torch 2.11 cu128 |
| `lite_linear-0.3.0+rocm72-cp310-cp310-linux_x86_64.whl` | CPython 3.10 | Linux x86_64 | AMD ROCm 7.2 | torch 2.11.0+rocm7.2 |
| `lite_linear-0.3.0+rocm72-cp312-cp312-linux_x86_64.whl` | CPython 3.12 | Linux x86_64 | AMD ROCm 7.2 | torch 2.11.0+rocm7.2 |

Use these wheels from either:

- the GitHub release assets for `v0.3.0+cu128+rocm72`
- the repository-local `install/` directory

Do not assume a PyPI distribution exists for this release unless a package is
published there.

## Not Published In This Release

The current public release does not include:

- a CPython 3.11 wheel
- macOS or Windows wheels
- CUDA 12.9 or CUDA 13.0 wheels
- public source-build files such as `pyproject.toml`, `setup.py`, or native
  kernel sources

If one of those environments is required, treat it as unsupported by the
published `v0.3.0+cu128+rocm72` artifacts until a matching wheel or source-build path
is released.

## Metadata Versus Wheel Tags

The wheel metadata can include broad package requirements such as
`Requires-Python: >=3.8` and `Requires-Dist: torch >=2.8.0`.
Those metadata fields do not replace the wheel tag contract.

For installation, the Python ABI tag, platform tag, and runtime target are the
first compatibility boundary:

```text
lite_linear-0.3.0+cu128-cp312-cp312-linux_x86_64.whl
                         ^^^^^ ^^^^^ ^^^^^^^^^^^^
                         Python ABI  Linux platform
```

The local version label, for example `+cu128`, identifies the runtime flavor of
the published artifact. It is informational for Python package resolution, so
users should choose a wheel from the compatibility table rather than relying on
extras such as `cu129`, `cu130`, or `rocm63` to select a different binary.

## Install Check

Before installing, verify the local Python version and torch build:

```bash
python -c "import sys; print(sys.version)"
python -c "import torch; print(torch.__version__, 'cuda=', torch.version.cuda, 'hip=', torch.version.hip)"
```

Then install the matching wheel explicitly:

```bash
python -m pip install --force-reinstall --no-deps \
  install/lite_linear-0.3.0+cu128-cp312-cp312-linux_x86_64.whl
```

Use the CUDA or ROCm wheel that matches the installed PyTorch backend, and use
the `cp310` wheel for Python 3.10 or the `cp312` wheel for Python 3.12.
