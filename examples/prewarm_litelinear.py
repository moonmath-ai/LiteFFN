#!/usr/bin/env python3
"""Prewarm LiteLinear fused-forward autotune for known FFN shapes.

The script uses synthetic tensors with production-like GEMM shapes. Tensor
values are arbitrary; the shape, dtype, rank, backend, and GPU determine which
cuBLASLt / hipBLASLt plans are selected and cached.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifests" / "ltx2_5s_1536x1024.json"


def load_shape_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"shape manifest must be a JSON list: {path}")
    for idx, row in enumerate(data):
        for key in ("M", "K", "N"):
            if key not in row:
                raise ValueError(f"shape row {idx} is missing {key!r}: {path}")
    return data


def gpu_autotune_cache_tag(torch_module, device: int | str = 0) -> str:
    props = torch_module.cuda.get_device_properties(device)
    if getattr(torch_module.version, "hip", None):
        arch = getattr(props, "gcnArchName", None) or "unknown"
        return arch.split(":")[0]
    major, minor = torch_module.cuda.get_device_capability(device)
    return f"sm{major}{minor}"


def default_autotune_cache_path(torch_module, device: int | str = 0) -> Path:
    explicit = os.environ.get("LITELINEAR_AUTOTUNE_CACHE_FILE")
    if explicit:
        return Path(explicit)
    tag = gpu_autotune_cache_tag(torch_module, device)
    leaf = (
        f"autotune_rocm_{tag}.cache"
        if getattr(torch_module.version, "hip", None)
        else f"autotune_cuda_{tag}.cache"
    )
    return Path.home() / ".cache" / "lite-linear" / leaf


def platform_fp8_dtype(torch_module):
    if getattr(torch_module.version, "hip", None) and hasattr(torch_module, "float8_e4m3fnuz"):
        return torch_module.float8_e4m3fnuz
    return torch_module.float8_e4m3fn


def import_fused_extension(torch_module):
    if getattr(torch_module.version, "hip", None):
        import lite_linear._rocm as ext

        return ext, "rocm"

    import lite_linear._cuda as ext

    return ext, "cuda"


def make_fused_tensors(torch_module, shape: dict[str, Any], *, dtype):
    m = int(shape["M"])
    k = int(shape["K"])
    n = int(shape["N"])
    rank = int(shape.get("rank", 64))
    seed = int(shape.get("seed", 30_000 + m + k + n + rank))
    fp8_dtype = platform_fp8_dtype(torch_module)

    torch_module.manual_seed(seed)
    x = torch_module.randn(m, k, device="cuda", dtype=dtype)
    q_weight = (torch_module.randn(n, k, device="cuda") * 0.1).to(fp8_dtype)
    a_weight = torch_module.randn(n, rank, device="cuda", dtype=dtype) * 0.1
    b_weight = torch_module.randn(rank, k, device="cuda", dtype=dtype) * 0.1
    bias = torch_module.randn(n, device="cuda", dtype=dtype) * 0.01
    return x, q_weight, a_weight, b_weight, bias


def prewarm_shape(torch_module, ext, shape: dict[str, Any], *, passes: int) -> dict[str, Any]:
    tensors = make_fused_tensors(torch_module, shape, dtype=torch_module.bfloat16)
    pass_wall_ms: list[float] = []
    finite_fraction = 0.0

    for _ in range(passes):
        torch_module.cuda.synchronize()
        started = time.perf_counter()
        output = ext.fused_forward(*tensors, 1.0)
        torch_module.cuda.synchronize()
        pass_wall_ms.append((time.perf_counter() - started) * 1000.0)
        finite_fraction = float(torch_module.isfinite(output).float().mean().item())

    return {
        **shape,
        "passes": passes,
        "pass_wall_ms": pass_wall_ms,
        "finite_fraction": finite_fraction,
    }


def prewarm_manifest(
    shapes: list[dict[str, Any]],
    *,
    cache_file: Path | None,
    passes: int,
    strict: bool,
) -> dict[str, Any]:
    if cache_file is not None:
        os.environ["LITELINEAR_AUTOTUNE_CACHE_FILE"] = str(cache_file)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP GPU is required for LiteLinear prewarm")

    ext, backend = import_fused_extension(torch)
    rows = [prewarm_shape(torch, ext, shape, passes=passes) for shape in shapes]

    if strict:
        bad = [row.get("name", "<unnamed>") for row in rows if row["finite_fraction"] < 1.0]
        if bad:
            raise RuntimeError(f"prewarm produced non-finite outputs for: {bad}")

    return {
        "schema": "litelinear_prewarm_v1",
        "backend": backend,
        "cache_file": str(default_autotune_cache_path(torch)),
        "passes": passes,
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_hip": getattr(torch.version, "hip", None),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_cache_tag": gpu_autotune_cache_tag(torch),
        },
        "rows": rows,
        "total_wall_ms": sum(sum(row["pass_wall_ms"]) for row in rows),
    }


def dry_run_report(shapes: list[dict[str, Any]], *, cache_file: Path | None, passes: int) -> dict[str, Any]:
    return {
        "schema": "litelinear_prewarm_dry_run_v1",
        "cache_file": str(cache_file) if cache_file is not None else None,
        "passes": passes,
        "shape_count": len(shapes),
        "shapes": shapes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-file", type=Path, default=None)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without importing torch or running GPU work.")
    parser.add_argument("--no-strict", action="store_true", help="Do not fail when a prewarm output is non-finite.")
    args = parser.parse_args()

    shapes = load_shape_manifest(args.shapes_json)
    report = (
        dry_run_report(shapes, cache_file=args.cache_file, passes=args.passes)
        if args.dry_run
        else prewarm_manifest(
            shapes,
            cache_file=args.cache_file,
            passes=args.passes,
            strict=not args.no_strict,
        )
    )

    text = json.dumps(report, indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
