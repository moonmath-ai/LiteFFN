#!/usr/bin/env python
"""AMD (ROCm) LiteLinear benchmark on FFN projection shapes.

Same shape set as `bench_litelinear.py` but only compares `nn.Linear` against
the ROCm LiteLinear path (no CUDA, no TE). Use on an AMD ROCm box with the
`lite_linear-0.3.0+rocm63` wheel installed.

Units:
  per-shape rows: us
  TOTAL row: ms (count-weighted aggregate)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from lite_linear import LiteLinear


try:
    import lite_linear._rocm as _rocm_ext

    _HAS_ROCM = True
    _ROCM_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - example code
    _rocm_ext = None
    _HAS_ROCM = False
    _ROCM_IMPORT_ERROR = exc


def _shape_name(n: int, k: int) -> str:
    if n == 16384 and k == 4096:
        return "w1"
    if n == 4096 and k == 16384:
        return "w2"
    return f"N{n}_K{k}"


def _load_configs_from_json(path: Path) -> list[tuple[str, int, int, int, int]]:
    data = json.loads(path.read_text())
    out: list[tuple[str, int, int, int, int]] = []
    for entry in data:
        m = int(entry["M"])
        n = int(entry["N"])
        k = int(entry["K"])
        calls = int(entry.get("count", 1))
        out.append((_shape_name(n, k), m, k, n, calls))
    return out


def _bench_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _fmt_us(ms: float, width: int = 7) -> str:
    return f"{ms * 1000:>{width}.0f}"


def _fmt_ms(ms: float, width: int = 7) -> str:
    return f"{ms:>{width}.0f}"


def _fmt_x(val: float, width: int = 7) -> str:
    return f"{val:>{width}.3f}x"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes-json", type=Path, default=None)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm is required for this benchmark.")
    if not _HAS_ROCM:
        print(
            f"[bench] lite_linear._rocm extension not loaded ({_ROCM_IMPORT_ERROR}); "
            "make sure the `lite_linear-0.3.0+rocm63` wheel is installed."
        )
    if not getattr(torch.version, "hip", None):
        print(
            "[bench] PyTorch does not report a ROCm build (`torch.version.hip` is None). "
            "LiteLinear will still construct via the platform-detected FP8 dtype, but the "
            "ROCm extension will not load."
        )

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    configs = _load_configs_from_json(args.shapes_json) if args.shapes_json else [
        ("w1", 1400, 4096, 16384, 195),
        ("w1", 5600, 4096, 16384, 84),
        ("w2", 1400, 16384, 4096, 196),
        ("w2", 5600, 16384, 4096, 84),
    ]
    if args.limit > 0:
        configs = configs[: args.limit]

    print("AMD LiteLinear Benchmark on captured FFN shapes")
    print("Units:")
    print("  per-shape rows: us")
    print("  TOTAL row: ms")
    print("Column glossary:")
    print("  Cfg: FFN projection shape (w1 = up-proj, w2 = down-proj).")
    print("  M: flattened activation rows for that GEMM shape.")
    print("  Count: number of calls for that shape in the captured workload.")
    print("  Lin: baseline nn.Linear latency.")
    print("  ROCm: LiteLinear ROCm path latency (skipped if extension not loaded).")
    print("  ROCm_x: speedup multiplier vs baseline (>1 is faster).")
    print("  TOTAL: count-weighted aggregate across listed shapes.")
    print()
    print(f"{'Cfg':<4} {'M':>6} {'Count':>6} | {'Lin':>6} {'ROCM':>6} | {'ROCM_x':>7}")
    print("-" * 50)

    total_calls = 0
    total_lin_ms = 0.0
    total_rocm_ms = 0.0
    rows: list[dict] = []

    for cfg, m, k, n, count in configs:
        total_calls += int(count)
        x = torch.randn(m, k, device=device, dtype=dtype)

        lin_mod = nn.Linear(k, n, bias=True, device=device, dtype=dtype)

        rocm_mod = LiteLinear(
            k, n, bias=True, device=device, dtype=dtype, rank=args.rank
        )
        with torch.no_grad():
            rocm_mod.A.copy_(torch.randn(n, args.rank, device=device, dtype=dtype) * 0.1)
            rocm_mod.B.copy_(torch.randn(args.rank, k, device=device, dtype=dtype) * 0.1)
            rocm_mod.Q_fp8.copy_(
                (torch.randn(n, k, device=device) * 0.1).to(rocm_mod.Q_fp8.dtype)
            )
            rocm_mod.Q_scale_inv.copy_(torch.tensor(1.0, device=device, dtype=torch.float32))
            if rocm_mod.bias is not None:
                rocm_mod.bias.copy_(lin_mod.bias)
        rocm_mod.eval()

        t_lin = _bench_ms(lambda: lin_mod(x), args.warmup, args.iters)
        t_rocm = _bench_ms(lambda: rocm_mod(x), args.warmup, args.iters)
        rocm_x = t_lin / t_rocm

        total_lin_ms += t_lin * count
        total_rocm_ms += t_rocm * count

        print(
            f"{cfg:<4} {m:>6} {count:>6} | {_fmt_us(t_lin):>6} {_fmt_us(t_rocm):>6} | "
            f"{_fmt_x(rocm_x):>7}"
        )

        rows.append(
            {
                "cfg": cfg,
                "m": int(m),
                "k": int(k),
                "n": int(n),
                "count": int(count),
                "lin_us": t_lin * 1000.0,
                "rocm_us": t_rocm * 1000.0,
                "rocm_x": rocm_x,
            }
        )

    total_rocm_x = total_lin_ms / total_rocm_ms
    print("-" * 50)
    print(
        f"{'TOTAL':<4} {'':>6} {total_calls:>6} | "
        f"{_fmt_ms(total_lin_ms):>6} {_fmt_ms(total_rocm_ms):>6} | {_fmt_x(total_rocm_x):>7}"
    )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "rank": args.rank,
                    "rows": rows,
                    "total": {
                        "calls": total_calls,
                        "lin_ms": total_lin_ms,
                        "rocm_ms": total_rocm_ms,
                        "rocm_x": total_rocm_x,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
