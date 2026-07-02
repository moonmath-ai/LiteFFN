#!/usr/bin/env python
"""Module-level benchmark: `LiteLinear` vs `nn.Linear` (and `transformer_engine` if available).

The kernel microbench (`bench_ffn.py`) calls `lite_linear._cuda.fused_forward`
directly. This script benches the `LiteLinear` *module* end-to-end, exercising
the Python forward path (`_q_scale_inv_cpu` cache, dtype guards, etc.) — use it
to confirm the module-level overhead is small and to compare against
`nn.Linear` (or `transformer_engine.pytorch.Linear` when installed).

Factor source (`--factor-source`):

  random      Random A/B/Q with bench-friendly magnitudes (fast setup).
  decompose   Build a random `nn.Linear` and run `LiteLinear.from_dense`
              (real SVD; representative of production factor distributions
              but slow at large shapes).
  load        Load a pre-decomposed `.safetensors` (output of
              `lite-linear convert`) and pick a specific FQN's factors.

Outputs: per-shape table + TOTAL row weighted by `count` from
`--shapes-json` (or unit weights for the default config list), plus
optional JSON via `--output`.

Requires a CUDA-capable GPU. The kernel extension is required for the
"CUDA" column — without it the script warns and skips that column.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from lite_linear import LiteLinear


try:
    import lite_linear._cuda as _cuda_ext

    _HAS_CUDA = True
    _CUDA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - example code
    _cuda_ext = None
    _HAS_CUDA = False
    _CUDA_IMPORT_ERROR = exc

try:
    import transformer_engine.pytorch as te

    _HAS_TE = True
    _TE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    te = None
    _HAS_TE = False
    _TE_IMPORT_ERROR = exc


DEFAULT_CONFIGS = [
    # (name, M, K, N, call_count) — real LTX 13B FFN shapes, 24-frame inference.
    ("w1", 1400, 4096, 16384, 195),
    ("w1", 5600, 4096, 16384, 84),
    ("w2", 1400, 16384, 4096, 196),
    ("w2", 5600, 16384, 4096, 84),
]


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


def _bench_ms(fn, warmup: int, iters: int) -> tuple[float, float] | None:
    """Return (wall_ms, cuda_event_ms) per call, averaged over `iters`."""

    def _run() -> tuple[float, float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wall0 = time.perf_counter()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - wall0) * 1000.0 / iters
        return wall, start.elapsed_time(end) / iters

    return _run()


def _ms_max(t: tuple[float, float] | None) -> float | None:
    if t is None:
        return None
    return max(t[0], t[1])


def _fmt_us(t: tuple[float, float] | None, width: int = 8) -> str:
    if t is None:
        return f"{'n/a':>{width}}"
    return f"{_ms_max(t) * 1000.0:>{width - 2}.0f}us"


def _fmt_ms(t: tuple[float, float] | None, width: int = 8) -> str:
    if t is None:
        return f"{'n/a':>{width}}"
    return f"{_ms_max(t):>{width - 2}.0f}ms"


def _pct(baseline, candidate) -> str:
    b = _ms_max(baseline)
    c = _ms_max(candidate)
    if b is None or c is None or b <= 0:
        return f"{'n/a':>8}"
    return f"{((b - c) / b) * 100.0:>+7.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes-json", type=Path, default=None)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--factor-source",
        choices=("random", "decompose"),
        default="random",
        help="How to build the LiteLinear factors. `random` is fast; `decompose` "
        "builds a random `nn.Linear` and calls `LiteLinear.from_dense` (real SVD).",
    )
    ap.add_argument("--include-te", action="store_true", help="Bench TE Linear too.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if not _HAS_CUDA:
        print(
            f"[bench] lite_linear._cuda extension not loaded ({_CUDA_IMPORT_ERROR}); "
            "the CUDA column will be skipped."
        )
    if args.include_te and not _HAS_TE:
        print(f"[bench] transformer_engine not available ({_TE_IMPORT_ERROR}); TE column = n/a")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    configs = (
        _load_configs_from_json(args.shapes_json)
        if args.shapes_json
        else DEFAULT_CONFIGS
    )
    if args.limit > 0:
        configs = configs[: args.limit]

    print(f"LiteLinear Benchmark | rank={args.rank} | factor_source={args.factor_source}")
    print(
        f"{'Cfg':<4} {'M':>5} {'Count':>5} | {'Linear':>9} {'TE':>9} {'LL':>9} "
        f"{'LL-CUDA':>9} | {'TE vs Lin':>9} {'LL vs Lin':>10} {'CUDA vs Lin':>12}"
    )
    print("-" * 110)

    total_lin: tuple[float, float] | None = None
    total_te: tuple[float, float] | None = None
    total_ll: tuple[float, float] | None = None
    total_cuda: tuple[float, float] | None = None
    total_calls = 0

    rows: list[dict] = []
    for cfg, m, k, n, calls in configs:
        total_calls += calls
        x = torch.randn(m, k, device=device, dtype=dtype)

        # Baseline nn.Linear
        linear_mod = nn.Linear(k, n, bias=True, device=device, dtype=dtype)
        with torch.no_grad():
            linear_mod.weight.normal_(mean=0.0, std=0.05)
            linear_mod.bias.zero_()

        # Optional TE Linear
        te_mod = None
        if args.include_te and _HAS_TE:
            try:
                try:
                    te_mod = te.Linear(
                        k, n, bias=True, device=device, params_dtype=dtype
                    )
                except TypeError:
                    te_mod = te.Linear(k, n, bias=True).to(device=device, dtype=dtype)
                with torch.no_grad():
                    te_mod.weight.copy_(linear_mod.weight)
                    if te_mod.bias is not None:
                        te_mod.bias.copy_(linear_mod.bias)
                te_mod.eval()
            except Exception as exc:
                print(f"[bench] TE init failed for {cfg} M={m}: {exc}")
                te_mod = None

        # LiteLinear: build via from_dense or random factors.
        if args.factor_source == "decompose":
            ll_mod = LiteLinear.from_dense(linear_mod, rank=args.rank).to(device=device)
        else:
            A = torch.randn(n, args.rank, device=device, dtype=dtype) * 0.1
            B = torch.randn(args.rank, k, device=device, dtype=dtype) * 0.1
            Q = (torch.randn(n, k, device=device) * 0.1).to(torch.float8_e4m3fn)
            scale = torch.tensor(1.0, device=device, dtype=torch.float32)
            ll_mod = LiteLinear(
                k, n, bias=True, device=device, dtype=dtype, rank=args.rank
            )
            with torch.no_grad():
                ll_mod.A.copy_(A)
                ll_mod.B.copy_(B)
                ll_mod.Q_fp8.copy_(Q)
                ll_mod.Q_scale_inv.copy_(scale)
                ll_mod.bias.copy_(linear_mod.bias)
        ll_mod.eval()

        t_lin = _bench_ms(lambda: linear_mod(x), args.warmup, args.iters)
        t_te = None
        if te_mod is not None:
            t_te = _bench_ms(lambda: te_mod(x), args.warmup, args.iters)
        t_ll = _bench_ms(lambda: ll_mod(x), args.warmup, args.iters)
        t_cuda = None
        if _HAS_CUDA:

            def _cuda_call():
                return _cuda_ext.fused_forward(
                    x, ll_mod.Q_fp8, ll_mod.A, ll_mod.B, ll_mod.bias, 1.0
                )

            t_cuda = _bench_ms(_cuda_call, args.warmup, args.iters)

        for t, total in (
            (t_lin, "lin"),
            (t_te, "te"),
            (t_ll, "ll"),
            (t_cuda, "cuda"),
        ):
            if t is None:
                continue
            ms = _ms_max(t)
            if total == "lin":
                total_lin = (
                    (total_lin[0] + ms * calls, total_lin[1] + ms * calls)
                    if total_lin is not None
                    else (ms * calls, ms * calls)
                )
            elif total == "te":
                total_te = (
                    (total_te[0] + ms * calls, total_te[1] + ms * calls)
                    if total_te is not None
                    else (ms * calls, ms * calls)
                )
            elif total == "ll":
                total_ll = (
                    (total_ll[0] + ms * calls, total_ll[1] + ms * calls)
                    if total_ll is not None
                    else (ms * calls, ms * calls)
                )
            elif total == "cuda":
                total_cuda = (
                    (total_cuda[0] + ms * calls, total_cuda[1] + ms * calls)
                    if total_cuda is not None
                    else (ms * calls, ms * calls)
                )

        print(
            f"{cfg:<4} {m:>5} {calls:>5} | "
            f"{_fmt_us(t_lin, 9)} {_fmt_us(t_te, 9)} {_fmt_us(t_ll, 9)} {_fmt_us(t_cuda, 9)} | "
            f"{_pct(t_lin, t_te):>9} {_pct(t_lin, t_ll):>10} "
            f"{_pct(t_lin, t_cuda):>12}"
        )
        rows.append(
            {
                "cfg": cfg,
                "m": m,
                "k": k,
                "n": n,
                "count": calls,
                "t_lin_ms": _ms_max(t_lin),
                "t_te_ms": _ms_max(t_te),
                "t_ll_ms": _ms_max(t_ll),
                "t_cuda_ms": _ms_max(t_cuda),
            }
        )

    print("-" * 110)
    print(
        f"{'TOTAL':<4} {'':>5} {total_calls:>5} | "
        f"{_fmt_ms(total_lin, 9)} {_fmt_ms(total_te, 9)} {_fmt_ms(total_ll, 9)} {_fmt_ms(total_cuda, 9)} | "
        f"{_pct(total_lin, total_te):>9} "
        f"{_pct(total_lin, total_ll):>10} "
        f"{_pct(total_lin, total_cuda):>12}"
    )
    print("=" * 110)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "rank": args.rank,
                    "factor_source": args.factor_source,
                    "configs": rows,
                    "totals_ms": {
                        "t_lin": _ms_max(total_lin),
                        "t_te": _ms_max(total_te),
                        "t_ll": _ms_max(total_ll),
                        "t_cuda": _ms_max(total_cuda),
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
