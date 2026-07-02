#!/usr/bin/env python
"""Compatibility wrapper for the renamed ROCm LiteLinear benchmark."""

if __package__:
    from .bench_litelinear_amd import main
else:
    from bench_litelinear_amd import main


if __name__ == "__main__":
    raise SystemExit(main())
