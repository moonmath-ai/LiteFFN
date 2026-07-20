#!/usr/bin/env python
"""Compatibility wrapper for the renamed LiteLinear benchmark."""

if __package__:
    from .bench_litelinear import main
else:
    from bench_litelinear import main


if __name__ == "__main__":
    raise SystemExit(main())
