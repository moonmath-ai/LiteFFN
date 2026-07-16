"""End-to-end Wan2.1 integration example for the 0.3.0 LiteLinear API.

The post-refactor API collapses the previous in-process patch-and-cache flow
into a single offline CLI step. The host model needs to do two things and
nothing more:

1. Construct the target FFN layers with `LiteLinear`.
2. Point the host loader at a converted `.safetensors` produced by
   `lite-linear convert`.

No online patching, no patch-config filtering, no model-side env vars.
The new shape is documented in `docs/integration_guide.md`; this file is
the smallest example that runs end-to-end against a real Wan2.1 shard.

Usage (against a Diffusers-style sharded Wan2.1 checkpoint):

.. code-block:: bash

    # 1. Convert one shard offline (do this once per checkpoint).
    python -m lite_linear convert-sharded \\
        /path/to/Wan2.1-I2V-14B-720P/diffusion_pytorch_model.safetensors.index.json \\
        --regex 'ffn\\.(0|2)' \\
        --rank 64 \\
        -o /path/to/Wan2.1-I2V-14B-720P-litelinear-r64-e4m3fn/

    # 2. Load the converted snapshot with the host framework as usual.
    python examples/wan_integration.py \\
        --checkpoint /path/to/Wan2.1-I2V-14B-720P-litelinear-r64-e4m3fn/

The example model below is intentionally tiny — it stands in for the
Wan FFN layout (`linear -> GELU -> linear`) without pulling in the
upstream model code. It exists to document the integration surface;
swap in your actual model definition and checkpoint path for real use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torch import nn

from lite_linear import LiteLinear


class WanLikeBlock(nn.Module):
    """Stand-in for a Wan FFN block: `linear -> GELU -> linear`.

    Wan2.1 uses explicit `nn.Sequential(linear, GELU, linear)` FFNs, so the
    two FFN linears (`ffn.0`, `ffn.2`) can be replaced with `LiteLinear`
    directly. LTX-style FFNs that wrap the first linear in a GEGLU need
    the upstream `LiteLinear.replace_activation_proj_()` helper instead —
    see `docs/integration_guide.md`.
    """

    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        # `bias=True` matches Wan's default; LiteLinear requires GPU weights.
        # PyTorch uses `.to("cuda")` for both NVIDIA CUDA and AMD ROCm builds.
        self.ffn = nn.Sequential(
            LiteLinear(dim, ffn_dim, bias=True),
            nn.GELU(approximate="tanh"),
            LiteLinear(ffn_dim, dim, bias=True),
        )

    def forward(self, x):
        return self.ffn(x)


class WanLikeTransformer(nn.Module):
    """Minimal host model that mirrors the Wan FFN replacement pattern."""

    def __init__(self, dim: int, ffn_dim: int, num_layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [WanLikeBlock(dim, ffn_dim) for _ in range(num_layers)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def build_model(*, dim: int = 1536, ffn_dim: int = 8960, num_layers: int = 2) -> nn.Module:
    """Construct a small Wan-shaped transformer with LiteLinear FFNs.

    Defaults match Wan2.1-I2V-14B head dimensions; `num_layers=2` keeps the
    example fast to instantiate for a smoke check.
    """
    return WanLikeTransformer(dim=dim, ffn_dim=ffn_dim, num_layers=num_layers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-check the LiteLinear integration shape on a Wan-like model. "
            "Run `lite-linear convert` (or `convert-sharded`) on a real "
            "checkpoint first, then point this script at the converted snapshot."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a `lite-linear convert`-produced checkpoint directory "
        "(or single `.safetensors` file). If omitted, the script just "
        "constructs the model and runs a random GPU forward.",
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--dim", type=int, default=1536)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("LiteLinear requires a GPU device.")

    model = build_model(dim=args.dim).to("cuda", dtype=torch.bfloat16)
    model.eval()

    if args.checkpoint is not None:
        # Standard Diffusers / Wan sharded loader: pass the converted dir.
        # The LiteLinear module-level state_dict contains A/B/Q_fp8/Q_scale_inv/
        # bias directly, so the host loader consumes the converted shards
        # with no special hooks.
        from safetensors.torch import load_model

        load_model(model, str(args.checkpoint))
        print(f"Loaded factors from {args.checkpoint}")
    else:
        print(
            "No --checkpoint given; running a random-weight forward to "
            "verify the construction works end-to-end."
        )

    with torch.no_grad():
        x = torch.randn(
            args.batch, args.seq, args.dim, device="cuda", dtype=torch.bfloat16
        )
        y = model(x)
    print(
        f"forward ok: input {tuple(x.shape)} -> output {tuple(y.shape)}, "
        f"dtype={y.dtype}"
    )


if __name__ == "__main__":
    main()
