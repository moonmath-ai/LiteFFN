# LiteLinear Integration Guide

This document describes how to integrate LiteLinear into a host transformer
(Wan, LTX-Video, LTX-2, Hunyuan-Video, HF LLMs) using the 0.3.0 surface.

The end-to-end shape is:

1. **Architecture integration** — at model construction time, replace the
   FFN linear(s) you want to accelerate with `LiteLinear`.
2. **Checkpoint integration** — before the host model loads weights, run
   `lite-linear convert` on the dense `.safetensors` (single shard or HF
   sharded directory). The host loader then loads the converted snapshot
   the same way it would load any other checkpoint.

No in-process patching, no patch-config filtering, no model-side env vars.
The `LiteLinear` state_dict is just four factor parameters (`A`, `B`,
`Q_fp8`, `Q_scale_inv`) plus optional `bias`; the host framework's regular
loader picks them up.

## Mental model

### LiteLinear at a glance

```python
from lite_linear import LiteLinear

class Block(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        # Replace the up-proj + down-proj pair:
        self.ffn = nn.Sequential(
            LiteLinear(dim, ffn_dim, bias=True),
            nn.GELU(approximate="tanh"),
            LiteLinear(ffn_dim, dim, bias=True),
        )
```

State dict layout (key prefixes the host loader sees):

| Key | dtype | shape |
| --- | --- | --- |
| `<prefix>.A` | bfloat16 | `(out_features, rank)` |
| `<prefix>.B` | bfloat16 | `(rank, in_features)` |
| `<prefix>.Q_fp8` | float8_e4m3fn (NVIDIA) / float8_e4m3fnuz (AMD ROCm) | `(out_features, in_features)` |
| `<prefix>.Q_scale_inv` | float32 | `(1,)` (FSDP-friendly shape) |
| `<prefix>.bias` | bfloat16 | `(out_features,)` (only when `bias=True`) |

`load_state_dict` is the only on-ramp. No lazy materialization, no cache
files, no env-var-driven online patching.

### Offline conversion

The CLI is the production path. It rewrites a `.safetensors` file (or
HF-sharded directory) in place, replacing each selected `<prefix>.weight`
with the four factor keys:

```bash
# Single shard:
python -m lite_linear convert model.safetensors \
    --regex 'ffn\.(0|2)' --rank 64

# HF sharded (Diffusers-style index):
python -m lite_linear convert-sharded \
    path/to/diffusion_pytorch_model.safetensors.index.json \
    --regex 'ffn\.(0|2)' --rank 64

# Per-prefix ranks via a manifest:
python -m lite_linear convert model.safetensors --manifest ranks.toml
```

Default output names embed the rank + FP8 variant + calibration tag, so
conversions for different settings don't collide on disk:

```
model.safetensors                              -> model-litelinear-r64-e4m3fn.safetensors
model.safetensors + --r-matrices foo.safetensors -> model-litelinear-r64-e4m3fn-calib.safetensors
diffusion_pytorch_model.safetensors.index.json -> <parent>-litelinear-r64-e4m3fn/
```

Selection is uniform (`--regex + --rank`) or per-prefix (`--manifest`).
The CLI rejects rank > min(d_out, d_in) at plan time before any I/O.

### R-matrix calibration (optional, data-aware decomposition)

For tighter low-rank quality, capture an `R = XᵀX / N` matrix per layer
from real prompts with the forward-hook `Calibrator` and pass it to
`convert --r-matrices`:

```python
import torch
from lite_linear.calibration import Calibrator

with Calibrator(model) as cal:
    for batch in prompt_iter:
        model(batch.to("cuda").bfloat16())

cal.save("r_matrices.safetensors")
```

```bash
python -m lite_linear convert model.safetensors \
    --regex 'ffn\.(0|2)' --rank 64 \
    --r-matrices r_matrices.safetensors
```

Calibration is decoupled from the model's forward signature (forward-pre
hooks on matched modules), so the same loop works for any host model that
exposes a `nn.Linear` / `LiteLinear` per layer.

## Wan integration

Wan2.1 uses explicit `nn.Sequential(linear, GELU, linear)` FFNs, so the
two FFN linears can be replaced directly:

| Wan key | Replaced with |
| --- | --- |
| `ffn.0` | `LiteLinear(dim, ffn_dim, bias=True)` |
| `ffn.2` | `LiteLinear(ffn_dim, dim, bias=True)` |

No `replace_activation_proj_` is needed (that's only for LTX-style GEGLU).

### End-to-end Wan recipe

```bash
# 1. Convert the sharded checkpoint (one-shot).
python -m lite_linear convert-sharded \
    /path/to/Wan2.1-I2V-14B-720P/diffusion_pytorch_model.safetensors.index.json \
    --regex 'ffn\.(0|2)' \
    --rank 64 \
    -o /path/to/Wan2.1-I2V-14B-720P-litelinear-r64-e4m3fn/

# 2. Construct the model with LiteLinear FFNs and load the converted snapshot.
#    See examples/wan_integration.py for the smallest runnable version.
```

## LTX-style integration (LTX-Video, LTX-2)

LTX FFNs wrap the first linear inside an activation module (GEGLU /
ApproximateGELU), so replacing it directly requires swapping the
activation's internal `proj` attribute. The `LiteLinear` module ships a
helper for this:

```python
from lite_linear import LiteLinear

class FeedForward(nn.Module):
    def __init__(self, dim, inner_dim, bias=True):
        super().__init__()
        # Replace the activation wrapper's projection with LiteLinear.
        self.net = nn.ModuleList([
            LiteLinear(dim, inner_dim, bias=bias),  # was: act_fn (e.g. GEGLU(proj=nn.Linear))
            nn.Identity(),
            LiteLinear(inner_dim, dim, bias=bias),
        ])
```

(See the upstream LTX-Video integration for the full `FeedForward`
shape; the point is just that the first `nn.Linear` becomes a
`LiteLinear` of the same `(dim, inner_dim)`.)

Once the model is constructed with `LiteLinear` linears, the rest of the
integration is identical to Wan: convert the checkpoint with
`lite-linear convert`, load the converted snapshot.

### LTX-2 LoRA pipeline (two-step case)

Some LTX-2 pipelines pass a distilled LoRA at runtime (e.g.
`ltx-2.3-22b-distilled-lora-384.safetensors`) on top of the base
checkpoint. The LoRA deltas are dense `lora_B @ lora_A` updates applied
to the model's `<prefix>.weight` tensors at inference time.

Plain conversion breaks this contract for the converted FFNs:

```bash
# Plain conversion: selected FFs lose their <prefix>.weight tensor,
# so the runtime LoRA delta has nothing to update on those layers.
python -m lite_linear convert \
    /path/to/ltx-2.3-22b-dev.safetensors \
    --regex 'transformer_blocks\..*(audio_)?ff\.' \
    --rank 64 \
    -o /path/to/ltx-2.3-22b-dev-litelinear.safetensors
```

After this, `transformer_blocks.{i}.ff…` weights are replaced with the
four factor keys (`A` / `B` / `Q_fp8` / `Q_scale_inv`), so a runtime
LoRA delta of the form `strength * lora_B @ lora_A` against the original
prefix no longer has a `<prefix>.weight` to update — those FF layers
silently revert to the no-LoRA behavior.

For LTX-2 pipelines that pass a distilled LoRA at runtime, fold the LoRA
into the selected dense FF weights *before* decomposition. `lite-linear
convert` does this for you when you pass `--lora` and `--lora-strength`:

```bash
python -m lite_linear convert \
    /path/to/ltx-2.3-22b-dev.safetensors \
    --regex 'transformer_blocks\..*(audio_)?ff\.' \
    --lora /path/to/ltx-2.3-22b-distilled-lora-384.safetensors \
    --lora-strength 0.8 \
    -o /path/to/ltx-2.3-22b-dev-distilled-lora0.8-litelinear.safetensors
```

The converter first applies `W + strength * lora_B @ lora_A` per selected
prefix, then decomposes the fused weight. The output filename embeds the
strength tag (`…-lora0.8-…`) so conversions with different LoRA strengths
don't collide on disk.

You can still pass the same distilled LoRA at runtime. Non-selected
dense layers (attention, embeddings, anything the `--regex` did not
match) still have their `<prefix>.weight` intact, so the runtime LoRA
delta applies to them normally. The selected FF layers have the LoRA
delta already baked into `Q_fp8`; passing the LoRA at runtime has no
effect on those layers.

Use plain conversion (no `--lora`) for LTX-2 pipelines that do *not* pass
a distilled LoRA at runtime — not every LTX pipeline has that extra
step. Keep the LoRA match tight (`transformer_blocks\..*(audio_)?ff\.`)
so the converter folds the LoRA into exactly the layers the runtime
expects; the audio branch is intentional — LTX-2 audio FFs live under
`audio_ff.…` and should be kept in LiteLinear too.

## What is generic vs what is still model-specific

Generic across Wan / LTX-style / HF LLMs:

- `lite_linear.LiteLinear` — the `nn.Linear` replacement.
- `lite-linear convert` / `convert-sharded` — offline checkpoint rewriting.
- `lite-linear inspect` — list 2D `.weight` keys (helps hand-craft regexes).
- `lite_linear.calibration.Calibrator` — forward-hook R-matrix capture.
- `lite_linear.decompose.decompose_weight` — the math primitive
  (`A, B, Q_fp8, Q_scale_inv = decompose_weight(W, rank=r, R=R_optional)`).
- `LiteLinear.from_dense(linear, rank=...)` — in-Python decomposition for
  research / ad-hoc use.

Still model-specific (your code, not ours):

- Which FFN modules to replace with `LiteLinear`.
- How checkpoint keys map to FFN prefixes (the `--regex`).
- Where to point the host loader for the converted snapshot.

## Notes on per-platform FP8 variants

`Q_fp8` is platform-pinned: NVIDIA builds use `float8_e4m3fn`
(max 448), AMD ROCm builds use `float8_e4m3fnuz` (max 240). The
`load_state_dict` pre-hook on `LiteLinear` raises on a cross-platform
mismatch with a message pointing at `lite-linear convert --fp8-dtype
{e4m3fn,e4m3fnuz}`. The `--fp8-dtype` flag on `convert` defaults to
auto-detect from the local GPU.

## Migrating from the pre-0.3 surface

The previous public surface (0.1.x / 0.2.x wheels) used:

- `lite_linear.LowRankDeltaLinear` (renamed to `LiteLinear`)
- `lite_linear.checkpoint_patch_core.{resolve_or_build_patched_checkpoint,
  resolve_strict_checkpoint_path}` (no longer needed; the offline CLI
  replaces this)
- `python -m lite_linear.offline_patch` (renamed to
  `python -m lite_linear convert`)
- `materialize_from_weight()` (no longer exists; `LiteLinear.from_dense`
  is the in-Python sibling)
- env vars `LITELINEAR_DISABLED`, `LITELINEAR_ONLINE_PATCH`,
  `LITELINEAR_CACHE`, `LITELINEAR_PATCH_TAG`, `LITELINEAR_PATCH_CONFIG`
  (removed; per-module disable is now source-level — use `nn.Linear`
  directly instead of `LiteLinear`)
- kernel-tuning env vars `CUBLASLT_AUTOTUNE_*` (renamed to
  `LITELINEAR_AUTOTUNE_*`; the import-time check raises with the
  direct swap).

If you're starting from a pre-0.3 checkpoint that still has
`<prefix>.weight` keys (not the factor quad), run it through
`lite-linear convert` to produce a 0.3-compatible snapshot.

