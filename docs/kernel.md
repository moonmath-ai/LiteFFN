# Runtime Notes

Use the packaged wheels that match the target Python ABI and PyTorch backend.

## Compatibility

- Wheels are platform-specific and must match the target Python ABI.
- Use a CUDA- or ROCm-enabled PyTorch environment whose backend matches the
  wheel build (`+cu128` for CUDA 12.8, `+rocm72` for ROCm 7.2; see the
  wheel's `lite_linear-*.dist-info/METADATA` for package requirements).
- Published wheels for 0.3.0 are Linux x86_64 `cp310` and `cp312` only:
  CUDA 12.8 (`+cu128`) and the official AMD target ROCm 7.2 (`+rocm72`).
- ROCm 6.3 and 7.0 are not official support targets for this release; use the
  `+rocm72` wheels with a matching PyTorch 2.11.0+rocm7.2 environment.
- Rebuild wheels when changing Python, platform, CUDA/PyTorch
  compatibility, or deployment hardware assumptions.

See `docs/wheel_compatibility.md` for the current public wheel matrix and
install compatibility boundaries.

### Wheel filename convention

Wheel filenames follow the standard format
([PEP 491](https://peps.python.org/pep-0491/#file-name-convention)):

```
{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
```

LiteLinear does not use the optional build tag, so:

```
lite_linear-{version}+{flavor}-cp{py}-cp{py}-{platform}.whl
```

Where:

| Component | Meaning |
| --- | --- |
| `{distribution}` | `lite_linear` (import name `lite_linear`, distribution name `lite-linear`) |
| `{version}` | PEP 440 version (e.g. `0.3.0`, `0.2.0`) |
| `{flavor}` | local version label after `+` (PEP 440): `cu128` for NVIDIA, `rocm72` for AMD |
| `cp{py}-cp{py}` | Python tag / ABI tag — both identical for CPython ABI-tagged builds (`cp310`, `cp312`, …) |
| `{platform}` | platform tag (`linux_x86_64` for the wheels shipped here) |

For example:

- `lite_linear-0.3.0+cu128-cp310-cp310-linux_x86_64.whl` — 0.3.0 release,
  built for CUDA 12.8 torch wheels, Python 3.10.
- `lite_linear-0.3.0+cu128-cp312-cp312-linux_x86_64.whl` — 0.3.0 release,
  built for CUDA 12.8 torch wheels, Python 3.12.
- `lite_linear-0.3.0+rocm72-cp310-cp310-linux_x86_64.whl` — 0.3.0 release,
  built for PyTorch 2.11.0+rocm7.2, Python 3.10.

The PEP 440 local label (`+cu128`, `+rocm72`) identifies the backend build.
When installing from release assets or file paths, choose the wheel whose
local label matches the PyTorch backend in the target environment.
The wheel does not install PyTorch for you, and the local label is not a
separate PyPI distribution.

## Cross-platform FP8 variants

`Q_fp8` is platform-pinned: NVIDIA builds use `float8_e4m3fn` (max value
448), AMD ROCm builds use `float8_e4m3fnuz` (max value 240). PyTorch's
default `copy_` would silently cast between the two variants; the
`LiteLinear._check_fp8_dtype` `load_state_dict` pre-hook raises on a
mismatch and points at `lite-linear convert --fp8-dtype {e4m3fn,e4m3fnuz}`
to produce a checkpoint with the correct variant.

## Autotune Cache And Prewarm

The accelerated path may autotune the first call for each new `(M, N, K)`
shape. For services, run representative shapes once during startup so the
first user request does not pay that setup cost.

Autotune picks are stored in a local cache file. The default path is tagged by
backend and GPU architecture:

- CUDA: `~/.cache/lite-linear/autotune_cuda_<smXY>.cache`
- ROCm: `~/.cache/lite-linear/autotune_rocm_<gfx>.cache`

Set `LITELINEAR_AUTOTUNE_CACHE_FILE` to choose an explicit cache path. If you
want cache paths to include PyTorch version, LiteLinear version, driver, or
shape-set identity, encode that in this explicit path; the default filename
only includes backend and GPU architecture.

Example startup pattern using the repo helper:

```bash
python examples/prewarm_litelinear.py \
    --shapes-json examples/manifests/ltx2_5s_1536x1024.json \
    --cache-file /var/cache/lite-linear/ltx2-cu128-sm90-v0.3.0.cache \
    --passes 1 \
    --output-json /tmp/litelinear_prewarm_report.json
```

Then start the service with the same cache path:

```bash
export LITELINEAR_AUTOTUNE_CACHE_FILE=/var/cache/lite-linear/ltx2-cu128-sm90-v0.3.0.cache
python serve.py
```

Use `--dry-run` to validate the manifest and command line without importing
PyTorch or running GPU work.

For benchmarks, use the same idea with the expected tensor shapes: set the
cache path, run warmup forwards for each `(M, N, K)` shape you care about, then
collect timings in a separate pass.

Do not treat autotune caches as portable release artifacts. Generate or refresh
them in the target runtime environment, especially after PyTorch, driver,
backend, or hardware changes. To force a refresh, run once with:

```bash
export LITELINEAR_AUTOTUNE_RESET=1
```

New autotune picks are still persisted while reset is enabled. Unseen runtime
shapes can still pay first-touch autotune, so prewarm the shapes your service
actually serves.

## Benchmarking

For stable timing comparisons:

- Warm up the target workload before collecting timings (the kernel's
  first call at each shape runs cuBLASLt heuristic selection).
- Compare against the same model, prompt, scheduler settings, precision,
  and hardware.
- Treat first-run setup costs separately from steady-state inference
  timings.

### Startup prewarm for serving

Long-lived services can move LiteLinear's first-touch setup out of the first
user request for known FFN shapes. Set a persistent
`LITELINEAR_AUTOTUNE_CACHE_FILE`, run the production shape set during startup,
and mark the service ready only after prewarm succeeds.

```bash
python examples/prewarm_litelinear.py \
  --shapes-json shapes.json \
  --cache-file litelinear_autotune.cache \
  --output-json prewarm_result.json
```

For production, keep the shape manifest and cache key tied to the runtime:
GPU, CUDA or ROCm backend, torch, LiteLinear wheel version, dtype, rank, and
shape set. A
request that hits a shape missing from the manifest can still pay the
first-touch cost on the request path.

This is a startup-latency control path. It does not change rank, weights,
decomposition, inputs, or the LiteLinear fused-forward call path, and it does
not claim a faster steady-state kernel.

Useful entry points:

- `examples/bench_ffn.py` — kernel microbench (calls
  `lite_linear._cuda.fused_forward` directly) on the captured LTX-Video
  FFN shape set.
- `examples/prewarm_litelinear.py` — startup prewarm helper for known
  fused-forward FFN shapes.
- `examples/bench_litelinear.py` — module-level bench (`LiteLinear` vs
  `nn.Linear`, optional TE comparison).
- `examples/bench_litelinear_amd.py` — same for the ROCm path.

The older `examples/bench_lrdelta*.py` names remain as compatibility wrappers.

## Known caveats

- The fused kernel hardcodes the x→FP8 cast at `scale=1.0`. Activations
  with `|x| > 448` (NVIDIA) / `|x| > 240` (AMD) saturate silently. If
  this is a problem for your workload, see the input-scale discussion in
  the upstream `lite_linear/linear.py` docstring.
- LiteLinear is inference-only: the autograd `Function` wrapping the
  fused kernel raises on `.backward()`.
- LiteLinear requires GPU inputs; running `forward` on CPU raises. PyTorch uses
  `cuda` APIs and device strings for ROCm builds too.
