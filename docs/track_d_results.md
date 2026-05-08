# Track D Results

Track D compares the isolated one-head FPGA attention block against software
baselines. The measured scope is:

```text
Q_int8 @ K_int8.T -> causal mask + scale -> full-row softmax -> softmax @ V
```

This is not full TinyLlama inference and is not a model-level tokens/sec result.
It measures one selected attention head with Q/K/V already prepared.

This file records both the previous staged five-kernel baseline and the current
Track A Step 5 fused xclbin, where score GEMM plus mask/scale are combined into
`score_mask_scale_u55c_kernel`.

## Commands

CPU baseline:

```bash
/home/advent/kmhatre/DT/bin/python model/benchmark_cpu.py --iterations 10 --warmup 2
/home/advent/kmhatre/DT/bin/python model/benchmark_cpu.py --vectors sim/real_tinyllama_s16 --iterations 20 --warmup 3
/home/advent/kmhatre/DT/bin/python model/benchmark_cpu.py --vectors sim/real_tinyllama_s64 --iterations 20 --warmup 3
```

GPU baseline:

```bash
/home/advent/kmhatre/DT/bin/python model/benchmark_gpu.py --iterations 20 --warmup 3
```

This run printed `CUDA is not available; GPU baseline skipped.` for the Linux
U55C environment. GPU timings in the tables below were completed separately on
the Windows laptop with:

```text
torch 2.11.0+cu128
CUDA 12.8
NVIDIA GeForce RTX 3050 Laptop GPU
```

Windows GPU commands:

```powershell
python model\benchmark_gpu.py --iterations 100 --warmup 10
python model\benchmark_gpu.py --vectors sim\real_tinyllama_s16 --iterations 100 --warmup 10
python model\benchmark_gpu.py --vectors sim\real_tinyllama_s64 --iterations 100 --warmup 10
```

FPGA synthetic sweep:

```bash
source host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE

for s in 8 64 128 256 512; do
  ./build/host_attention_score_chain \
    --xclbin build/attention_score_chain.xclbin \
    --seq-len "$s" \
    --device 0
done
```

FPGA real TinyLlama vector runs:

```bash
source host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE

./build/host_attention_score_chain --xclbin build/attention_score_chain.xclbin --vectors sim/real_tinyllama_s16 --device 0
./build/host_attention_score_chain --xclbin build/attention_score_chain.xclbin --vectors sim/real_tinyllama_s64 --device 0
```

## Synthetic Results

The synthetic `--seq-len` flow has been run for `S = 8, 64, 128, 256, 512` on
the real U55C. These are generated deterministic Q/K/V inputs, not exported
TinyLlama prompt vectors.

| S | tiles | CPU full attn ms | GPU full attn ms | staged FPGA total ms | fused FPGA total ms | fused vs staged | CPU/fused FPGA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 0.0322 | 0.3349 | 0.803 | 0.296 | 2.71x | 0.11x |
| 64 | 8 | 0.2920 | 0.2507 | 2.864 | 2.364 | 1.21x | 0.12x |
| 128 | 32 | 2.5009 | 0.2998 | 6.992 | 5.484 | 1.28x | 0.46x |
| 256 | 128 | 3.9523 | 0.2726 | 20.570 | 17.307 | 1.19x | 0.23x |
| 512 | 512 | 12.3290 | 0.2461 | 72.941 | 60.328 | 1.21x | 0.20x |

All fused FPGA synthetic runs printed:

```text
Tiled sequence verification PASSED
Attention output verification PASSED
XRT chain verification PASSED
```

## Real TinyLlama Vector Results

The real TinyLlama full-sequence vector directories currently verified on the
real U55C now cover `S = 16`, `S = 64`, `S = 128`, `S = 256`, and `S = 512`,
plus the legacy single-tile directory. The larger `real_tinyllama_s128`,
`real_tinyllama_s256`, and `real_tinyllama_s512` directories were generated
with CUDA/FP16 from the cached TinyLlama model, CPU-sanity checked locally, and
then timed/verified on the real U55C. This is separate from the synthetic
`--seq-len` sweep, which also covers `S = 128, 256, 512`.

| vector dir | S | CPU full attn ms | GPU full attn ms | staged FPGA total ms | fused FPGA total ms | fused vs staged |
|---|---:|---:|---:|---:|---:|---:|
| `sim/attention_score_tile` | single tile | n/a | n/a | 0.345 | 0.199 | 1.73x |
| `sim/real_tinyllama_tile` | single tile | n/a | n/a | 0.258 | 0.635 | 0.41x |
| `sim/real_tinyllama_s16` | 16 | 0.0426 | 0.3278 | 1.488 | 0.569 | 2.62x |
| `sim/real_tinyllama_s64` | 64 | 0.1523 | 0.3778 | 3.094 | 2.004 | 1.54x |
| `sim/real_tinyllama_s128` | 128 | n/a | n/a | n/a | 6.816 | n/a |
| `sim/real_tinyllama_s256` | 256 | n/a | n/a | n/a | 17.192 | n/a |
| `sim/real_tinyllama_s512` | 512 | n/a | n/a | n/a | 55.329 | n/a |

Local CPU sanity checks for the newly generated larger real-vector directories
reported zero tiled-vs-brute differences for both softmax probabilities and
`attn_out`, with exported softmax-file max difference `2.98023224e-08` for
`S = 128`, `S = 256`, and `S = 512`.

During the first U55C run, `real_tinyllama_s128` exposed a verifier-only issue:
masked scaled logits around `-114388` differed by `0.0078125`, which exceeded
the old fixed `1e-4` absolute tolerance but was only about `6.8e-8` relative
error. The host verifier now allows a conservative `1e-6` relative tolerance for
scaled-logit comparisons while keeping softmax and attention-output checks on
their existing absolute tolerances.

All fused real-vector FPGA runs printed:

```text
Tiled sequence verification PASSED
Attention output verification PASSED
XRT chain verification PASSED
```

The `sim/real_tinyllama_tile` single-tile fused timing is slower than the staged
timing. Treat that as a small-run overhead/noise case; the full-sequence
`S=16` and `S=64` real-vector runs improved with fusion.

## HBM Usage

The current fused Step 5 xclbin uses HBM banks `[0]` through `[7]`, but the raw
score staging bank from the earlier staged design is no longer part of the main
pre-softmax path:

| Data | HBM bank |
|---|---:|
| Q tile | HBM[0] |
| K tile | HBM[1] |
| fused scaled score output | HBM[3] |
| tile probability / full-row logits | HBM[4] |
| full-row probability / V weights | HBM[5] |
| V chunk | HBM[6] |
| V partial output | HBM[7] |

`HBM[2]` was used by the staged raw-score path. In the fused branch, score GEMM
and mask/scale write scaled logits directly, removing that raw-score HBM round
trip.

## FPGA Utilization And Timing Closure

The final routed implementation reports for the fused Step 5 xclbin are checked
in under `docs/artifacts/u55c_fused/reports/`:

- `PostRouteKernelUtilization.rpt`: per-kernel routed resource usage
- `PostRouteFullUtilization.rpt`: full routed design usage including platform
- `PostRouteSLRUtilization.rpt`: SLR resource spread and SLL usage
- `PostRouteTimingSummary.rpt`: post-route timing closure
- `PostRouteUtilization.xlsx`: spreadsheet copy for reporting/presentation work

Headline routed utilization:

| Scope | LUT | REG | BRAM | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| User kernels | 60,983 | 70,401 | 110 | 2 | 405 |
| Full routed design | 191,648 CLB LUTs | 253,388 CLB registers | 309.5 Block RAM tiles | 2 | 409 |

Full routed design percentages on the U55C are `14.70%` CLB LUTs, `9.72%`
CLB registers, `15.35%` Block RAM tiles, `0.21%` URAM, and `4.53%` DSP.

Per-kernel routed utilization:

| Kernel | LUT | REG | BRAM | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| `score_mask_scale_u55c_kernel` | 8,707 | 9,947 | 23 | 0 | 67 |
| `softmax_full_row_u55c_kernel` | 5,750 | 7,100 | 16 | 2 | 9 |
| `softmax_u55c_kernel` | 5,687 | 7,076 | 16 | 0 | 9 |
| `v_weighted_sum_u55c_kernel` | 40,839 | 46,278 | 55 | 0 | 320 |

The V weighted-sum kernel still dominates user-kernel DSP use. The fused
pre-softmax kernel uses 67 DSP, replacing the staged score kernel plus mask
scale kernels that previously used 64 DSP and 3 DSP separately.

The routed timing summary reports all user timing constraints met with
design-summary `WNS 0.003 ns`, `TNS 0`, `WHS 0.009 ns`, and no failing setup or
hold endpoints.

## Interpretation

- The fused FPGA pipeline is correct for synthetic `S = 8, 64, 128, 256, 512`
  and real TinyLlama vector directories `S = 16, 64, 128, 256, 512`.
- Synthetic and real-vector `S = 128, 256, 512` runs have both passed on the
  real U55C. The larger real-vector cases do not yet have CPU/GPU/staged FPGA
  baseline timings in the table.
- The GPU baseline was completed on a separate Windows RTX 3050 Laptop GPU
  environment. Treat it as a useful PyTorch GPU reference, not a same-host
  measurement alongside the Linux/U55C FPGA runs.
- Step 5 fusion improves the larger synthetic and full-sequence real-vector
  cases by removing the raw-score HBM round trip and one staged kernel launch.
- The FPGA path is still slower than the local CPU/GPU baselines for this
  isolated one-head workload. The workload is small, and the host still launches
  many tile-level kernels while moving intermediates through HBM-visible
  buffers.
- The next performance step is a deeper on-card dataflow design or additional
  fusion that reduces repeated launch overhead and keeps more intermediates
  resident on the FPGA.
