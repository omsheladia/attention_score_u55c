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
real U55C are `S = 16` and `S = 64`, plus the legacy single-tile directories.
Larger real TinyLlama vector directories such as `real_tinyllama_s128`,
`real_tinyllama_s256`, and `real_tinyllama_s512` have not been generated or run
yet. That is separate from the synthetic `--seq-len` sweep above, which already
covers `S = 128, 256, 512`.

| vector dir | S | CPU full attn ms | GPU full attn ms | staged FPGA total ms | fused FPGA total ms | fused vs staged |
|---|---:|---:|---:|---:|---:|---:|
| `sim/attention_score_tile` | single tile | n/a | n/a | 0.345 | 0.199 | 1.73x |
| `sim/real_tinyllama_tile` | single tile | n/a | n/a | 0.258 | 0.635 | 0.41x |
| `sim/real_tinyllama_s16` | 16 | 0.0426 | 0.3278 | 1.488 | 0.569 | 2.62x |
| `sim/real_tinyllama_s64` | 64 | 0.1523 | 0.3778 | 3.094 | 2.004 | 1.54x |

All fused real-vector FPGA runs printed:

```text
Tiled sequence verification PASSED
Attention output verification PASSED
XRT chain verification PASSED
```

The `sim/real_tinyllama_tile` single-tile fused timing is slower than the staged
timing. Treat that as a small-run overhead/noise case; the full-sequence
`S=16` and `S=64` real-vector runs improved with fusion.

## Optimization Track O1 Baseline And Profiling

Track O1 freezes the current fused Step 5 design as the baseline for future
optimization work.

Baseline identifiers:

| Item | Value |
|---|---|
| git commit | `329d6a41134e09e303b5f4e87f27c35193e43909` |
| xclbin UUID | `56cd611d-8c32-c19b-f5ef-358bed40d459` |
| branch when profiled | `optimization-device-resident-online-attn` |
| XRT/tool version | XRT 2022.2 / 2.14.354 |
| platform shell | `xilinx_u55c_gen3x16_xdma_base_3` |

Baseline synthetic command:

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

Baseline launch and timing table:

| S | score launches | softmax launches | V launches | total launches | score ms | softmax ms | V ms | kernel launch/wait sum ms | host/DMA/sync gap ms | total ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 1 | 1 | 3 | 0.064 | 0.076 | 0.037 | 0.177 | 0.119 | 0.296 |
| 64 | 8 | 8 | 8 | 24 | 0.230 | 0.628 | 0.275 | 1.132 | 1.232 | 2.364 |
| 128 | 32 | 16 | 32 | 80 | 0.913 | 1.391 | 1.090 | 3.393 | 2.092 | 5.484 |
| 256 | 128 | 32 | 128 | 288 | 3.808 | 3.170 | 4.672 | 11.651 | 5.656 | 17.307 |
| 512 | 512 | 64 | 512 | 1088 | 16.956 | 8.164 | 18.976 | 44.096 | 16.231 | 60.328 |

For `S=512`, the baseline spends about `73.1%` of host-measured chain time in
kernel launch-through-wait windows and about `26.9%` in the remaining host,
DMA, and sync gap. Inside the kernel launch/wait sum, score is `38.5%`, full-row
softmax is `18.5%`, and V weighted sum is `43.0%`.

XRT profiling command:

```bash
cp docs/xrt_profile_s512.ini xrt.ini
source host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE
./build/host_attention_score_chain \
  --xclbin build/attention_score_chain.xclbin \
  --seq-len 512 \
  --device 0
rm -f xrt.ini
```

The profiled `S=512` run passed:

```text
Tiled sequence verification PASSED
Attention output verification PASSED
XRT chain verification PASSED
```

Profiled run timing was slightly slower, as expected with tracing enabled:

| S | score ms | softmax ms | V ms | kernel launch/wait sum ms | host/DMA/sync gap ms | total ms |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 20.683 | 8.238 | 19.271 | 48.191 | 17.464 | 65.655 |

Profile artifacts copied into `docs/`:

- `xrt_profile_s512.ini`
- `optimization_o1_s512_profile_run.txt`
- `optimization_o1_s512_summary.csv`
- `optimization_o1_s512_native_trace.csv`
- `optimization_o1_s512_device_trace_0.csv`
- `optimization_o1_s512_xrt.run_summary`

Key XRT profile observations from `optimization_o1_s512_summary.csv`:

| Profile item | Count | Total time ms | Note |
|---|---:|---:|---|
| `xrt::run::start` | 1088 | 1.342 | one call per tile-level kernel launch |
| `xrt::run::wait` | 1088 | 25.820 | host wait time for launched kernels |
| `xrt::bo::sync` | 2816 | 24.171 | host/device buffer synchronization |
| Host reads from global memory | 1088 | 8.404 | max transfer 16 KB |
| Host writes to global memory | 1728 | 15.906 | max transfer 16 KB |

The profile confirms that the current path is dominated by launch count and
host-visible buffer staging, not FPGA fabric utilization. The next optimization
should therefore start with device-resident full buffers and fewer launches
rather than minor clock or directive tuning.

The XRT summary generated `device_trace_0.csv`, but it reported no accelerator
performance monitors (`NUM_MONITORS=0`) and the device trace marks compute
units as `No Trace`. Treat the native XRT API and host data-transfer profile as
the authoritative O1 evidence for this run.

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
in under `docs/`:

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
  and real TinyLlama vector directories `S = 16, 64`.
- Synthetic `S = 128, 256, 512` has been run. Larger real TinyLlama vector
  directories for those same sequence lengths have not been run yet.
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
