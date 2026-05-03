# Track D Results

Track D compares the current isolated one-head FPGA attention block against
software baselines. The measured scope is:

```text
Q_int8 @ K_int8.T -> causal mask -> scale -> full-row softmax -> softmax @ V
```

This is not full TinyLlama inference and is not a model-level tokens/sec result.
It measures one selected attention head with Q/K/V already prepared.

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

This run printed `CUDA is not available; GPU baseline skipped.` for the current
Linux environment.

FPGA synthetic sweep:

```bash
source attention_score_u55c/host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE

for s in 8 64 128 256 512; do
  ./attention_score_u55c/build/host_attention_score_chain \
    --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
    --seq-len "$s" \
    --device 0
done
```

FPGA real TinyLlama vector runs:

```bash
source attention_score_u55c/host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --vectors attention_score_u55c/sim/real_tinyllama_s16 \
  --device 0

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --vectors attention_score_u55c/sim/real_tinyllama_s64 \
  --device 0
```

## Synthetic Results

`FPGA compute` is the sum of kernel launch-through-wait timings printed by the
host. `Host/DMA gap` is `total_chain - FPGA compute`; it includes BO writes,
syncs, reads, host staging, and other non-kernel overhead observed by the host.

| S | tiles | CPU full attn ms | GPU full attn ms | FPGA compute ms | Host/DMA gap ms | FPGA total ms | CPU/FPGA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 0.0322 | N/A | 0.275 | 0.528 | 0.803 | 0.04x |
| 64 | 8 | 0.2920 | N/A | 2.143 | 0.721 | 2.864 | 0.10x |
| 128 | 32 | 2.5009 | N/A | 5.250 | 1.742 | 6.992 | 0.36x |
| 256 | 128 | 3.9523 | N/A | 18.601 | 1.969 | 20.570 | 0.19x |
| 512 | 512 | 12.3290 | N/A | 69.792 | 3.149 | 72.941 | 0.17x |

All FPGA runs printed:

```text
Tiled sequence verification PASSED
Attention output verification PASSED
XRT chain verification PASSED
```

## Real TinyLlama Vector Results

| vector dir | S | tiles | CPU full attn ms | GPU full attn ms | FPGA compute ms | Host/DMA gap ms | FPGA total ms | CPU/FPGA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sim/real_tinyllama_s16` | 16 | 2 | 0.0426 | N/A | 0.582 | 0.906 | 1.488 | 0.03x |
| `sim/real_tinyllama_s64` | 64 | 8 | 0.1523 | N/A | 1.722 | 1.372 | 3.094 | 0.05x |

Both real-vector FPGA runs printed:

```text
Tiled sequence verification PASSED
Attention output verification PASSED
XRT chain verification PASSED
```

## HBM Usage

The current five-kernel xclbin uses HBM banks `[0]` through `[7]`:

| Data | HBM bank |
|---|---:|
| Q tile | HBM[0] |
| K tile | HBM[1] |
| raw score tile | HBM[2] |
| scaled score tile | HBM[3] |
| tile probability / full-row logits | HBM[4] |
| full-row probability / V weights | HBM[5] |
| V chunk | HBM[6] |
| V partial output | HBM[7] |

This is the main HBM story for the demo: the staged design spreads independent
traffic across eight HBM banks instead of routing all buffers through one global
memory bank. The tradeoff is that the current kernels still round-trip staged
intermediates through HBM and the host, so the design showcases HBM placement
but does not yet keep the full attention pipeline resident on-card.

## Interpretation

- The current FPGA pipeline is correct for synthetic `S = 8, 64, 128, 256, 512`
  and real TinyLlama `S = 16, 64`.
- The staged FPGA implementation is slower than the local CPU NumPy baseline
  for this one-head workload. That is expected for the current architecture:
  the work is small, kernels are launched many times, and intermediate data is
  moved through HBM/host-visible buffers between stages.
- Larger `S` shifts the bottleneck toward kernel compute/launch time. At
  `S=512`, the measured host/DMA gap is only `3.149 ms` of `72.941 ms`; the
  staged kernel work dominates.
- Small real-vector cases are dominated by fixed overhead. At `S=16`, the
  host/DMA gap is larger than the kernel sum.
- The next performance step is not more benchmarking. It is reducing staged
  launches and HBM round trips, most likely by fusing mask/scale/softmax/V work
  or moving to a persistent on-card dataflow design.
