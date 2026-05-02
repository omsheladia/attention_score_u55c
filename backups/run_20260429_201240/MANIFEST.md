# U55C Attention Score Hardware Run Backup

Created: 2026-04-29 20:12 America/Phoenix

Purpose: preserve the known-good isolated U55C attention-score demo after the
card shell was updated to base_3 and real hardware verification passed.

## Verified State

- XRT: 2.14.354 / 2022.2
- Vitis/Vivado: 2022.2
- Card shell: `xilinx_u55c_gen3x16_xdma_base_3`
- Platform UUID: `97088961-FEAE-DA91-52A2-1D9DFD63CCEF`
- Build platform: `xilinx_u55c_gen3x16_xdma_3_202210_1`
- Hardware xclbin: `build_artifacts/attention_score_chain.xclbin`
- Real hardware verification: `verification_logs/hw_run_20260429.txt`
- Result: `XRT chain verification PASSED`

## Run Command

From `/home/advent/Desktop/RC19`:

```bash
source attention_score_u55c/host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE
./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --vectors attention_score_u55c/sim/attention_score_tile \
  --device 0
```

## Backup Contents

- `repo_snapshot/`: source, scripts, docs, reference model, vector files, and current handoff
- `build_artifacts/`: xclbin, xo files, host binary, link summary, compile summaries, emconfig
- `hls_reports/`: copied HLS synthesis reports
- `vitis_reports/`: Vitis `_x/reports` and `_x/logs`
- `verification_logs/`: real hardware run log and hw_emu simulate log
- `device_state/`: XRT examine outputs, xclbin metadata, package list, lspci, checksums

## Notes

The detailed `xbmgmt examine --report platform` command requires root
privileges, so the non-root backup captures standard `xbutil examine` and
`xbmgmt examine` output instead.

