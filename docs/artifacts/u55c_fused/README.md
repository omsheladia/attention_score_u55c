# U55C Fused Build Artifacts

This directory keeps generated evidence for the final fused U55C attention
build without mixing it into the top-level `docs/` folder.

- `reports/`: post-route utilization and timing reports copied from the Vitis
  hardware build.
- `vitis/`: xclbin info/link summaries, system estimates, and system diagrams.
- `guidance/`: Vitis analyzer guidance HTML reports from compile/link steps.

These files are not source inputs for the host or HLS build. They are retained
for reporting, presentation, and reproducibility.
