Reference-task timing evidence

The JSON files contain per-step samples for all included runs. All included runs exited successfully with no other compute process detected on the selected GPU. A fourth IsaacRTX run was excluded due to GPU interference; it is not included in any aggregate.

Run benchmark_pose_flag.py in the configured pinned IsaacLab environment, selecting an idle GPU with CUDA_VISIBLE_DEVICES. Example:

python benchmark_pose_flag.py --backend isaacsim_rtx --refresh 0 --num-envs 4096 --warmup 30 --steps 80 --output ./run --visualizer none

Use --backend ovrtx for the OVRTX comparison. See the issue for dependency versions, setup and renderer-quality limitations. ARTIFACT_ROOT in summary configuration differences is a portable replacement for local log output paths.
