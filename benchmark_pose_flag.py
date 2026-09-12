"""Synchronized upstream task step timing with only camera pose refresh varied."""

import argparse
import atexit
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from contextlib import ExitStack

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--refresh", type=int, choices=(0, 1), required=True)
parser.add_argument("--backend", choices=("ovrtx", "isaacsim_rtx"), default="ovrtx")
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--steps", type=int, default=80)
add_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.output.mkdir(parents=True, exist_ok=True)

# Start Kit before task resolution imports pip OpenUSD. The later public
# launch_simulation call validates the actual task and reuses the live Kit app.
bootstrap = ExitStack()
if args.backend == "isaacsim_rtx":
    from isaaclab.physics import PhysicsCfg

    bootstrap.enter_context(launch_simulation(PhysicsCfg(), vars(args) | {"require_kit": True}))
    atexit.register(bootstrap.close)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from isaaclab.utils.io import dump_yaml

TASK = "Isaac-Reorient-KukaAllegro-Camera"
PRESETS = f"newton_mjwarp,{args.backend},cube,duo_camera,rgb64"
cfg, _ = resolve_task_config(TASK, None, overrides=[f"presets={PRESETS}"])
cfg.scene.num_envs = args.num_envs
cfg.sim.device = "cuda:0"
cfg.seed = 42
for cam_cfg in (cfg.scene.base_camera, cfg.scene.wrist_camera):
    cam_cfg.update_latest_camera_pose = bool(args.refresh)
    if args.backend == "ovrtx":
        cam_cfg.renderer_cfg.log_file_path = str(args.output.resolve() / "native.log")
dump_yaml(str(args.output / "env_cfg.yaml"), cfg)


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


with launch_simulation(cfg, args):
    with gym.make(TASK, cfg=cfg) as wrapped, torch.inference_mode():
        env = wrapped.unwrapped
        wrapped.reset(seed=42)
        generator = torch.Generator(device=env.device).manual_seed(12345)
        actions = torch.rand(
            (args.warmup + args.steps, *wrapped.action_space.shape),
            device=env.device, generator=generator,
        ) * 2 - 1
        report = {
            "task": TASK, "presets": PRESETS, "num_envs": args.num_envs,
            "refresh": bool(args.refresh), "seed": 42, "action_seed": 12345,
            "action_sha256": digest(actions),
            "initial_joint_sha256": digest(env.scene["robot"].data.joint_pos.torch),
            "ovrtx": version("ovrtx"), "ovstage": version("ovstage"),
            "newton": version("newton"), "warp": version("warp-lang"),
            "gpu": torch.cuda.get_device_name(),
            "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
            "dt": cfg.sim.dt, "decimation": cfg.decimation,
            "render_interval": cfg.sim.render_interval,
            "warmup_steps": args.warmup, "measured_steps": args.steps,
            "measurement": "Full env.step wall time, CUDA synchronized before and after each step; no policy.",
            "steps_ms": [], "terminated_envs": [], "truncated_envs": [],
        }
        if args.backend == "isaacsim_rtx":
            import carb

            settings = carb.settings.get_settings()
            report["isaac_rtx_settings"] = {
                key: settings.get(key) for key in (
                    "/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/dlss/execMode",
                    "/rtx/rtpt/maxBounces", "/rtx/rtpt/spp",
                    "/rtx/rtpt/cached/enabled", "/rtx/rtpt/lightcache/cached/enabled",
                )
            }
        print("ENV_READY", json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}), flush=True)
        for step in range(args.warmup):
            wrapped.step(actions[step])
        torch.cuda.synchronize()
        report["post_warmup_joint_sha256"] = digest(env.scene["robot"].data.joint_pos.torch)
        print("WARMUP_DONE", flush=True)
        for step in range(args.steps):
            action = actions[args.warmup + step]
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = wrapped.step(action)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            report["steps_ms"].append(elapsed * 1000)
            # Counts are collected outside the timed interval.
            report["terminated_envs"].append(int(output[2].sum().item()))
            report["truncated_envs"].append(int(output[3].sum().item()))
            if (step + 1) % 20 == 0:
                print("PROGRESS", step + 1, float(np.mean(report["steps_ms"])), flush=True)
        times = np.asarray(report["steps_ms"])
        report.update({
            "mean_step_ms": float(times.mean()),
            "median_step_ms": float(np.median(times)),
            "p95_step_ms": float(np.percentile(times, 95)),
            "env_fps": float(args.num_envs * 1000 / times.mean()),
            "final_joint_sha256": digest(env.scene["robot"].data.joint_pos.torch),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        })
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print("RESULTS", json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}), flush=True)
