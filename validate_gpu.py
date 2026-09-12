"""Native GPU reproduction of the upstream Kuka duo-camera routing issue.

No renderer patches. The control changes only the public
CameraCfg.update_latest_camera_pose flag after capturing the default behavior.
"""

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--settle", type=int, default=24)
add_launcher_args(parser)
args, overrides = parser.parse_known_args()
args.headless = True
args.enable_cameras = True
args.output.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0]]
assert os.environ["CUDA_VISIBLE_DEVICES"].startswith("GPU-")
assert "," not in os.environ["CUDA_VISIBLE_DEVICES"]

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

TASK = "Isaac-Reorient-KukaAllegro-Camera"
cfg, _ = resolve_task_config(
    TASK, None,
    overrides=["presets=newton_mjwarp,ovrtx,cube,duo_camera,rgb64", *overrides],
)
cfg.scene.num_envs = args.num_envs
cfg.seed = 42
cfg.sim.device = "cuda:0"
for camera_cfg in (cfg.scene.base_camera, cfg.scene.wrist_camera):
    assert camera_cfg.update_latest_camera_pose is False
    camera_cfg.renderer_cfg.log_file_path = str(args.output.resolve() / "native.log")
cfg.validate()


def array(value):
    return value.torch.detach().cpu().numpy().copy()


def image(camera):
    return array(camera.data.output["rgb"])[..., :3]


def save(name, pixels):
    np.save(args.output / f"{name}.npy", pixels)
    for i, frame in enumerate(pixels):
        Image.fromarray(frame).save(args.output / f"{name}-env{i}.png")


def difference(a, b):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {"mae_0_255": float(d.mean()), "fraction_equal_channels": float((d == 0).mean())}


with launch_simulation(cfg, args):
    with gym.make(TASK, cfg=cfg) as wrapped, torch.inference_mode():
        env = wrapped.unwrapped
        wrapped.reset(seed=42)
        actions = torch.zeros(wrapped.action_space.shape, device=env.device)
        for _ in range(5):
            wrapped.step(actions)
        base = env.scene.sensors["base_camera"]
        wrist = env.scene.sensors["wrist_camera"]
        renderer = base._renderer
        report = {
            "task": TASK,
            "presets": "newton_mjwarp,ovrtx,cube,duo_camera,rgb64",
            "num_envs": args.num_envs,
            "seed": 42,
            "gpu": torch.cuda.get_device_name(),
            "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
            "ovrtx": version("ovrtx"),
            "ovstage": version("ovstage"),
            "use_ovstage": renderer._use_ovstage,
            "shared_renderer": base._renderer is wrist._renderer,
            "native_render_products": list(renderer._render_product_paths),
            "native_camera_relative_path": renderer._camera_rel_path,
            "cameras": {
                name: {
                    "prim_path": cam.cfg.prim_path,
                    "width": cam.cfg.width,
                    "height": cam.cfg.height,
                    "refresh_pose_default": cam.cfg.update_latest_camera_pose,
                    "initial_cached_positions": array(cam.data.pos_w).tolist(),
                } for name, cam in (("base", base), ("wrist", wrist))
            },
            "note": "Scene physics is frozen after five zero-action steps. Camera captures alone follow.",
        }
        print("ENV_READY", json.dumps(report), flush=True)
        # Preserve default camera behavior. Reads are copied before another eye renders.
        for _ in range(args.settle):
            base.update(0.0, force_recompute=True)
            baseline_base = image(base)
            wrist.update(0.0, force_recompute=True)
            baseline_wrist = image(wrist)
        save("default-base", baseline_base)
        save("default-wrist", baseline_wrist)
        report["default_pair"] = difference(baseline_base, baseline_wrist)

        # Public config control, no native binding or implementation changes.
        base.cfg.update_latest_camera_pose = True
        wrist.cfg.update_latest_camera_pose = True
        controls = {}
        for name, cam in (("base", base), ("wrist", wrist)):
            for _ in range(args.settle):
                cam.update(0.0, force_recompute=True)
                controls[name] = image(cam)
            save(f"refresh-{name}", controls[name])
            report["cameras"][name]["refreshed_positions"] = array(cam.data.pos_w).tolist()
        report["refreshed_pair"] = difference(controls["base"], controls["wrist"])
        report["default_base_vs_refreshed_base"] = difference(baseline_base, controls["base"])
        report["default_base_vs_refreshed_wrist"] = difference(baseline_base, controls["wrist"])

        # Stronger routing check: publish only base's pose, then read wrist with
        # refresh disabled. Its image should remain wrist's if views are independent.
        wrist.cfg.update_latest_camera_pose = False
        for _ in range(args.settle):
            base.update(0.0, force_recompute=True)
        base_control = image(base)
        for _ in range(args.settle):
            wrist.update(0.0, force_recompute=True)
        wrist_after_base = image(wrist)
        save("base-published-base", base_control)
        save("base-published-wrist", wrist_after_base)
        report["wrist_after_base_publication_vs_base"] = difference(wrist_after_base, base_control)
        report["wrist_after_base_publication_vs_wrist"] = difference(wrist_after_base, controls["wrist"])
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print("RESULTS", json.dumps(report), flush=True)
