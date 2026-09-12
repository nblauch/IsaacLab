"""Check both pose-refresh flags during normal upstream environment steps."""

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
add_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.output.mkdir(parents=True, exist_ok=True)

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

task = "Isaac-Reorient-KukaAllegro-Camera"
cfg, _ = resolve_task_config(task, None, overrides=["presets=newton_mjwarp,ovrtx,cube,duo_camera,rgb64"])
cfg.scene.num_envs = 4
cfg.sim.device = "cuda:0"
cfg.seed = 42
cfg.scene.base_camera.update_latest_camera_pose = True
cfg.scene.wrist_camera.update_latest_camera_pose = True
for cam_cfg in (cfg.scene.base_camera, cfg.scene.wrist_camera):
    cam_cfg.renderer_cfg.log_file_path = str(args.output.resolve() / "native.log")


def pixels(cam):
    return cam.data.output["rgb"].torch.detach().cpu().numpy()[..., :3].copy()


def save(name, data):
    for i, frame in enumerate(data):
        Image.fromarray(frame).save(args.output / f"{name}-env{i}.png")


def mae(a, b):
    return np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=(1, 2, 3)).tolist()


with launch_simulation(cfg, args):
    with gym.make(task, cfg=cfg) as wrapped, torch.inference_mode():
        env = wrapped.unwrapped
        wrapped.reset(seed=42)
        cams = {name: env.scene.sensors[f"{name}_camera"] for name in ("base", "wrist")}
        report = {
            "task": task,
            "presets": "newton_mjwarp,ovrtx,cube,duo_camera,rgb64",
            "num_envs": 4,
            "ovrtx": version("ovrtx"),
            "ovstage": version("ovstage"),
            "gpu": torch.cuda.get_device_name(),
            "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
            "refresh_pose_from_initialization": {name: cam.cfg.update_latest_camera_pose for name, cam in cams.items()},
            "shared_renderer": cams["base"]._renderer is cams["wrist"]._renderer,
            "render_products": list(cams["base"]._renderer._render_product_paths),
            "action_value": 0.5,
            "frames": [],
        }
        actions = torch.full(wrapped.action_space.shape, 0.5, device=env.device)
        for step in range(5):
            wrapped.step(actions)
            # Capture actual normal-step sensor outputs before any diagnostic updates.
            normal = {name: pixels(cam) for name, cam in cams.items()}
            positions = {name: cam.data.pos_w.torch.cpu().tolist() for name, cam in cams.items()}
            q_before = env.scene["robot"].data.joint_pos.torch.clone()
            # Reference captures use the same physical state, settling each eye alone.
            reference = {}
            for name, cam in cams.items():
                for _ in range(24):
                    cam.update(0.0, force_recompute=True)
                reference[name] = pixels(cam)
            assert torch.equal(q_before, env.scene["robot"].data.joint_pos.torch)
            comparisons = {}
            for name, other in (("base", "wrist"), ("wrist", "base")):
                comparisons[name] = {
                    "normal_vs_own_reference_mae": mae(normal[name], reference[name]),
                    "normal_vs_other_reference_mae": mae(normal[name], reference[other]),
                }
                save(f"step{step}-normal-{name}", normal[name])
                save(f"step{step}-reference-{name}", reference[name])
            report["frames"].append({"step": step, "positions": positions, "comparisons": comparisons})
        checks = [
            own < other
            for frame in report["frames"]
            for value in frame["comparisons"].values()
            for own, other in zip(value["normal_vs_own_reference_mae"], value["normal_vs_other_reference_mae"])
        ]
        report["all_normal_images_closer_to_own_view"] = all(checks)
        report["image_checks"] = len(checks)
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print("RESULTS", json.dumps(report), flush=True)
        assert all(checks), "Some normal-step images are closer to the other camera's view."
