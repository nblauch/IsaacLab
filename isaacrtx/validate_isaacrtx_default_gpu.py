"""Validate default IsaacRTX camera images against pose-refreshed controls."""

import argparse
import atexit
from contextlib import ExitStack
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

# Kit must be initialized before task imports select its OpenUSD bindings.
from isaaclab.physics import PhysicsCfg
bootstrap = ExitStack()
bootstrap.enter_context(launch_simulation(PhysicsCfg(), vars(args) | {"require_kit": True}))
atexit.register(bootstrap.close)

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

task = "Isaac-Reorient-KukaAllegro-Camera"
cfg, _ = resolve_task_config(task, None, overrides=["presets=newton_mjwarp,isaacsim_rtx,cube,duo_camera,rgb64"])
cfg.scene.num_envs = 4
cfg.sim.device = "cuda:0"
cfg.seed = 42
for cam_cfg in (cfg.scene.base_camera, cfg.scene.wrist_camera):
    assert cam_cfg.update_latest_camera_pose is False


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
            "presets": "newton_mjwarp,isaacsim_rtx,cube,duo_camera,rgb64",
            "num_envs": 4,
            "ovrtx": version("ovrtx"),
            "ovstage": version("ovstage"),
            "gpu": torch.cuda.get_device_name(),
            "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
            "refresh_pose_from_initialization": {name: cam.cfg.update_latest_camera_pose for name, cam in cams.items()},
            "shared_renderer": cams["base"]._renderer is cams["wrist"]._renderer,
            "render_products": {name: str(cam._render_data.render_product.path) for name, cam in cams.items()},
            "action_value": 0.5,
            "frames": [],
        }
        print("ENV_READY", json.dumps(report), flush=True)
        actions = torch.full(wrapped.action_space.shape, 0.5, device=env.device)
        for step in range(5):
            for _ in range(10):
                wrapped.step(actions)
            # Capture actual normal-step sensor outputs before any diagnostic updates.
            normal = {name: pixels(cam) for name, cam in cams.items()}
            positions = {name: cam.data.pos_w.torch.cpu().tolist() for name, cam in cams.items()}
            q_before = env.scene["robot"].data.joint_pos.torch.clone()
            physics_step_before = env.sim.get_physics_step_count()
            # Refresh controls and explicitly pump rendering at frozen physics.
            # Restore defaults before the next normal environment steps.
            reference = {}
            for cam in cams.values():
                cam.cfg.update_latest_camera_pose = True
            for _ in range(24):
                env.sim.render()
                for cam in cams.values():
                    cam.update(0.0, force_recompute=True)
                    pixels(cam)
            reference = {name: pixels(cam) for name, cam in cams.items()}
            refreshed_positions = {name: cam.data.pos_w.torch.cpu().tolist() for name, cam in cams.items()}
            for cam in cams.values():
                cam.cfg.update_latest_camera_pose = False
            assert physics_step_before == env.sim.get_physics_step_count()
            assert torch.equal(q_before, env.scene["robot"].data.joint_pos.torch)
            comparisons = {}
            for name, other in (("base", "wrist"), ("wrist", "base")):
                comparisons[name] = {
                    "normal_vs_own_reference_mae": mae(normal[name], reference[name]),
                    "normal_vs_other_reference_mae": mae(normal[name], reference[other]),
                }
                save(f"step{step}-normal-{name}", normal[name])
                save(f"step{step}-reference-{name}", reference[name])
            report["frames"].append({"step": step, "positions": positions, "refreshed_positions": refreshed_positions, "comparisons": comparisons, "normal_base_vs_wrist_mae": mae(normal["base"], normal["wrist"])})
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
