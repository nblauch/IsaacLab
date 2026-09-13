"""Capture upstream duo-camera temporal behavior without camera-routing or history patches."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from importlib.metadata import version
import warp as wp
wp.config.enable_backward = False
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--refresh', type=int, choices=[0,1], required=True)
parser.add_argument('--count', type=int, default=4096)
add_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.output.mkdir(parents=True, exist_ok=True)
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
import isaaclab_tasks
from isaaclab_tasks.utils import resolve_task_config
TASK = 'Isaac-Reorient-KukaAllegro-Camera'
PRESETS = 'newton_mjwarp,ovrtx,cube,duo_camera,rgb64'
cfg,_ = resolve_task_config(TASK, None, overrides=[f'presets={PRESETS}'])
cfg.scene.num_envs = args.count
cfg.sim.device = 'cuda:0'
cfg.seed = 42
for camera in (cfg.scene.base_camera,cfg.scene.wrist_camera):
    camera.update_latest_camera_pose = bool(args.refresh)
    camera.renderer_cfg.log_file_path = str(args.output.resolve()/'native.log')

def encode(name, panels, codec):
    h,w=panels.shape[1:3]
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','30','-i','-','-an',*codec,str(args.output/name)],input=panels.tobytes(),check=True)

with launch_simulation(cfg,args):
    with gym.make(TASK,cfg=cfg) as wrapped, torch.inference_mode():
        env=wrapped.unwrapped
        wrapped.reset(seed=42)
        cameras=[env.scene.sensors[name] for name in ('base_camera','wrist_camera')]
        action=torch.full(wrapped.action_space.shape,0.5,device=env.device)
        for _ in range(5):
            wrapped.step(action)
        def capture():
            # Intentionally do not explicitly publish poses: exercise configured flag.
            images=[]
            for cam in cameras:
                cam.update(cfg.sim.dt,force_recompute=True)
                images.append(cam.data.output['rgb'].torch[:3,...,:3].cpu().numpy().copy())
            return np.stack(images)
        for _ in range(40):
            capture()
        frozen=env.scene['robot'].data.joint_pos.torch.clone()
        start_hash=hashlib.sha256(frozen.cpu().numpy().tobytes()).hexdigest()
        frames=[]
        for i in range(360):
            if 120 <= i < 240:
                wrapped.step(action)
            elif i < 120:
                assert torch.equal(frozen,env.scene['robot'].data.joint_pos.torch)
            if i == 240:
                frozen=env.scene['robot'].data.joint_pos.torch.clone()
            if i>=240:
                assert torch.equal(frozen,env.scene['robot'].data.joint_pos.torch)
            frames.append(capture())
            if i%60==0:
                print('FRAME',i,flush=True)
        frames=np.asarray(frames)
        np.save(args.output/'frames.npy',frames)
        panels=np.concatenate([np.concatenate([frames[:,0,i],frames[:,1,i]],axis=2) for i in range(3)],axis=1)
        encode('stereo.mp4',panels,['-c:v','libx264','-crf','12','-pix_fmt','yuv420p','-movflags','+faststart'])
        encode('stereo-lossless.mkv',panels,['-c:v','ffv1','-level','3'])
        Image.fromarray(panels[60]).save(args.output/'static-frame.png')
        metrics={}
        for name,start,end in [('static_start',20,120),('static_end',260,360)]:
            data=frames[start:end].astype(np.float32)
            metrics[name]={'frame_difference_mae_by_eye':np.abs(np.diff(data,axis=0)).mean(axis=(0,2,3,4,5)).tolist(),'temporal_std_by_eye':data.std(axis=0).mean(axis=(1,2,3,4)).tolist()}
        result={'task':TASK,'presets':PRESETS,'num_envs':args.count,'refresh':bool(args.refresh),'seed':42,'gpu':torch.cuda.get_device_name(),'ovrtx':version('ovrtx'),'ovstage':version('ovstage'),'frames':360,'fps':30,'phases':'0-4s frozen; 4-8s nonzero action steps plus capture; 8-12s frozen','static_start_joints_sha256':start_hash,'static_metrics':metrics,'shared_renderer':cameras[0]._renderer is cameras[1]._renderer,'render_products':list(cameras[0]._renderer._render_product_paths),'layout':'Rows env0/1/2, columns base/wrist. Native 64x64 RGB.','camera_routing_or_history_patches':False,'local_configuration_patch':'OVRTXRendererConfig(enable_geometry_streaming=False); Kit UJITSO disabled (Kit not launched in these OVRTX runs).','tested_isaaclab_commit':'9adf4831384dbe9320de72e9b7809c0d87252b9a'}
        (args.output/'results.json').write_text(json.dumps(result,indent=2))
        print('COMPLETE',json.dumps(result),flush=True)
