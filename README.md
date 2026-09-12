# OVRTX duo-camera GPU reproduction

Native GPU validation of two IsaacLab `Camera` sensors sharing one OVRTX camera binding/render product.

Tested unmodified IsaacLab commit `9adf4831384dbe9320de72e9b7809c0d87252b9a`, task `Isaac-Reorient-KukaAllegro-Camera`, presets `newton_mjwarp,ovrtx,cube,duo_camera,rgb64`, four environments, seed 42. No renderer patches. The OVRTX renderer file is byte-for-byte identical at upstream develop `487b5cac4a2a94ddd60070b3de21a16904ef0679`; the newer revision was source-checked, not GPU-tested.

## Run

With that IsaacLab revision and its pinned dependencies installed, select an idle GPU UUID:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-YOUR-IDLE-GPU-UUID
export ISAAC_LAB_OVRTX_USE_OVSTAGE=0
# Used in this pip OpenUSD + pinned OVRTX environment:
export OVRTX_SKIP_USD_CHECK=1
python validate_gpu.py --output ./captures --visualizer none
```

The script runs five zero-action environment steps, then holds physics still. It captures the default two-camera output, independently refreshed camera views, and a cross-camera interference control. The only behavioral change in the controls is the public `CameraCfg.update_latest_camera_pose` flag. Private attributes are inspected only to report renderer identity and the native render-product path.

## Files

- `validate_gpu.py`: standalone native GPU reproducer.
- `results.json`: metrics and runtime camera configuration for all four environments.
- `images/`: untouched native RGB PNGs, 64 by 64, for every environment and condition.
- `validation.json`: dependency versions, source identity, successful exit and GPU isolation summary.

`default-base` and `default-wrist` are nearly the same stale view. `refresh-base` and `refresh-wrist` show different viewpoints. `base-published-wrist` shows the base view after publishing the base camera pose without refreshing the wrist pose. These are real renderer outputs; no synthetic or generated images are included.

Mean absolute pixel-channel differences over all four environments, on the 0–255 scale:

| Comparison | MAE |
| --- | ---: |
| Default base vs wrist | 0.1368 |
| Independently refreshed base vs wrist | 84.9075 |
| Wrist after base publication vs base | 0.9609 |
| Wrist after base publication vs its own refreshed view | 84.9224 |

Small nonzero differences between repeated same-view captures are compatible with temporal rendering; the report does not claim bit-identical pixels. The run validates camera routing, not FPS or a performance regression magnitude.
