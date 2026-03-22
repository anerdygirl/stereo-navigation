# Mobility aid prototype (OAK-D S2)

**Research / prototype** — stereo depth–based scene understanding to support navigation for people with low vision or blindness. Not a certified medical or safety device.

## What it does

- Uses a **Luxonis OAK-D S2** (stereo RGB + depth) with **DepthAI**.
- Builds a **3×3 grid** over the depth image via `SpatialLocationCalculator` ROIs in normalized coordinates.
- For each cell, reads **3D coordinates** (X, Y, Z in mm) relative to the camera.
- **Obstacle cue:** if depth **Z &lt; 1 m** (and valid), the cell is highlighted in **red** and labeled as an obstacle; otherwise X/Y/Z are shown for inspection.
- Optionally applies a **median filter** on **disparity** for visualization; spatial depth uses **direct `StereoDepth` output** so depth + median ROI math stay stable.

## Requirements

- Python **3.10+** (3.12 works with recent DepthAI wheels)
- [depthai](https://docs.luxonis.com/) (must match your OAK firmware / platform)
- `opencv-python`, `numpy`

```bash
pip install depthai opencv-python numpy
```

## Hardware

- **OAK-D S2** (or compatible OAK device with stereo depth)
- USB 3 recommended for stable bandwidth

## Run

From this folder:

```bash
python "navigation - final.py"
```

- **Quit:** press `q` in the depth window.
- **Console:** obstacle events print as `OBSTACLE at Z=...mm` when something is closer than 1 m in that cell.

## Configuration knobs (in code)

| Setting | Role |
|--------|------|
| `num_rows`, `num_cols` | Grid size (default 3×3) |
| `z < 1000` | Obstacle distance threshold (mm) |
| `depthThresholds` on each ROI | Valid depth range for the calculator (mm) |
| Stereo preset / resolution | `requestOutput((640, 400))`, `FAST_ACCURACY`, etc. |

Tune thresholds and grid size for your mounting height, walking speed, and field of view.

## Future ideas

- **Haptic feedback** — map obstacle cells (or proximity zones) to vibration motors or a wearable actuator (e.g. stronger pulse = closer / more central threat).
- Audio cues (spatialized beeps, TTS distance bands).
- Larger or adaptive grid; temporal smoothing to reduce flicker.
- Calibration / IMU fusion if the camera is head- or chest-mounted.

## Limitations

- Prototype only: no guarantees in real traffic or clinical use.
- Depth quality depends on lighting, texture, and stereo baseline; black regions are often **no data**, not “infinite distance.”
- The 1 m rule is a simple demo; production systems need richer logic and testing.
