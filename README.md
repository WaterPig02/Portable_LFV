# RealDynLFV Processing Pipeline

RealDynLFV provides a staged post-processing pipeline for 5x5 dynamic light-field video captures. It turns multi-camera raw videos into synchronized clips, calibration assets, rectification maps, release crops, benchmark frames, and QA reports.

The public v0 pipeline is designed for reproducible dataset construction: every stage can be checked, rerun, and configured through a local YAML file.

## ✨ What You Can Do

- Check audio synchronization manifests.
- Clip raw videos into synchronized MP4 files.
- Run checkerboard-based camera calibration.
- Generate rectification maps and release ROI metadata.
- Export benchmark frames as `8-bit JPEG + LUT + 1920x1080`.
- Validate exported scenes with frame/view/resolution/metadata/border checks.

## 🚀 Quick Start

### 1. Install dependencies

```powershell
pip install opencv-python numpy scipy matplotlib sounddevice tqdm PyYAML Pillow
```

Install FFmpeg separately and make sure these commands work:

```powershell
ffmpeg -version
ffprobe -version
```

The current synchronized clipping stage uses FFmpeg/NVENC. If your machine does not provide NVENC, check the dry-run command before launching a full clip job.

### 2. Create a local batch config

```powershell
Copy-Item configs\batches\batch.example.yaml configs\batches\my_batch.local.yaml
```

Edit `configs/batches/my_batch.local.yaml` with your own paths. Local configs are ignored by Git.

### 3. Run one stage at a time

Preflight first:

```powershell
python scripts/run_pipeline.py --config configs/batches/my_batch.local.yaml --stage export --profile benchmark --scene 0001 --preflight-only
```

Print the resolved command:

```powershell
python scripts/run_pipeline.py --config configs/batches/my_batch.local.yaml --stage export --profile benchmark --scene 0001 --dry-run
```

Export one benchmark scene:

```powershell
python scripts/run_pipeline.py --config configs/batches/my_batch.local.yaml --stage export --profile benchmark --scene 0001
```

Run QA for that scene:

```powershell
python scripts/run_pipeline.py --config configs\batches\my_batch.local.yaml --stage export-qa --profile benchmark --scene 0001
```

After one scene is verified, export the full benchmark split:

```powershell
python scripts/run_pipeline.py --config configs\batches\my_batch.local.yaml --stage export --profile benchmark
python scripts/run_pipeline.py --config configs\batches\my_batch.local.yaml --stage export-qa --profile benchmark
```

## 🧭 Pipeline

```text
raw videos
  -> sync manifest QA
  -> synchronized MP4 clips
  -> checkerboard calibration
  -> calibration QA
  -> rectification maps
  -> release ROI
  -> benchmark frame export
  -> export QA
```

Benchmark frame export keeps this image-processing order:

```text
decode -> rectification -> common ROI -> fixed 16:9 crop
       -> benchmark resize -> LUT -> 8-bit JPEG export
```

## 🧩 Stages

Use `scripts/run_pipeline.py --stage <name>` to run a stage.

| Stage | Purpose | Main output |
| --- | --- | --- |
| `sync-qa` | Check sync manifest offsets | offset QA summary |
| `sync-clip` | Clip raw videos using the sync manifest | synchronized MP4 files |
| `calibrate` | Run checkerboard calibration | calibration JSON |
| `calibration-qa` | Diagnose calibration quality | QA report and debug frames |
| `rectify` | Generate rectification maps | `.npz` maps and `rectify_meta.json` |
| `roi` | Generate release ROI metadata | `release_roi_metadata.json` |
| `roi-qa` | Validate ROI on sampled frames | ROI validation report |
| `export` | Export benchmark frames | scene/view/frame folders and metadata |
| `export-qa` | Validate exported frames | QA reports and check visuals |

Pipeline v0 intentionally keeps stages separate. There is no `--stage all`; run and verify each stage before moving to the next one.

## 📁 Input Layout

Use 25 camera folders named from `CAM_A1` to `CAM_E5`.

```text
raw_root/
  CAM_A1/
    0001.mp4
    0002.mp4
  CAM_A2/
    0001.mp4
    0002.mp4
  ...
  CAM_E5/
    0001.mp4
    0002.mp4
```

Rules:

- Use zero-padded scene IDs starting from `0001.mp4`.
- The same scene must use the same filename under all 25 camera folders.
- Use the same scene ID in the sync manifest, time segment JSON, and CLI `--scene`.
- Public v0 convention is lowercase `.mp4`.

Scenes missing any of the 25 views should not be used as official 5x5 benchmark scenes.

## ⚙️ Batch Config

Start from:

```text
configs/batches/batch.example.yaml
```

Key fields:

- `batch_id`
- `paths.raw_root`
- `paths.synced_root`
- `paths.sync_manifest`
- `paths.calibration_json`
- `paths.rectification_dir`
- `paths.roi_metadata`
- `paths.time_segments`
- `paths.output_root`
- `paths.lut`
- `camera.reference`
- `camera.nominal_fps`
- `camera.image_size`
- `calibration.checkerboard_inner_corners`
- `calibration.square_size_mm`

`paths.lut` must point to a local LUT file. This repository does not redistribute DJI `.cube` LUT files. If you use an official DJI LUT, confirm its license and download source yourself.

## ✅ Preflight Checks

Before running a stage, the wrapper checks:

- YAML readability.
- 5x5 / 25-view camera layout.
- `CAM_C3` reference camera.
- Required directories and JSON assets.
- Calibration and rectification camera coverage.
- Time segment and scene presence.
- Existing output paths.
- FFmpeg / ffprobe availability.
- Reported `hevc_nvenc` support for synchronized clipping.

Existing outputs generate warnings. Review warnings and dry-run commands before running a long job.

## 📦 Outputs

Typical benchmark output:

```text
output_root/
  <batch_id>/
    benchmark/
      0001/
        metadata.json
        view_00/
        view_01/
        ...
    checks/
      0001/
        check_report.json
```

Each scene contains sequence-level metadata and one folder per view. Metadata records phase offsets, ROI/crop policy, rectification version, LUT version, output format, and resolution.

## 📝 Notes

- The stable public target is the benchmark path: `8-bit JPEG + LUT + 1920x1080`.
- Fidelity-related code exists, but v0 does not describe it as a formal true 16-bit release.
- Legacy scripts are not part of the public pipeline v0.
- Real data, generated assets, exports, `.local.yaml` configs, and LUT files should remain outside Git.
