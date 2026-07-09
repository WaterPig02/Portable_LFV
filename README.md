# RealDynLFV Processing Pipeline

This repository contains the post-processing pipeline used to prepare RealDynLFV benchmark data. Pipeline v0 is a thin orchestration layer over the existing scripts. It does not rewrite synchronization, calibration, rectification, ROI, export, or QA algorithms.

## Pipeline

```text
raw videos
  -> audio synchronization QA
  -> synchronized MP4 clips
  -> checkerboard calibration
  -> calibration QA
  -> rectification maps
  -> common valid ROI and fixed 16:9 crop
  -> benchmark frame export
  -> export QA reports
```

The benchmark exporter keeps this image-processing order unchanged:

```text
decode -> rectification -> common ROI -> fixed 16:9 crop
       -> benchmark resize -> LUT -> 8-bit JPEG export
```

## Pipeline v0 Scope

The unified entry point is:

```text
scripts/run_pipeline.py
```

Supported stage names:

- `sync-qa`
- `sync-clip`
- `calibrate`
- `calibration-qa`
- `rectify`
- `roi`
- `roi-qa`
- `export`
- `export-qa`

Each stage can be run independently with `--stage <name>`. Pipeline v0 intentionally does not provide `--stage all`; operators should run and verify stages step by step.

Not included in v0:

- GUI rewrite
- calibration or rectification algorithm changes
- export performance optimization
- full Python package conversion
- automatic white balance
- automatic residual epipolar correction
- a formal end-to-end true 16-bit fidelity release

The current fidelity path is not documented as a formal true 16-bit release. The benchmark path is the current stable release target.

Legacy scripts under `Code/legacy/` are retained for historical reference but are not part of the official pipeline v0.

## Batch Configuration

Start from the generic example:

```text
configs/batches/batch.example.yaml
```

Copy it to a local config and edit paths for your machine:

```powershell
Copy-Item configs\batches\batch.example.yaml configs\batches\my_batch.local.yaml
```

`*.local.yaml` files are ignored by Git and may contain real data paths. Public examples should use generic batch IDs and placeholder paths only.

The YAML config records:

- `batch_id` and optional `runtime.time_key`
- raw and synchronized video roots
- sync manifest
- calibration JSON and reports
- rectification and ROI assets
- selected time segments
- output root and user-provided LUT
- selected time segment JSON (`paths.time_segments`)
- 5x5 camera layout and `CAM_C3` reference camera
- nominal FPS and image size
- checkerboard parameters
- worker and tool settings
- benchmark export and QA thresholds

## Dependencies

PyYAML is required for the v0 configuration layer:

```powershell
pip install PyYAML
```

Main runtime dependencies include:

- Python 3
- FFmpeg and ffprobe
- OpenCV (`cv2`)
- NumPy
- SciPy
- Matplotlib
- sounddevice
- tqdm
- PyYAML
- NVIDIA driver/NVENC for the current `sync-clip` implementation

NVENC is detected during preflight. The v0 wrapper does not assume every machine provides it and does not silently substitute a different encoder.

## Commands

Run preflight only:

```powershell
python scripts/run_pipeline.py --config configs/batches/my_batch.local.yaml --stage sync-clip --preflight-only
```

Print the resolved command without executing it:

```powershell
python scripts/run_pipeline.py --config configs/batches/my_batch.local.yaml --stage calibrate --scene 0001 --dry-run
```

Export one benchmark scene:

```powershell
python scripts/run_pipeline.py --config configs/batches/my_batch.local.yaml --stage export --profile benchmark --scene 0027
```

Check benchmark outputs:

```powershell
python scripts/run_pipeline.py --config configs/batches/my_batch.local.yaml --stage export-qa --profile benchmark
```

The underlying scripts can still be called directly, but public use should pass `--config` or explicit paths. No-argument defaults are generic placeholders and are not intended for real datasets.

## Safety and Preflight

The v0 wrapper checks stage-relevant inputs before execution:

- YAML readability
- 5x5 / 25-view camera layout
- `CAM_C3` reference camera
- required directories and JSON assets
- calibration and rectification camera coverage
- time segment and scene presence
- existing output paths
- FFmpeg/ffprobe availability
- reported `hevc_nvenc` support for `sync-clip`

Existing outputs generate warnings. The underlying production scripts retain their existing overwrite/skip behavior, so operators must review resolved commands and warnings before execution.

## Data and Assets

Real data, calibration results, exports, generated maps, and local configs remain local and are not committed by default:

- `Calibration_Data/`
- `Output/`
- `Exports/`
- `configs/batches/*.local.yaml`

Time segment files are dataset-specific. Start from `configs/time.example.json`, then point `paths.time_segments` in your local YAML config to your own segment file.

LUT files are user-provided assets. This repository does not redistribute DJI `.cube` LUT files. If you use an official DJI LUT, configure its local path in YAML and confirm the license and download source yourself.
