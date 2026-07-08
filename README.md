# RealDynLFV Processing Pipeline

This repository contains the internal post-processing pipeline used to prepare the RealDynLFV dataset. Pipeline v0 provides a lightweight configuration, preflight, and orchestration layer over the existing scripts. It does not rewrite the synchronization, calibration, rectification, ROI, or export algorithms.

## Pipeline

```text
raw videos
  -> audio synchronization
  -> synchronized MP4 clips
  -> checkerboard calibration
  -> rectification maps
  -> common valid ROI and fixed 16:9 crop
  -> benchmark frame export
  -> QA reports
```

The current production exporter keeps this image-processing order unchanged:

```text
decode -> rectification -> common ROI -> fixed 16:9 crop
       -> benchmark resize -> LUT -> 8-bit JPEG export
```

## Pipeline v0 Scope

Supported orchestration stages:

- `sync-qa`
- `sync-clip`
- `calibrate`
- `calibration-qa`
- `rectify`
- `roi`
- `roi-qa`
- `export`
- `export-qa`

Pipeline v0 is intentionally a thin wrapper. Some QA/export stages still require the existing internal batch names `firstsyn` or `secondsyn`. The wrapper reports a clear TODO instead of pretending a generic batch is supported.

Not included in v0:

- GUI rewrite
- calibration or rectification algorithm changes
- export performance optimization
- full Python package conversion
- automatic white balance
- automatic residual epipolar correction
- a formal end-to-end 16-bit fidelity release

The current fidelity path applies the LUT through 8-bit `bgr24` rawvideo and then expands the result into a 16-bit PNG container. It must not be described as a true end-to-end 16-bit processing path.

Legacy scripts under `Code/legacy/` are retained for historical reference but are not part of the official pipeline v0.

## Configuration

Start from:

```text
configs/batches/batch.example.yaml
```

The batch config records:

- raw and synchronized video roots
- sync manifest
- calibration JSON and reports
- rectification and ROI assets
- selected time segments
- output root and LUT
- 5x5 camera layout and `CAM_C3` reference camera
- nominal FPS and image size
- checkerboard parameters
- worker and tool settings

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

NVENC is detected during preflight. The v0 wrapper does not assume that every machine provides it and does not silently substitute a different encoder.

## Commands

Run preflight only:

```powershell
python scripts/run_pipeline.py --config configs/batches/batch.example.yaml --stage sync-clip --preflight-only
```

Print the resolved command without executing it:

```powershell
python scripts/run_pipeline.py --config configs/batches/batch.example.yaml --stage calibrate --scene 0001 --dry-run
```

Export one benchmark scene:

```powershell
python scripts/run_pipeline.py --config configs/batches/batch.example.yaml --stage export --profile benchmark --scene 0027
```

Check all benchmark scenes:

```powershell
python scripts/run_pipeline.py --config configs/batches/batch.example.yaml --stage export-qa --profile benchmark
```

The three previously hard-coded core scripts also accept direct overrides:

```powershell
python sync/Batch_Clip_Engine.py --config configs/batches/batch.example.yaml --scene 0001
python Code/calibration/Calib_Solver_V7_Locked.py --config configs/batches/batch.example.yaml
python Code/prepare/Rectify_Generator.py --config configs/batches/batch.example.yaml
```

Calling those scripts without arguments preserves their existing local defaults.

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

Existing outputs generate warnings. The underlying production scripts retain their existing overwrite/skip behavior, so operators must review the resolved command and warnings before execution.

## Data and Assets

Real data, calibration results, exports, and generated maps remain local and are not committed by default:

- `Calibration_Data/`
- `Output/`
- `Exports/`

The LUT license and redistribution permission must be checked before publishing the LUT with an open-source release.
