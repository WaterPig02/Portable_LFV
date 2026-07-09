# RealDynLFV 后处理 Pipeline

RealDynLFV 提供一套面向 5x5 动态光场视频的分阶段后处理流程。它把多相机原始视频处理为同步视频、标定资产、几何校正映射、发布裁剪区域、benchmark 图像帧和 QA 报告。

public v0 的目标是让数据构建过程可复现、可检查、可分步重跑。每个 stage 都通过本地 YAML 配置驱动。

## ✨ 能做什么

- 检查音频同步 manifest。
- 根据同步 offset 裁切 synchronized MP4。
- 运行棋盘格相机标定。
- 生成 rectification maps 和 release ROI metadata。
- 导出 `8-bit JPEG + LUT + 1920x1080` benchmark frames。
- 检查导出结果的视角数、帧数、分辨率、metadata 和黑边风险。

## 🚀 快速开始

### 1. 安装依赖

```powershell
pip install opencv-python numpy scipy matplotlib sounddevice tqdm PyYAML Pillow
```

FFmpeg 需要单独安装，并确保终端里可以直接调用：

```powershell
ffmpeg -version
ffprobe -version
```

当前 synchronized clipping stage 使用 FFmpeg/NVENC。如果机器没有 NVENC，先看 `--dry-run` 输出，不要直接跑全量裁切。

### 2. 创建本地 batch 配置

```powershell
Copy-Item configs\batches\batch.example.yaml configs\batches\my_batch.local.yaml
```

在 `configs/batches/my_batch.local.yaml` 中填写自己的真实路径。本地配置已被 Git 忽略。

### 3. 分 stage 运行

先做 preflight：

```powershell
python scripts/run_pipeline.py --config configs\batches\my_batch.local.yaml --stage export --profile benchmark --scene 0001 --preflight-only
```

打印将要执行的命令：

```powershell
python scripts/run_pipeline.py --config configs\batches\my_batch.local.yaml --stage export --profile benchmark --scene 0001 --dry-run
```

导出一个 benchmark scene：

```powershell
python scripts/run_pipeline.py --config configs\batches\my_batch.local.yaml --stage export --profile benchmark --scene 0001
```

检查这个 scene：

```powershell
python scripts/run_pipeline.py --config configs\batches\my_batch.local.yaml --stage export-qa --profile benchmark --scene 0001
```

确认单个 scene 没问题后，再全量导出：

```powershell
python scripts/run_pipeline.py --config configs\batches\my_batch.local.yaml --stage export --profile benchmark
python scripts/run_pipeline.py --config configs\batches\my_batch.local.yaml --stage export-qa --profile benchmark
```

## 🧭 处理链路

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

benchmark 导出的图像处理顺序固定为：

```text
decode -> rectification -> common ROI -> fixed 16:9 crop
       -> benchmark resize -> LUT -> 8-bit JPEG export
```

## 🧩 Stages

使用 `scripts/run_pipeline.py --stage <name>` 运行指定 stage。

| Stage | 用途 | 主要输出 |
| --- | --- | --- |
| `sync-qa` | 检查 sync manifest offset | offset QA summary |
| `sync-clip` | 根据 sync manifest 裁切原始视频 | synchronized MP4 files |
| `calibrate` | 运行棋盘格标定 | calibration JSON |
| `calibration-qa` | 诊断标定质量 | QA report and debug frames |
| `rectify` | 生成几何校正映射 | `.npz` maps and `rectify_meta.json` |
| `roi` | 生成发布 ROI | `release_roi_metadata.json` |
| `roi-qa` | 用抽样帧检查 ROI | ROI validation report |
| `export` | 导出 benchmark frames | scene/view/frame folders and metadata |
| `export-qa` | 检查导出帧 | QA reports and check visuals |

Pipeline v0 保持分 stage 运行，不提供 `--stage all`。建议每一步检查通过后再进入下一步。

## 📁 输入目录

使用 25 个 camera 目录，命名从 `CAM_A1` 到 `CAM_E5`。

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

规则：

- 使用从 `0001.mp4` 开始的四位补零 scene 编号。
- 同一个 scene 在 25 个 camera 目录下必须使用相同文件名。
- sync manifest、time segment JSON 和 CLI `--scene` 使用同一个 scene ID。
- public v0 统一推荐小写 `.mp4`。

缺少任一视角的 scene 不建议作为正式 5x5 benchmark scene。

## ⚙️ Batch 配置

从这个模板开始：

```text
configs/batches/batch.example.yaml
```

关键字段：

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

`paths.lut` 需要指向本地 LUT 文件。本仓库不再分发 DJI `.cube` LUT。如果使用 DJI 官方 LUT，需要自行确认许可和下载来源。

## ✅ Preflight 检查

每个 stage 运行前，wrapper 会检查：

- YAML 是否可读。
- 5x5 / 25-view camera layout。
- `CAM_C3` reference camera。
- 必需目录和 JSON 资产。
- calibration 和 rectification camera 覆盖情况。
- time segment 和 scene 是否存在。
- 输出路径是否已有结果。
- FFmpeg / ffprobe 是否可用。
- synchronized clipping 所需的 `hevc_nvenc` 支持。

已有输出会产生 warning。长任务运行前应先检查 warning 和 `--dry-run` 命令。

## 📦 输出结构

典型 benchmark 输出：

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

每个 scene 包含 sequence-level metadata 和每个 view 的帧目录。metadata 会记录 phase offset、ROI/crop policy、rectification version、LUT version、输出格式和分辨率。

## 📝 说明

- 当前稳定公开目标是 benchmark：`8-bit JPEG + LUT + 1920x1080`。
- 代码里存在 fidelity 相关路径，但 v0 不把它描述为正式 true 16-bit release。
- Legacy scripts 不属于 public pipeline v0。
- 真实数据、生成资产、导出结果、`.local.yaml` 配置和 LUT 文件不应进入 Git。
