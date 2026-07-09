# RealDynLFV Pipeline v0 内部交接文档

## 1. 文档用途

RealDynLFV 是一个基于 5x5 多相机阵列采集动态光场视频的数据处理流程，目标是从原始多视角视频中生成结构统一、可用于评测和研究的 benchmark 数据。

这份文档给后续接手 RealDynLFV 采集和后处理的人使用。默认读者已经新采集了一批 5x5 多相机视频，需要从 raw videos 开始，生成 benchmark frames 和 QA 结果。

当前 pipeline v0 采用分 stage 运行。每一步都可以单独检查、单独重跑。第一次处理新数据时，不建议直接全量导出，应先用一个短 scene 做验证。

## 2. 完整处理链路

```text
raw videos
-> sync manifest
-> synchronized MP4
-> calibration JSON
-> rectification maps
-> release ROI
-> benchmark frames
-> export QA
```

各阶段含义：

- `raw videos`：25 个相机的原始视频。
- `sync manifest`：根据音频打板同步得到的每个相机 offset。
- `synchronized MP4`：按 offset 裁切后的同步视频。
- `calibration JSON`：棋盘格标定得到的内参、外参和相对位姿。
- `rectification maps`：由 calibration 生成的几何校正映射。
- `release ROI`：25 个视角共同有效的裁剪区域和最终 16:9 release crop。
- `benchmark frames`：最终公开评测用的 5x5 图像序列。
- `export QA`：检查帧数、视角、分辨率、metadata、黑边风险等。

## 3. 开始前准备

### 3.1 环境准备

建议使用独立 Python 环境。公开交接时不要依赖固定机器路径，接手者只需要保证下面这些工具可用：

- Python 3.x。
- FFmpeg 和 ffprobe，并且能在终端直接调用。
- OpenCV，也就是 Python 包 `cv2`。
- NumPy。
- SciPy。
- Matplotlib。
- sounddevice，用于同步 GUI 中的音频播放。
- tqdm。
- PyYAML，用于读取 batch YAML 配置。
- Pillow，用于部分 QA 图像检查。

最小安装示例：

```powershell
pip install opencv-python numpy scipy matplotlib sounddevice tqdm PyYAML Pillow
```

FFmpeg 需要单独安装，并加入系统 `PATH`。检查方式：

```powershell
ffmpeg -version
ffprobe -version
```

如果要运行 `sync-clip`，当前实现默认使用 FFmpeg/NVENC 路径。机器没有 NVIDIA GPU 或没有 NVENC 支持时，需要先调整配置或脚本参数，不要直接全量跑。

### 3.2 数据准备

新采集一批数据后，先整理好这些内容：

- 25 个 camera 目录，命名为 `CAM_A1` 到 `CAM_E5`。
- 每个 camera 目录里有对应 scene 的 raw videos。
- 标定视频也要包含 25 个视角。
- 音频打板同步所需的视频片段。
- 棋盘格参数，例如 inner corners 和 square size。
- time segments JSON，记录最终要导出的精选时间片段。
- LUT 路径，用于颜色标准化。
- 输出目录，用于保存 synchronized MP4、标定资产、导出帧和 QA 报告。

缺 25 views 的 scene 不适合作为正式 5x5 benchmark。可以保留作内部参考，但不要混进正式 benchmark 输出。

### 3.3 视频命名规范

新采集 batch 必须使用统一编号命名。规定从 `0001.mp4` 开始连续编号：

```text
0001.mp4
0002.mp4
0003.mp4
...
```

25 个 camera 目录下，同一个 scene 的文件名必须一致。例如：

```text
CAM_A1/0001.mp4
CAM_A2/0001.mp4
...
CAM_E5/0001.mp4
```

不要在同一个 batch 里混用 `take_001.mp4`、`scene1.mp4`、`0001.MP4` 这类不同风格。虽然部分脚本可以兼容 `.mp4` 和 `.MP4`，但交接规范统一使用小写 `.mp4`。

同步 manifest 和 time segments 也要跟这个编号一致：

```json
{
  "sample_batch": {
    "0001": [[0.0, 1.0]],
    "0002": [[2.5, 4.0]]
  }
}
```

这里的 `0001` 是 `0001.mp4` 去掉扩展名后的 scene id。

### 3.4 配置准备

复制一份本地 batch 配置：

```powershell
Copy-Item configs\batches\batch.example.yaml configs\batches\<batch>.local.yaml
```

然后编辑：

```text
configs/batches/<batch>.local.yaml
```

`.local.yaml` 里写真实路径，但不要提交。它已经被 `.gitignore` 忽略。

最关键字段如下：

```yaml
paths:
  raw_root:
  synced_root:
  sync_manifest:
  calibration_json:
  rectification_dir:
  roi_metadata:
  time_segments:
  output_root:
  lut:

camera:
  reference: CAM_C3
  nominal_fps:
  image_size:

calibration:
  target_video:
  checkerboard_inner_corners:
  square_size_mm:
```

注意：

- `camera.reference` 固定使用 `CAM_C3`。
- `paths.lut` 指向本地 LUT 文件。仓库不分发 DJI `.cube` LUT。
- 如果使用 DJI 官方 LUT，需要自行确认许可和下载来源。
- `paths.time_segments` 指向自己的 JSON。可以从 `configs/time.example.json` 复制一份再改。
- `calibration.target_video` 是同步后标定视频文件名，例如 `0001.mp4`。pipeline 不会自动帮你选择最佳标定视频。

## 4. 第一次跑新 batch

原则：每个 stage 先跑 `--preflight-only`，再跑真实命令。第一次处理新数据时，不要直接全量 export，先选一个短 scene 验证。

下面命令都使用 generic `<batch>.local.yaml`，实际使用时替换成自己的本地配置文件名。

### 4.1 检查同步结果

```powershell
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage sync-qa --preflight-only
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage sync-qa
```

主要看：

- 25 个 camera 是否都有 offset。
- `CAM_C3` 的 offset 是否为 `0.0`。
- 是否有明显离群 offset。

如果 offset 异常，先回到同步步骤复查音频，不要继续裁切。

### 4.2 生成同步视频

```powershell
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage sync-clip --preflight-only
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage sync-clip
```

输出是 synchronized MP4。常见失败原因：

- raw video 缺失。
- manifest 里的 scene/video 名和文件名对不上。
- FFmpeg 或 ffprobe 不可用。
- 当前机器没有可用 NVENC，而配置要求使用 NVENC。

### 4.3 标定

```powershell
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage calibrate --preflight-only
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage calibrate
```

`calibration.target_video` 指定同步后的标定视频，例如 `0001.mp4`。标定视频需要在 25 个 camera 目录下都存在。

pipeline 不会自动选择最佳标定视频。拍摄时应提前确认棋盘格清晰、覆盖范围足够、25 个视角都能检测到角点。

### 4.4 标定检查

```powershell
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage calibration-qa --preflight-only
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage calibration-qa
```

主要看：

- 每个 camera 的 checkerboard 检测率。
- reprojection error。
- stereo / relative pose 相关误差。
- `R_rel` 方向诊断。
- debug frames 中角点检测是否明显错误。

QA 报告只做诊断，不会自动修改 calibration JSON。

### 4.5 生成 rectification maps

```powershell
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage rectify --preflight-only
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage rectify
```

输出包括：

- `rectify_meta.json`
- 每个 camera 的 `.npz` rectification map

这些 maps 是正式资产，后续 export 会逐帧读取并执行 remap。

### 4.6 生成 release ROI

```powershell
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage roi --preflight-only
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage roi
```

输出是 `release_roi_metadata.json`。

ROI 是正式资产，不建议 export 时临时重算。更新 rectification maps 或 ROI policy 后，应重新生成 ROI，并重新做 ROI QA。

### 4.7 检查 ROI

```powershell
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage roi-qa --preflight-only
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage roi-qa
```

主要看：

- 是否仍有明显黑边。
- 是否有无效区域进入最终 crop。
- 16:9 crop 是否合法。
- ROI metadata 是否和 rectification assets 匹配。

### 4.8 导出一个 scene

```powershell
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage export --profile benchmark --scene 0001 --preflight-only
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage export --profile benchmark --scene 0001 --dry-run
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage export --profile benchmark --scene 0001
```

先导出一个 scene，确认输出正常后再全量导出。检查重点：

- 是否有 25 个 view。
- 每个 view 帧数是否一致。
- 输出分辨率是否符合 benchmark 配置。
- metadata 是否写出。
- 颜色是否正常。

### 4.9 检查导出结果

```powershell
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage export-qa --profile benchmark --scene 0001 --preflight-only
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage export-qa --profile benchmark --scene 0001
```

QA 会检查：

- view 数。
- frame 数。
- resolution。
- metadata。
- `.in_progress` 残留。
- 黑边风险。
- 辅助检查图。

自然场景 LK residual 只作参考，不要把它直接当成严格几何误差。

### 4.10 全量导出

确认单 scene 没问题后，再跑：

```powershell
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage export --profile benchmark
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage export-qa --profile benchmark
```

全量导出前建议确认输出目录是否已经存在。如果目录里有旧结果，需要先决定是保留、删除还是另换输出目录。

## 5. 输出结果

典型输出结构如下：

```text
output_root/
  <batch>/
    benchmark/
      <scene>/
        metadata.json
        view_00/
        view_01/
        ...
    checks/
      <scene>/
        check_report.json
```

说明：

- `view_XX/` 中是导出的 frame。
- `metadata.json` 是 scene-level metadata。
- `view_metadata.json` 是 view-level metadata，位于每个 view 目录中。
- metadata 会记录 crop、phase offset、asset version、LUT 信息、输出格式等。
- `checks/<scene>/check_report.json` 是 export QA 报告。

## 6. 只从中间阶段继续

如果已经有部分资产，不需要每次都从 raw videos 重跑。

常见情况：

- 已有 synchronized MP4：可以从 `calibrate` 开始。
- 已有 calibration JSON：可以从 `rectify` 开始。
- 已有 rectification maps 和 ROI：可以直接从 `export` 开始。

前提是这些资产来自同一个 batch，不能混用。尤其是 calibration JSON、rectification maps、ROI metadata 和 synchronized MP4 必须对应同一批相机状态。

## 7. Stage 一览

| Stage | 用途 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `sync-qa` | 检查同步 manifest | sync manifest | offset 检查结果 |
| `sync-clip` | 裁切同步 MP4 | raw videos, sync manifest | synchronized MP4 |
| `calibrate` | 相机标定 | synchronized calibration video | calibration JSON |
| `calibration-qa` | 标定质量检查 | calibration JSON, synced MP4 | quality report, debug frames |
| `rectify` | 生成几何校正 maps | calibration JSON | rectification maps |
| `roi` | 生成发布 ROI | rectification maps | release ROI metadata |
| `roi-qa` | 检查 ROI | ROI, synced MP4, time segments | ROI QA report |
| `export` | 导出 benchmark | synced MP4, ROI, maps, LUT, time segments | benchmark frames, metadata |
| `export-qa` | 检查导出结果 | benchmark frames | QA reports, check visuals |

## 8. 常见问题

### preflight 报缺路径

先检查 `<batch>.local.yaml`。大多数情况是路径没改完整，或文件移动后配置没同步。

### 缺 25 views

正式 5x5 benchmark 需要 25 个视角。缺 view 的 scene 不建议进入正式发布。

### 视频编号不一致

新 batch 统一使用从 `0001.mp4` 开始的连续编号。manifest、time segments 和 25 个 camera 目录下的文件名必须一致。

### LUT 缺失

export 会失败。确认 `paths.lut` 指向本地存在的 LUT 文件。

### time segments 没有 scene

export 会跳过该 scene。确认 `paths.time_segments` 指向正确 JSON，并且 JSON 中包含对应 scene id。

### 已有输出目录 warning

这只是提醒，不一定是错误。继续运行前要确认是否会覆盖、跳过或混入旧结果。

### 中断后残留 `.in_progress`

说明对应 view 或 scene 可能是半成品。清理该 scene 输出目录后再重跑。

### calibration target video 不存在

检查 `calibration.target_video` 是否写对，以及 synchronized MP4 根目录下 25 个 camera 是否都有这个视频。

### export QA 出现黑边 warning

先看 check report 和检查图。暗内容、黑衣服、遮挡等可能触发误报；真正的几何黑边通常会在固定边缘位置持续出现。

### 自然场景 LK residual 很差

该指标只作辅助参考。自然场景跨视角匹配会受遮挡、视差、运动和纹理影响，不能直接解释成严格极线误差。

## 9. 当前边界

当前 pipeline v0 的边界：

- 不自动选择标定视频。
- 不自动判断拍摄质量。
- 不提供 `--stage all`。
- fidelity 暂不作为 true 16-bit release。
- legacy scripts 不属于正式流程。
