# RealDynLFV Pipeline v0 内部交接说明

本文档面向后续接手 RealDynLFV 采集与后处理的人。它说明当前 pipeline v0 的能力边界、配置方式、分 stage 运行方法和常见注意事项。

## 1. Pipeline v0 当前能做什么

统一入口为：

```powershell
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage <stage>
```

当前支持以下 stage：

| Stage | 作用 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `sync-qa` | 检查 sync manifest 中异常 `offset_ms` | `sync_manifest` | 终端报告，可选 CSV |
| `sync-clip` | 根据 sync manifest 裁切同步后 MP4 | raw camera videos, sync manifest | synchronized MP4 clips + clip metadata |
| `calibrate` | 基于 checkerboard 视频做相机标定 | synchronized MP4 calibration video | calibration JSON, debug images |
| `calibration-qa` | 检查 calibration 质量与 `R_rel` 方向 | synchronized MP4, calibration JSON | calibration quality report |
| `rectify` | 生成 rectification maps | calibration JSON | `rectify_meta.json`, `CAM_XX_rect_map.npz` |
| `roi` | 生成 release ROI | rectification maps | `release_roi_metadata.json` |
| `roi-qa` | 抽样验证 release ROI | synced MP4, ROI metadata, time segments | ROI validation report |
| `export` | 导出 benchmark frames | synced MP4, time segments, maps, ROI, LUT | benchmark frame folders + metadata |
| `export-qa` | 检查导出结果 | exported benchmark frames | scene reports, batch summary, check visuals |

Benchmark export 的图像处理顺序固定为：

```text
decode -> rectification -> common ROI -> fixed 16:9 crop -> benchmark resize -> LUT -> 8-bit JPEG export
```

## 2. Pipeline v0 当前不能做什么

当前 v0 是可交接的后处理入口，但不是完整自动化系统：

- 不自动选择标定视频。
- 不自动判断拍摄质量或决定某个 scene 是否适合发布。
- 不自动从 raw videos 一键跑到最终 benchmark；需要逐 stage 检查和运行。
- 暂不提供 `--stage all`。
- fidelity 路径存在，但暂不作为正式 true 16-bit release。
- legacy scripts 不属于正式流程，只作为内部参考保留在 ignored 本地目录。
- 不自动修正残余极线误差、相机抖动或拍摄过程中的几何漂移。
- 不做自动白平衡；颜色标准化依赖用户配置的固定 LUT。

## 3. 本地配置准备

公开仓库只提供 generic 示例配置：

```text
configs/batches/batch.example.yaml
```

新 batch 的本地配置准备方式：

```powershell
Copy-Item configs\batches\batch.example.yaml configs\batches\<batch>.local.yaml
```

然后编辑：

```text
configs/batches/<batch>.local.yaml
```

必须注意：

- `.local.yaml` 不提交，里面可以写真实数据路径。
- LUT 需要本地自备，仓库不分发 DJI `.cube` LUT。
- 如果使用 DJI 官方 LUT，需要自行确认许可和下载来源。
- `paths.time_segments` 指向自己的 time segment JSON。
- time segment JSON 可从 `configs/time.example.json` 复制后修改。
- `camera.reference` 固定使用 `CAM_C3`。
- 5x5 layout 必须包含 25 个 views。

## 4. 分 Stage 运行方式

### 4.1 Preflight

只检查配置和关键资产，不执行 stage：

```powershell
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage export --profile benchmark --scene 0001 --preflight-only
```

建议每个 stage 正式运行前先跑一次 `--preflight-only`。

### 4.2 Dry Run

打印最终解析出的命令，不执行真实处理：

```powershell
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage export --profile benchmark --scene 0001 --dry-run
```

用于确认路径、profile、scene 和输入资产是否符合预期。

### 4.3 Export

导出单个 benchmark scene：

```powershell
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage export --profile benchmark --scene 0001
```

导出全部 benchmark scenes：

```powershell
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage export --profile benchmark
```

### 4.4 Export QA

检查单个 scene：

```powershell
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage export-qa --profile benchmark --scene 0001
```

检查整批 benchmark：

```powershell
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage export-qa --profile benchmark
```

## 5. 推荐运行顺序

新 batch 推荐按以下顺序逐步运行，不建议跳过 QA：

```powershell
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage sync-qa --preflight-only
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage sync-qa

python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage sync-clip --preflight-only
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage sync-clip

python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage calibrate --preflight-only
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage calibrate

python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage calibration-qa --preflight-only
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage calibration-qa

python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage rectify --preflight-only
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage rectify

python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage roi --preflight-only
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage roi

python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage roi-qa --preflight-only
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage roi-qa

python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage export --profile benchmark --preflight-only
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage export --profile benchmark

python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage export-qa --profile benchmark --preflight-only
python scripts/run_pipeline.py --config configs/batches/<batch>.local.yaml --stage export-qa --profile benchmark
```

## 6. 已有内部数据如何跑

公开文档不把历史内部 batch 名作为概念使用。内部历史数据仍可通过本地 `.local.yaml` 跑：

- 在 `.local.yaml` 中写入历史数据的 raw/synced/calibration/rectification/ROI/export 路径。
- 不要把 `.local.yaml` 提交到 Git。
- 不要把真实 LUT、真实 time segments、真实数据路径提交到 Git。
- 如果历史数据已有 synchronized MP4，可以从 `calibrate`、`rectify`、`roi` 或 `export` 等中间 stage 开始。
- 如果历史数据已有 calibration/rectification/ROI 资产，必须确认它们来自同一 batch，不能混用。

## 7. 常见注意事项

- 已有输出目录会触发 warning；这不一定是错误，但要确认是否会覆盖或跳过已有结果。
- 中断后的半成品输出需要人工清理后再重跑。
- 视角不齐、缺少 25 views，preflight 或正式处理会失败。
- LUT 缺失会导致 export 失败。
- 标定视频必须在 synchronized MP4 根目录下存在，并且 25 个相机都应可读。
- ROI metadata 必须和 rectification maps 属于同一 batch。
- `CAM_C3` 是 reference camera，phase offsets 都相对于 `CAM_C3`。
- `CAM_C3.phase_offset_ms` 应为 `0.0`。
- export QA 中的自然场景极线指标只是辅助探索指标，不应直接解释为严格几何误差。
- 之前做过新旧入口 export 回归：同一小 scene 下，旧入口和 YAML 入口导出的 JPEG SHA-256 一致，metadata 在忽略运行时字段后等价。

## 8. 交接时建议检查

接手者拿到项目后，先做以下最小验证：

```powershell
python -m py_compile scripts\pipeline_config.py scripts\run_pipeline.py
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage export --profile benchmark --scene 0001 --preflight-only
python scripts\run_pipeline.py --config configs\batches\<batch>.local.yaml --stage export --profile benchmark --scene 0001 --dry-run
```

如果以上通过，再逐 stage 执行真实处理。
