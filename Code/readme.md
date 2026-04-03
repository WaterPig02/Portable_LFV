# 标定与导出工具说明

## 1. 概览
当前仓库的正式发布链以“同步后的 MP4”为输入，统一导出两类数据：

- `benchmark`
- `fidelity`

统一处理顺序固定为：

1. 读取同步后的 MP4
2. 几何校正 `rectify`
3. 应用 `common_valid_roi`
4. 应用 `final_release_crop_16_9`
5. `benchmark` 才 resize 到 `1920x1080`
6. `benchmark` 和 `fidelity` 都应用 LUT
7. 写出图像与 metadata

默认生产环境：

- Python：`D:\anaconda3\envs\pytorch1\python.exe`

## 2. 目录结构
```text
Code/
  calibration/
  prepare/
  export/
  legacy/
  readme.md
  time.json
```

目录职责：

- `calibration/`：标定求解
- `prepare/`：生成 rectification maps 与 ROI metadata
- `export/`：正式导出与检查工具
- `legacy/`：旧脚本与人工辅助工具

## 3. 正式生产资产
以下资产仍然是正式生产资产：

- `CAM_XX_rect_map.npz`
- `rectify_meta.json`
- `release_roi_metadata.json`

用途：

- `.npz`：逐帧提供 `map_x / map_y` 给 `cv2.remap`
- `rectify_meta.json`：提供 map 索引、图像尺寸、`rectification_asset_version`
- `release_roi_metadata.json`：提供 `common_valid_roi` 与 `final_release_crop_16_9`

## 4. 批次隔离
当前要求第一批次与第二批次的资产完全隔离。

### 第二批次 `secondsyn`
- `Calibration_Data/sync_manifest_secondsyn.json`
- `Output/calibration_raw_stereo_locked_secondsyn.json`
- `Output/Rectify_Maps/secondsyn/`

### 第一批次 `firstsyn`
- `Calibration_Data/sync_manifest_firstsyn.json`
- `Output/calibration_raw_stereo_locked_firstsyn.json`
- `Output/Rectify_Maps/firstsyn/`

`time.json` 保持共用，但以下五项必须始终指向同一批次：

- `input_root`
- `sync_manifest`
- `rectify_dir`
- `roi_metadata`
- `time_key`

## 5. 配置文件
主配置文件：

- `Code/export/config.py`

重要配置项：

### `environment`
- `python_executable`
- `ffmpeg_executable`

### `paths`
- `input_root`
- `sync_manifest`
- `time_json`
- `rectify_dir`
- `roi_metadata`
- `output_root`
- `benchmark_output_root`
- `fidelity_output_root`
- `lut_path`

### `profiles`
- `benchmark.jpeg_quality`
- `benchmark.resize_target`
- `benchmark.bit_depth`
- `fidelity.png_bit_depth`
- `fidelity.no_resize`
- `fidelity.bit_depth`

### `runtime`
- `batch_name`
- `default_scene`
- `reference_camera`
- `max_workers`
- `ffmpeg_lut_hwaccel`
- `shutdown_timeout_sec`
- `time_key`

参数优先级：

1. 命令行参数
2. `config.py` 默认值
3. 若关键项仍缺失则直接失败

## 6. 导出 profile 规则
### `benchmark`
- 输出格式：`JPEG`
- 文件扩展名：`.jpg`
- 位深：`8-bit`
- 分辨率：`1920x1080`
- 应用 LUT

### `fidelity`
- 输出格式：`PNG`
- 位深：`16-bit`
- 不 resize
- 保留裁剪后的原始分辨率
- 应用 LUT

## 7. 日志与终端进度
当前 exporter 会为每次运行生成一份中文日志。

日志目录：
- `Exports/<batch>/logs/`

日志内容包括：
- run 开始与结束时间
- batch / profile / scene / time_key
- 关键输入路径
- `max_workers` 与 LUT 配置
- scene / segment / camera 开始与完成信息
- 异常与 traceback
- 最终摘要

终端会同步输出：
- 当前场景/片段
- 已完成视角数 / 总视角数
- 每个视角完成时的帧数和分辨率
- 异常提示
- 中断清理信息

## 8. Ctrl+C 与安全退出
如果你在 VSCode 终端中运行导出，按 `Ctrl+C` 时当前代码会：

1. 停止继续提交新视角任务
2. 通知活跃 worker 尽快退出
3. 超时后终止 worker
4. 在 Windows 上通过 `taskkill /T /F` 清理 worker 进程树
5. ffmpeg 子进程会随 worker 一起被清理

中断后的状态：

- 未完成的 `view_xx/` 下会保留 `.in_progress`
- 中断的 scene 不会写 sequence `metadata.json`
- 建议清理该 scene 输出目录后再重跑

## 9. 检查工具
新增导出检查工具：

- `Code/export/validate_export.py`

它会输出：
- 中文检查日志
- `check_report.json`
- 人工检查图目录

自动检查内容：
- 黑边 / 无效区域
- 分辨率一致性
- 帧数一致性
- metadata 完整性
- `.in_progress` 残留
- 多视点极线垂直残差估计

人工检查图包括：
- 5x5 总拼图
- 同一行拼图
- 同一列拼图

### 同一行拼图
- A 行：`CAM_A1` 到 `CAM_A5`
- B 行：`CAM_B1` 到 `CAM_B5`
- C 行：`CAM_C1` 到 `CAM_C5`
- D 行：`CAM_D1` 到 `CAM_D5`
- E 行：`CAM_E1` 到 `CAM_E5`

### 同一列拼图
- 1 列：`CAM_A1`、`CAM_B1`、`CAM_C1`、`CAM_D1`、`CAM_E1`
- 2 列：`CAM_A2`、`CAM_B2`、`CAM_C2`、`CAM_D2`、`CAM_E2`
- 3 列：`CAM_A3`、`CAM_B3`、`CAM_C3`、`CAM_D3`、`CAM_E3`
- 4 列：`CAM_A4`、`CAM_B4`、`CAM_C4`、`CAM_D4`、`CAM_E4`
- 5 列：`CAM_A5`、`CAM_B5`、`CAM_C5`、`CAM_D5`、`CAM_E5`

检查图输出目录：
- `Exports/<batch>/checks/<scene>/`

## 10. 常用命令
### 生成 ROI metadata
```powershell
D:\anaconda3\envs\pytorch1\python.exe Code\prepare\generate_release_roi.py --batch-name secondsyn
```

### 跑 benchmark
```powershell
D:\anaconda3\envs\pytorch1\python.exe Code\export\export_dataset.py --batch-name secondsyn --profile benchmark
```

### 跑 fidelity
```powershell
D:\anaconda3\envs\pytorch1\python.exe Code\export\export_dataset.py --batch-name secondsyn --profile fidelity
```

### 单场景 benchmark 回归
```powershell
D:\anaconda3\envs\pytorch1\python.exe Code\export\export_dataset.py --batch-name secondsyn --profile benchmark --scene 0027 --max-workers 4
```

### 导出后检查
```powershell
D:\anaconda3\envs\pytorch1\python.exe Code\export\validate_export.py --batch-name secondsyn --profile benchmark --scene 0027
```

## 11. 发布前检查清单
至少检查以下内容：

- `time.json` 片段是否正确
- 25 视图是否齐全
- 帧数是否一致
- 分辨率是否一致
- `metadata.json` / `view_metadata.json` 是否完整
- `CAM_C3.phase_offset_ms = 0.0`
- LUT / ROI / rectification 版本信息是否齐全
- benchmark 颜色是否正常
- 无残留 `.in_progress`
- 5x5 总拼图是否正常
- 同一行拼图是否平滑
- 同一列拼图是否平滑
- 邻近视角与大视差视角是否存在明显极线漂移

## 12. 重跑说明
同一 scene 重跑时，会写回同一个输出目录。

这意味着：
- 同名图像会被覆盖
- 若上一次中断，可能残留旧图和 `.in_progress`

因此建议：
- 若某个 scene 上一次没完整跑完，重跑前先手动删除该 scene 对应输出目录
