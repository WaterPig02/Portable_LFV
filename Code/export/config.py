from copy import deepcopy
from pathlib import Path


# 项目根目录。后续所有默认路径都从这里展开，避免在代码里散落绝对路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 默认处理第二批次。切到第一批次时只改 batch_name，不手工改一堆路径。
DEFAULT_BATCH_NAME = "secondsyn"
# 所有导出结果统一放在 Exports/<batch>/ 下。
BATCH_EXPORT_ROOT = PROJECT_ROOT / "Exports"

BATCH_CONFIGS = {
    "secondsyn": {
        "paths": {
            "input_root": r"E:\5x5_LFV\2nd_synced_output",
            "sync_manifest": str(PROJECT_ROOT / "Calibration_Data" / "sync_manifest_secondsyn.json"),
            "rectify_dir": str(PROJECT_ROOT / "Output" / "Rectify_Maps" / "secondsyn"),
            "roi_metadata": str(PROJECT_ROOT / "Output" / "Rectify_Maps" / "secondsyn" / "release_roi_metadata.json"),
        },
        "runtime": {
            "time_key": "secondsyn",
        },
    },
    "firstsyn": {
        "paths": {
            "input_root": None,
            "sync_manifest": str(PROJECT_ROOT / "Calibration_Data" / "sync_manifest_firstsyn.json"),
            "rectify_dir": str(PROJECT_ROOT / "Output" / "Rectify_Maps" / "firstsyn"),
            "roi_metadata": str(PROJECT_ROOT / "Output" / "Rectify_Maps" / "firstsyn" / "release_roi_metadata.json"),
        },
        "runtime": {
            "time_key": "firstsyn",
        },
    },
}

DEFAULT_CONFIG = {
    "environment": {
        # 默认生产环境解释器。
        "python_executable": r"D:\anaconda3\envs\pytorch1\python.exe",
        # LUT 处理当前仍依赖 ffmpeg。
        "ffmpeg_executable": "ffmpeg",
    },
    "paths": {
        # 片段筛选文件，firstsyn/secondsyn 共用一个 time.json。
        "time_json": str(PROJECT_ROOT / "Code" / "time.json"),
        # 默认 LUT 资产路径。命令行传入时可覆盖。
        "lut_path": str(PROJECT_ROOT / "DJI OSMO Action 5 Pro D-Log M to Rec.709 V1.cube"),
    },
    "profiles": {
        "benchmark": {
            "jpeg_quality": 97,
            "resize_target": [1920, 1080],
            "bit_depth": 8,
        },
        "fidelity": {
            "png_bit_depth": 16,
            "no_resize": True,
            "bit_depth": 16,
        },
    },
    "runtime": {
        "batch_name": DEFAULT_BATCH_NAME,
        "default_scene": None,
        "reference_camera": "CAM_C3",
        # camera 级并行默认 worker 数。
        "max_workers": 4,
        # ffmpeg LUT 命令默认带 CUDA 标志；传参可改成 none。
        "ffmpeg_lut_hwaccel": "cuda",
        # Ctrl+C 后等待 worker 自行退出的秒数，超时再强杀。
        "shutdown_timeout_sec": 5,
    },
}


def get_default_config(batch_name=None):
    config = deepcopy(DEFAULT_CONFIG)
    selected_batch = batch_name or config["runtime"]["batch_name"]
    if selected_batch not in BATCH_CONFIGS:
        raise ValueError(f"unsupported batch_name: {selected_batch}")
    batch_config = deepcopy(BATCH_CONFIGS[selected_batch])
    config["runtime"]["batch_name"] = selected_batch
    config["paths"].update(batch_config.get("paths", {}))
    config["runtime"].update(batch_config.get("runtime", {}))
    # 输出目录按批次隔离，避免 firstsyn / secondsyn 互相覆盖。
    batch_output_root = BATCH_EXPORT_ROOT / selected_batch
    config["paths"]["output_root"] = str(batch_output_root)
    config["paths"]["benchmark_output_root"] = str(batch_output_root / "benchmark")
    config["paths"]["fidelity_output_root"] = str(batch_output_root / "fidelity")
    return config
