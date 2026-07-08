from copy import deepcopy
from pathlib import Path


# 项目根目录。所有默认路径都从这里展开，避免手写绝对路径时混淆批次。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_NAME = "secondsyn"
# BATCH_EXPORT_ROOT = PROJECT_ROOT / "Exports"
BATCH_EXPORT_ROOT = Path(r"E:\5x5_LFV\LFV_Exports")


BATCH_CONFIGS = {
    "secondsyn": {
        "paths": {
            "input_root": r"E:\5x5_LFV\2nd_synced_output",
            "sync_manifest": str(PROJECT_ROOT / "Calibration_Data" / "secondsyn" / "sync_manifest_secondsyn.json"),
            "rectify_dir": str(PROJECT_ROOT / "Output" / "Rectify_Maps" / "secondsyn"),
            "roi_metadata": str(PROJECT_ROOT / "Output" / "Rectify_Maps" / "secondsyn" / "release_roi_metadata.json"),
        },
        "runtime": {
            "time_key": "secondsyn",
        },
    },
    "firstsyn": {
        "paths": {
            "input_root": r"E:\5x5_LFV\1st_synced_output",
            "sync_manifest": str(PROJECT_ROOT / "Calibration_Data" / "firstsyn" / "sync_manifest_firstsyn.json"),
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
        "python_executable": r"D:\anaconda3\envs\pytorch1\python.exe",
        "ffmpeg_executable": "ffmpeg",
    },
    "paths": {
        # time.json 仍然是两批次共用的片段清单。
        "time_json": str(PROJECT_ROOT / "Code" / "time.json"),
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
        "max_workers": 4,
        "ffmpeg_lut_hwaccel": "cuda",
        "shutdown_timeout_sec": 5,
        # ROI 生成与验证默认值。
        "roi_generation_margin_px": 4,
        "roi_validation_sample_scenes": ["0027", "0028", "0030"],
        "roi_validation_frame_positions": ["start", "middle", "end"],
        "roi_black_border_threshold": 0.01,
        "roi_near_black_threshold": 0.03,
        "roi_validation_fail_on_threshold_exceed": True,
        # 导出后检查默认先抽 3 帧；可疑场景自动升级到 9 帧。
        "export_validation_default_samples": 3,
        "export_validation_upgraded_samples": 9,
        "export_validation_black_border_fail_threshold": 0.01,
        "export_validation_near_black_warning_threshold": 0.03,
        "export_validation_enable_auto_upgrade": True,
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

    batch_output_root = BATCH_EXPORT_ROOT / selected_batch
    config["paths"]["output_root"] = str(batch_output_root)
    config["paths"]["benchmark_output_root"] = str(batch_output_root / "benchmark")
    config["paths"]["fidelity_output_root"] = str(batch_output_root / "fidelity")
    return config
