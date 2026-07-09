from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_NAME = "sample_batch"
BATCH_EXPORT_ROOT = PROJECT_ROOT / "Exports"

BATCH_CONFIGS = {
    "sample_batch": {
        "paths": {
            "input_root": str(PROJECT_ROOT / "data" / "synced" / "sample_batch"),
            "sync_manifest": str(PROJECT_ROOT / "metadata" / "sample_batch" / "sync_manifest.json"),
            "rectify_dir": str(PROJECT_ROOT / "metadata" / "sample_batch" / "rectification"),
            "roi_metadata": str(PROJECT_ROOT / "metadata" / "sample_batch" / "rectification" / "release_roi_metadata.json"),
        },
        "runtime": {
            "time_key": "sample_batch",
        },
    },
}

DEFAULT_CONFIG = {
    "environment": {
        "python_executable": "python",
        "ffmpeg_executable": "ffmpeg",
    },
    "paths": {
        "time_json": str(PROJECT_ROOT / "configs" / "time.example.json"),
        "lut_path": str(PROJECT_ROOT / "assets" / "user_provided_lut.cube"),
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
        "batch_id": DEFAULT_BATCH_NAME,
        "default_scene": None,
        "reference_camera": "CAM_C3",
        "max_workers": 4,
        "ffmpeg_lut_hwaccel": "cuda",
        "shutdown_timeout_sec": 5,
        "roi_generation_margin_px": 4,
        "roi_validation_sample_scenes": ["0001"],
        "roi_validation_frame_positions": ["start", "middle", "end"],
        "roi_black_border_threshold": 0.01,
        "roi_near_black_threshold": 0.03,
        "roi_validation_fail_on_threshold_exceed": True,
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
        raise ValueError(
            f"unsupported batch_name: {selected_batch}. "
            "Use --config with a local YAML file for real datasets."
        )

    batch_config = deepcopy(BATCH_CONFIGS[selected_batch])
    config["runtime"]["batch_name"] = selected_batch
    config["runtime"]["batch_id"] = selected_batch
    config["paths"].update(batch_config.get("paths", {}))
    config["runtime"].update(batch_config.get("runtime", {}))

    batch_output_root = BATCH_EXPORT_ROOT / selected_batch
    config["paths"]["output_root"] = str(batch_output_root)
    config["paths"]["benchmark_output_root"] = str(batch_output_root / "benchmark")
    config["paths"]["fidelity_output_root"] = str(batch_output_root / "fidelity")
    return config
