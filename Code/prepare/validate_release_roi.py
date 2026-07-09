import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

EXPORT_DIR = Path(__file__).resolve().parents[1] / "export"
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from asset_versioning import ROI_METADATA_FILENAME, compute_export_signature, normalize_rectify_metadata, normalize_roi_metadata
from config import get_default_config
from pipeline_config import batch_id, config_value, load_pipeline_config, time_key


CAMERAS = [f"CAM_{row}{col}" for row in "ABCDE" for col in range(1, 6)]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class ValidationLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("w", encoding="utf-8")

    def log(self, level, message):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def info(self, message):
        self.log("信息", message)

    def warn(self, message):
        self.log("警告", message)

    def error(self, message):
        self.log("错误", message)

    def close(self):
        self.handle.close()


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_rectification_assets(rectify_dir):
    rectify_dir = Path(rectify_dir)
    rectify_meta_path = rectify_dir / "rectify_meta.json"
    ensure(rectify_meta_path.exists(), f"缺少 rectification metadata：{rectify_meta_path}")
    rectification_meta = normalize_rectify_metadata(load_json(rectify_meta_path))
    if rectification_meta["image_size"] is None:
        first_camera_meta = next(iter(rectification_meta["cameras"].values()), None)
        ensure(first_camera_meta is not None, "rectification metadata 内没有 camera 信息")
        map_path = rectify_dir / first_camera_meta["map_file"]
        ensure(map_path.exists(), f"缺少 rectification map：{map_path}")
        data = np.load(map_path)
        rectification_meta["image_size"] = [int(data["map_x"].shape[1]), int(data["map_x"].shape[0])]
        rectification_meta["rectification_asset_version"] = compute_export_signature(
            {
                "schema_version": rectification_meta["schema_version"],
                "master_camera": rectification_meta["master_camera"],
                "image_size": rectification_meta["image_size"],
                "crop_alpha": rectification_meta["crop_alpha"],
                "cameras": rectification_meta["cameras"],
            }
        )
    return rectify_dir, rectification_meta


def load_roi_metadata(rectify_dir, roi_metadata_path=None):
    roi_path = Path(roi_metadata_path) if roi_metadata_path else Path(rectify_dir) / ROI_METADATA_FILENAME
    ensure(roi_path.exists(), f"缺少 ROI metadata：{roi_path}")
    return roi_path, normalize_roi_metadata(load_json(roi_path))


def load_time_config(path, time_key):
    raw = load_json(path)
    ensure(time_key in raw, f"time.json 缺少 key：{time_key}")
    selected = raw[time_key]
    result = {}
    for scene_id, value in selected.items():
        if value == "del":
            continue
        if isinstance(value[0], list):
            segments = value
        else:
            segments = [value]
        result[scene_id] = [(float(segment[0]), float(segment[1])) for segment in segments]
    return result


def scene_to_video_map(sync_manifest):
    mapping = {}
    for video_name in sync_manifest.keys():
        mapping[Path(video_name).stem] = video_name
    return mapping


def resolve_scene_ids(args_scenes, config_runtime, available_segments):
    if args_scenes:
        scenes = args_scenes
    else:
        scenes = list(config_runtime.get("roi_validation_sample_scenes", []))
    resolved = [scene_id for scene_id in scenes if scene_id in available_segments]
    ensure(resolved, "没有可用于 ROI 验证的场景")
    return resolved


def compute_border_stats(image, black_threshold=2, near_black_threshold=8):
    edges = {
        "top": image[0],
        "bottom": image[-1],
        "left": image[:, 0],
        "right": image[:, -1],
    }
    per_edge = {}
    black_ratios = []
    near_black_ratios = []
    for edge_name, edge_pixels in edges.items():
        black_mask = np.all(edge_pixels <= black_threshold, axis=1)
        near_black_mask = np.all(edge_pixels <= near_black_threshold, axis=1)
        black_ratio = float(np.mean(black_mask))
        near_black_ratio = float(np.mean(near_black_mask))
        per_edge[edge_name] = {
            "black_ratio": black_ratio,
            "near_black_ratio": near_black_ratio,
        }
        black_ratios.append(black_ratio)
        near_black_ratios.append(near_black_ratio)
    return {
        "overall_black_ratio": float(np.mean(black_ratios)),
        "overall_near_black_ratio": float(np.mean(near_black_ratios)),
        "per_edge": per_edge,
    }


def crop_frame(frame, roi):
    x = int(roi["x"])
    y = int(roi["y"])
    w = int(roi["width"])
    h = int(roi["height"])
    cropped = frame[y : y + h, x : x + w]
    ensure(cropped.size != 0, "ROI 裁剪结果为空")
    return cropped


def sample_times_for_segment(start_time, end_time, positions):
    length = end_time - start_time
    mapping = {
        "start": start_time,
        "middle": start_time + length * 0.5,
        "end": max(start_time, end_time - min(0.05, length * 0.01)),
    }
    return [(position, float(mapping[position])) for position in positions]


def validate_static_geometry(roi_metadata):
    common_roi = roi_metadata["common_valid_roi"]
    safe_rect = roi_metadata["safe_rect_roi"]
    final_crop = roi_metadata["final_release_crop_16_9"]
    return {
        "safe_rect_inside_common_roi": (
            safe_rect["x"] >= common_roi["x"]
            and safe_rect["y"] >= common_roi["y"]
            and safe_rect["x"] + safe_rect["width"] <= common_roi["x"] + common_roi["width"]
            and safe_rect["y"] + safe_rect["height"] <= common_roi["y"] + common_roi["height"]
        ),
        "final_crop_inside_safe_rect": (
            final_crop["x"] >= safe_rect["x"]
            and final_crop["y"] >= safe_rect["y"]
            and final_crop["x"] + final_crop["width"] <= safe_rect["x"] + safe_rect["width"]
            and final_crop["y"] + final_crop["height"] <= safe_rect["y"] + safe_rect["height"]
        ),
        "final_crop_exact_16_9": final_crop["width"] * 9 == final_crop["height"] * 16,
        "validity_margin_px": roi_metadata.get("validity_margin_px"),
        "common_valid_mask_bbox": roi_metadata.get("common_valid_mask_bbox"),
        "safe_rect_roi": safe_rect,
        "final_release_crop_16_9": final_crop,
        "camera_crop_margin_summary": roi_metadata.get("roi_validation_summary", {}).get("camera_crop_margin_summary", {}),
        "global_min_crop_margin_px": roi_metadata.get("roi_validation_summary", {}).get("global_min_crop_margin_px"),
    }


def validate_sampled_frames(
    rectify_dir,
    rectification_meta,
    roi_metadata,
    input_root,
    sync_manifest,
    time_segments,
    scenes,
    positions,
    black_threshold,
    near_black_threshold,
    logger,
):
    safe_rect = roi_metadata["common_valid_roi"]
    final_crop = roi_metadata["final_release_crop_16_9"]
    final_crop_in_common = {
        "x": final_crop["x"] - safe_rect["x"],
        "y": final_crop["y"] - safe_rect["y"],
        "width": final_crop["width"],
        "height": final_crop["height"],
    }
    video_mapping = scene_to_video_map(sync_manifest)
    scene_reports = []
    worst_cameras = []

    for scene_id in scenes:
        ensure(scene_id in video_mapping, f"sync manifest 中找不到场景 {scene_id} 对应视频")
        source_video = video_mapping[scene_id]
        scene_report = {"scene_id": scene_id, "source_video": source_video, "segments": []}
        for segment_index, (start_time, end_time) in enumerate(time_segments[scene_id]):
            segment_report = {
                "segment_index": segment_index,
                "start_time": start_time,
                "end_time": end_time,
                "samples": [],
            }
            for position_name, sample_time in sample_times_for_segment(start_time, end_time, positions):
                sample_report = {
                    "position": position_name,
                    "sample_time_sec": sample_time,
                    "cameras": {},
                }
                for camera_id in CAMERAS:
                    source_path = Path(input_root) / camera_id / source_video
                    ensure(source_path.exists(), f"缺少同步后视频：{source_path}")
                    capture = cv2.VideoCapture(str(source_path))
                    ensure(capture.isOpened(), f"无法打开视频：{source_path}")
                    capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
                    ok, frame = capture.read()
                    capture.release()
                    ensure(ok, f"无法在 {source_path} 的 {sample_time:.3f}s 读取帧")

                    map_path = Path(rectify_dir) / rectification_meta["cameras"][camera_id]["map_file"]
                    data = np.load(map_path)
                    rectified = cv2.remap(frame, data["map_x"], data["map_y"], cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
                    common = crop_frame(rectified, safe_rect)
                    cropped = crop_frame(common, final_crop_in_common)
                    border_stats = compute_border_stats(cropped)
                    sample_report["cameras"][camera_id] = border_stats
                    worst_cameras.append(
                        {
                            "scene_id": scene_id,
                            "segment_index": segment_index,
                            "position": position_name,
                            "camera_id": camera_id,
                            "overall_black_ratio": border_stats["overall_black_ratio"],
                            "overall_near_black_ratio": border_stats["overall_near_black_ratio"],
                        }
                    )
                segment_report["samples"].append(sample_report)
            scene_report["segments"].append(segment_report)
        scene_reports.append(scene_report)
        logger.info(f"已完成 ROI 抽样验证场景 {scene_id}")

    worst_cameras.sort(key=lambda item: (item["overall_black_ratio"], item["overall_near_black_ratio"]), reverse=True)
    return scene_reports, worst_cameras


def summarize_validation(worst_cameras, black_threshold, near_black_threshold):
    if not worst_cameras:
        return {"status": "fail", "summary": "没有任何抽样结果"}, []
    failing = [item for item in worst_cameras if item["overall_black_ratio"] > black_threshold]
    warning = [item for item in worst_cameras if item["overall_black_ratio"] <= black_threshold and item["overall_near_black_ratio"] > near_black_threshold]
    if failing:
        status = "fail"
        summary = f"存在 {len(failing)} 个抽样结果的边界黑像素比例超过阈值 {black_threshold:.4f}"
    elif warning:
        status = "warning"
        summary = f"没有黑边失败样本，但存在 {len(warning)} 个近黑边比例超过阈值 {near_black_threshold:.4f} 的样本"
    else:
        status = "pass"
        summary = "所有抽样结果的黑边与近黑边比例均在阈值内"
    return {"status": status, "summary": summary}, worst_cameras[:10]


def main():
    parser = argparse.ArgumentParser(description="自动验证 release ROI 是否足够保守，并输出独立验证报告。")
    parser.add_argument("--config", default=None, help="Optional RealDynLFV batch YAML config.")
    parser.add_argument("--batch-name", default=None)
    parser.add_argument("--rectify-dir", default=None)
    parser.add_argument("--roi-metadata", default=None)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--time-json", default=None)
    parser.add_argument("--time-key", default=None)
    parser.add_argument("--scenes", nargs="*", default=None)
    args = parser.parse_args()

    if args.config:
        pipeline_config = load_pipeline_config(args.config)
        selected_batch = args.batch_name or batch_id(pipeline_config)
        selected_time_key = args.time_key or time_key(pipeline_config)
        config = {
            "paths": {
                "rectify_dir": config_value(pipeline_config, "paths", "rectification_dir"),
                "roi_metadata": config_value(pipeline_config, "paths", "roi_metadata"),
                "input_root": config_value(pipeline_config, "paths", "synced_root"),
                "time_json": config_value(pipeline_config, "paths", "time_segments"),
                "sync_manifest": config_value(pipeline_config, "paths", "sync_manifest"),
            },
            "runtime": {
                "batch_name": selected_batch,
                "time_key": selected_time_key,
                "roi_validation_sample_scenes": config_value(pipeline_config, "roi", "validation_scenes", default=["*"]),
                "roi_validation_frame_positions": config_value(pipeline_config, "roi", "validation_frame_positions", default=["start", "middle", "end"]),
                "roi_black_border_threshold": config_value(pipeline_config, "roi", "black_border_threshold", default=0.01),
                "roi_near_black_threshold": config_value(pipeline_config, "roi", "near_black_threshold", default=0.03),
            },
        }
    else:
        config = get_default_config(args.batch_name)
    rectify_dir_value = args.rectify_dir or config["paths"]["rectify_dir"]
    roi_metadata_value = args.roi_metadata or config["paths"]["roi_metadata"]
    input_root = args.input_root or config["paths"]["input_root"]
    time_json = args.time_json or config["paths"]["time_json"]
    time_key = args.time_key or config["runtime"]["time_key"]
    sync_manifest_path = config["paths"]["sync_manifest"]
    black_threshold = float(config["runtime"].get("roi_black_border_threshold", 0.01))
    near_black_threshold = float(config["runtime"].get("roi_near_black_threshold", 0.03))
    positions = list(config["runtime"].get("roi_validation_frame_positions", ["start", "middle", "end"]))

    rectify_dir, rectification_meta = load_rectification_assets(rectify_dir_value)
    roi_path, roi_metadata = load_roi_metadata(rectify_dir, roi_metadata_value)
    ensure(
        roi_metadata["rectification_asset_version"] == rectification_meta["rectification_asset_version"],
        "ROI metadata 与 rectification assets 版本不匹配",
    )
    time_segments = load_time_config(time_json, time_key)
    sync_manifest = load_json(sync_manifest_path)
    scenes = resolve_scene_ids(args.scenes, config["runtime"], time_segments)

    log_path = Path(rectify_dir) / "release_roi_validation.log"
    report_path = Path(rectify_dir) / "release_roi_validation.json"
    logger = ValidationLogger(log_path)
    logger.info(f"开始 ROI 自动验证，batch={config['runtime']['batch_name']}，scenes={scenes}")

    static_validation = validate_static_geometry(roi_metadata)
    sampled_scene_reports, worst_cameras = validate_sampled_frames(
        rectify_dir=rectify_dir,
        rectification_meta=rectification_meta,
        roi_metadata=roi_metadata,
        input_root=input_root,
        sync_manifest=sync_manifest,
        time_segments=time_segments,
        scenes=scenes,
        positions=positions,
        black_threshold=black_threshold,
        near_black_threshold=near_black_threshold,
        logger=logger,
    )
    status_summary, top_worst = summarize_validation(worst_cameras, black_threshold, near_black_threshold)

    report = {
        "schema_version": "release_roi_validation_v1",
        "rectification_asset_version": rectification_meta["rectification_asset_version"],
        "roi_asset_version": roi_metadata["roi_asset_version"],
        "roi_metadata_path": str(roi_path),
        "batch_name": config["runtime"]["batch_name"],
        "time_key": time_key,
        "validation_scenes": scenes,
        "sample_positions": positions,
        "thresholds": {
            "black_border_threshold": black_threshold,
            "near_black_threshold": near_black_threshold,
        },
        "static_validation": static_validation,
        "sampled_scene_reports": sampled_scene_reports,
        "worst_camera_samples_top10": top_worst,
        "validation_status": status_summary["status"],
        "validation_summary": status_summary["summary"],
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)
    logger.info(f"ROI 自动验证完成，状态={status_summary['status']}，报告={report_path}")
    logger.close()


if __name__ == "__main__":
    main()
