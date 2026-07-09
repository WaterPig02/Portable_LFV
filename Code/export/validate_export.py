import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import get_default_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from pipeline_config import batch_id as pipeline_batch_id
from pipeline_config import config_value as pipeline_config_value
from pipeline_config import load_pipeline_config


CAMERAS = [f"CAM_{row}{col}" for row in "ABCDE" for col in range(1, 6)]
NEIGHBOR_CAMERAS = ["CAM_C2", "CAM_C4", "CAM_B3", "CAM_D3"]
WIDE_BASELINE_CAMERAS = ["CAM_A1", "CAM_E5"]


def apply_pipeline_config(config, pipeline_config_path):
    pipeline = load_pipeline_config(pipeline_config_path)
    output_root = pipeline_config_value(pipeline, "paths", "output_root")
    if output_root is not None:
        config["paths"]["output_root"] = str(output_root)
    config["runtime"]["batch_name"] = pipeline_batch_id(pipeline)

    validation_map = {
        "export_validation_default_samples": ("export_validation", "default_samples", int),
        "export_validation_upgraded_samples": ("export_validation", "upgraded_samples", int),
        "export_validation_black_border_fail_threshold": ("export_validation", "black_border_fail_threshold", float),
        "export_validation_near_black_warning_threshold": ("export_validation", "near_black_warning_threshold", float),
        "export_validation_enable_auto_upgrade": ("export_validation", "enable_auto_upgrade", bool),
    }
    for target_key, (section, key, caster) in validation_map.items():
        value = pipeline_config_value(pipeline, section, key)
        if value is not None:
            config["runtime"][target_key] = caster(value)
    return config


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class CheckLogger:
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


def camera_to_view_id(camera_id):
    suffix = camera_id.split("_", 1)[1]
    row = suffix[0]
    col = int(suffix[1:])
    return f"view_{'ABCDE'.index(row) * 5 + (col - 1):02d}"


def collect_view_dirs(scene_dir):
    return {camera_id: scene_dir / camera_to_view_id(camera_id) for camera_id in CAMERAS}


def list_frame_files(view_dir):
    return sorted([path for path in view_dir.iterdir() if path.suffix.lower() in {".jpg", ".png"}])


def sample_frame_indices(frame_count, sample_count):
    if frame_count <= 0 or sample_count <= 0:
        return []
    if sample_count == 1:
        return [0]
    positions = np.linspace(0, frame_count - 1, sample_count)
    return sorted({int(round(value)) for value in positions})


def longest_true_run(mask):
    current = 0
    longest = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def edge_strip_stats(image, edge_name, threshold, band_px=8, full_line_ratio=0.80):
    if edge_name == "top":
        strip = image[:band_px, :, :]
        line_axis = 1
    elif edge_name == "bottom":
        strip = image[-band_px:, :, :]
        line_axis = 1
    elif edge_name == "left":
        strip = image[:, :band_px, :]
        line_axis = 0
    elif edge_name == "right":
        strip = image[:, -band_px:, :]
        line_axis = 0
    else:
        raise ValueError(f"unsupported edge: {edge_name}")

    mask = np.all(strip <= threshold, axis=2)
    line_black_ratio = np.mean(mask, axis=line_axis)
    full_line_mask = line_black_ratio >= full_line_ratio
    return {
        "band_px": int(band_px),
        "full_line_ratio_threshold": float(full_line_ratio),
        "max_line_black_ratio": float(np.max(line_black_ratio)) if line_black_ratio.size else 0.0,
        "full_line_count": int(np.sum(full_line_mask)),
        "longest_full_line_run_px": longest_true_run(full_line_mask),
    }


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
            "black_strip": edge_strip_stats(image, edge_name, black_threshold),
            "near_black_strip": edge_strip_stats(image, edge_name, near_black_threshold),
        }
        black_ratios.append(black_ratio)
        near_black_ratios.append(near_black_ratio)
    blocking_black_border = any(edge["black_strip"]["longest_full_line_run_px"] >= 3 for edge in per_edge.values())
    near_black_border_warning = any(edge["near_black_strip"]["longest_full_line_run_px"] >= 3 for edge in per_edge.values())
    return {
        "overall_black_ratio": float(np.mean(black_ratios)),
        "overall_near_black_ratio": float(np.mean(near_black_ratios)),
        "blocking_black_border": bool(blocking_black_border),
        "near_black_border_warning": bool(near_black_border_warning),
        "per_edge": per_edge,
    }


def estimate_vertical_residual(ref_image, target_image):
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(ref_gray, maxCorners=300, qualityLevel=0.01, minDistance=12)
    if points is None or len(points) < 10:
        return {"count": 0, "vertical_residual_mean": None, "vertical_residual_p95": None, "vertical_residual_max": None}
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(ref_gray, target_gray, points, None)
    if next_points is None or status is None:
        return {"count": 0, "vertical_residual_mean": None, "vertical_residual_p95": None, "vertical_residual_max": None}
    points_2d = points.reshape(-1, 2)
    next_points_2d = next_points.reshape(-1, 2)
    valid = status.reshape(-1) == 1
    if int(valid.sum()) < 10:
        return {"count": int(valid.sum()), "vertical_residual_mean": None, "vertical_residual_p95": None, "vertical_residual_max": None}
    residuals = np.abs(next_points_2d[valid, 1] - points_2d[valid, 1])
    return {
        "count": int(valid.sum()),
        "vertical_residual_mean": float(np.mean(residuals)),
        "vertical_residual_p95": float(np.percentile(residuals, 95)),
        "vertical_residual_max": float(np.max(residuals)),
    }


def add_label(image, text):
    canvas = image.copy()
    cv2.putText(canvas, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def write_grid(output_path, images, cols):
    rows = []
    for idx in range(0, len(images), cols):
        rows.append(np.hstack(images[idx : idx + cols]))
    grid = np.vstack(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)


def prepare_checks_dir(checks_dir):
    if checks_dir.exists():
        shutil.rmtree(checks_dir)
    checks_dir.mkdir(parents=True, exist_ok=True)


def generate_visual_checks(scene_dir, checks_dir, frame_indices, logger):
    view_dirs = collect_view_dirs(scene_dir)
    ref_frames = list_frame_files(view_dirs["CAM_C3"])
    logger.info(f"生成人工检查图，抽样帧索引：{frame_indices}")
    for frame_index in frame_indices:
        frame_name = ref_frames[frame_index].name
        images = []
        for camera_id in CAMERAS:
            image = cv2.imread(str(view_dirs[camera_id] / frame_name), cv2.IMREAD_COLOR)
            images.append(add_label(image, camera_id))
        write_grid(checks_dir / f"grid_5x5_{frame_name}", images, cols=5)
        for row in "ABCDE":
            row_cameras = [f"CAM_{row}{col}" for col in range(1, 6)]
            row_images = [add_label(cv2.imread(str(view_dirs[camera_id] / frame_name), cv2.IMREAD_COLOR), camera_id) for camera_id in row_cameras]
            write_grid(checks_dir / f"grid_row_{row}_{frame_name}", row_images, cols=5)
        for col in range(1, 6):
            col_cameras = [f"CAM_{row}{col}" for row in "ABCDE"]
            col_images = [add_label(cv2.imread(str(view_dirs[camera_id] / frame_name), cv2.IMREAD_COLOR), camera_id) for camera_id in col_cameras]
            write_grid(checks_dir / f"grid_col_{col}_{frame_name}", col_images, cols=1)


def rank_worst_cameras(view_reports):
    ranked = []
    for camera_id, payload in view_reports.items():
        for sample in payload["border_stats_samples"]:
            ranked.append(
                {
                    "camera_id": camera_id,
                    "frame_name": sample["frame_name"],
                    "overall_black_ratio": sample["border_stats"]["overall_black_ratio"],
                    "overall_near_black_ratio": sample["border_stats"]["overall_near_black_ratio"],
                    "blocking_black_border": sample["border_stats"].get("blocking_black_border", False),
                    "near_black_border_warning": sample["border_stats"].get("near_black_border_warning", False),
                }
            )
    ranked.sort(
        key=lambda item: (
            item["blocking_black_border"],
            item["near_black_border_warning"],
            item["overall_black_ratio"],
            item["overall_near_black_ratio"],
        ),
        reverse=True,
    )
    return ranked[:10]


def inspect_scene_structure(scene_dir):
    report = {
        "missing_view_dirs": [],
        "in_progress_markers": [],
        "missing_view_metadata": [],
        "empty_views": [],
        "resolution_consistent": True,
        "frame_count_consistent": True,
        "sequence_metadata_exists": False,
        "reference_camera": None,
        "phase_offset_c3": None,
        "view_reports": {},
        "scene_errors": [],
    }

    view_dirs = collect_view_dirs(scene_dir)
    resolution_set = set()
    frame_count_set = set()
    for camera_id, view_dir in view_dirs.items():
        if not view_dir.exists():
            report["missing_view_dirs"].append(camera_id)
            continue
        if (view_dir / ".in_progress").exists():
            report["in_progress_markers"].append(camera_id)
        metadata_path = view_dir / "view_metadata.json"
        if not metadata_path.exists():
            report["missing_view_metadata"].append(camera_id)
            continue
        metadata = load_json(metadata_path)
        frame_files = list_frame_files(view_dir)
        if not frame_files:
            report["empty_views"].append(camera_id)
            continue
        resolution_set.add(tuple(metadata["output_resolution"]))
        frame_count_set.add(len(frame_files))
        report["view_reports"][camera_id] = {
            "frame_files": frame_files,
            "frame_count": len(frame_files),
            "output_resolution": metadata["output_resolution"],
            "border_stats_samples": [],
        }

    if len(resolution_set) > 1:
        report["resolution_consistent"] = False
    if len(frame_count_set) > 1:
        report["frame_count_consistent"] = False

    sequence_metadata_path = scene_dir / "metadata.json"
    report["sequence_metadata_exists"] = sequence_metadata_path.exists()
    if sequence_metadata_path.exists():
        sequence_metadata = load_json(sequence_metadata_path)
        report["reference_camera"] = sequence_metadata.get("reference_camera")
        report["phase_offset_c3"] = sequence_metadata.get("phase_offset_ms", {}).get("CAM_C3")

    if report["missing_view_dirs"]:
        report["scene_errors"].append(f"缺少视角目录：{report['missing_view_dirs']}")
    if report["in_progress_markers"]:
        report["scene_errors"].append(f"存在未完成导出标记：{report['in_progress_markers']}")
    if report["missing_view_metadata"]:
        report["scene_errors"].append(f"缺少 view_metadata.json：{report['missing_view_metadata']}")
    if report["empty_views"]:
        report["scene_errors"].append(f"存在空视角目录：{report['empty_views']}")
    if not report["resolution_consistent"]:
        report["scene_errors"].append("视角输出分辨率不一致")
    if not report["frame_count_consistent"]:
        report["scene_errors"].append("视角输出帧数不一致")
    if not report["sequence_metadata_exists"]:
        report["scene_errors"].append("缺少 metadata.json")

    return report


def evaluate_scene_samples(scene_dir, view_reports, frame_indices):
    ref_frames = view_reports["CAM_C3"]["frame_files"]
    for camera_id, payload in view_reports.items():
        frame_files = payload["frame_files"]
        for frame_index in frame_indices:
            image = cv2.imread(str(frame_files[frame_index]), cv2.IMREAD_COLOR)
            payload["border_stats_samples"].append(
                {
                    "frame_index": int(frame_index),
                    "frame_name": frame_files[frame_index].name,
                    "border_stats": compute_border_stats(image),
                }
            )

    epipolar_checks = {}
    for camera_id in NEIGHBOR_CAMERAS + WIDE_BASELINE_CAMERAS:
        metrics = []
        target_frames = view_reports[camera_id]["frame_files"]
        for frame_index in frame_indices:
            ref_img = cv2.imread(str(ref_frames[frame_index]), cv2.IMREAD_COLOR)
            tgt_img = cv2.imread(str(target_frames[frame_index]), cv2.IMREAD_COLOR)
            metrics.append(
                {
                    "frame_index": int(frame_index),
                    "frame_name": ref_frames[frame_index].name,
                    "metrics": estimate_vertical_residual(ref_img, tgt_img),
                }
            )
        epipolar_checks[camera_id] = metrics
    return epipolar_checks


def detect_border_suspicious_reasons(view_reports, black_fail_threshold, near_black_warning_threshold):
    reasons = []
    for camera_id, payload in view_reports.items():
        for sample in payload["border_stats_samples"]:
            stats = sample["border_stats"]
            if stats.get("blocking_black_border", False):
                reasons.append(f"{camera_id}/{sample['frame_name']} 检测到连续整边黑色无效带")
            elif stats.get("near_black_border_warning", False):
                reasons.append(f"{camera_id}/{sample['frame_name']} 检测到连续近黑整边，建议复查")
            elif stats["overall_near_black_ratio"] > near_black_warning_threshold:
                reasons.append(
                    f"{camera_id}/{sample['frame_name']} 的 overall_near_black_ratio={stats['overall_near_black_ratio']:.6f} 超过阈值 {near_black_warning_threshold:.6f}"
                )
            for edge_name, edge_payload in stats["per_edge"].items():
                if edge_payload["black_strip"]["longest_full_line_run_px"] >= 3:
                    reasons.append(
                        f"{camera_id}/{sample['frame_name']} 的 {edge_name} 边存在连续 {edge_payload['black_strip']['longest_full_line_run_px']}px 整边黑色无效带"
                    )
        if len(reasons) >= 20:
            break
    return reasons


def collect_epipolar_warnings(epipolar_checks):
    warnings = []
    for camera_id, metrics_list in epipolar_checks.items():
        for payload in metrics_list:
            metrics = payload["metrics"]
            if metrics["vertical_residual_mean"] is not None and metrics["vertical_residual_mean"] > 8.0:
                warnings.append(
                    f"{camera_id}/{payload['frame_name']} 的辅助极线指标 vertical_residual_mean={metrics['vertical_residual_mean']:.3f} 偏大"
                )
                if len(warnings) >= 20:
                    break
        if len(warnings) >= 20:
            break
    return warnings


def classify_scene(
    structure_report,
    border_suspicious_reasons,
    epipolar_warnings,
    upgraded,
    black_fail_threshold,
    near_black_warning_threshold,
):
    if structure_report["scene_errors"]:
        return "fail", "结构检查失败，存在缺失或不一致文件"

    has_black_fail = False
    has_warning = False
    for payload in structure_report["view_reports"].values():
        for sample in payload["border_stats_samples"]:
            stats = sample["border_stats"]
            if stats.get("blocking_black_border", False):
                has_black_fail = True
                break
            if stats.get("near_black_border_warning", False) or stats["overall_near_black_ratio"] > near_black_warning_threshold:
                has_warning = True
        if has_black_fail:
            break

    if has_black_fail:
        return "fail", "抽样结果存在黑边失败样本"
    if border_suspicious_reasons or has_warning:
        if upgraded:
            return "warning", "抽样结果没有黑边失败，但存在可疑信号，建议重点复查"
        return "warning", "初检发现可疑信号"
    if epipolar_warnings:
        return "warning", "黑边检查通过，但辅助极线指标偏大，建议按需复查"
    return "pass", "结构完整，抽样检查通过"


def build_scene_report(
    scene_dir,
    structure_report,
    epipolar_checks,
    epipolar_warnings,
    initial_indices,
    final_indices,
    upgraded,
    suspicious_reasons,
    scene_status,
    scene_summary,
):
    return {
        "scene_dir": str(scene_dir),
        "sampling_plan": {
            "initial_sample_count": len(initial_indices),
            "final_sample_count": len(final_indices),
            "upgraded": upgraded,
            "sample_frame_indices": [int(index) for index in final_indices],
        },
        "scene_status": scene_status,
        "scene_summary": scene_summary,
        "suspicious_reasons": suspicious_reasons,
        "epipolar_warnings": epipolar_warnings,
        "views": {
            camera_id: {
                "frame_count": payload["frame_count"],
                "output_resolution": payload["output_resolution"],
                "border_stats_samples": payload["border_stats_samples"],
            }
            for camera_id, payload in structure_report["view_reports"].items()
        },
        "border_worst_cameras": rank_worst_cameras(structure_report["view_reports"]),
        "resolution_consistent": structure_report["resolution_consistent"],
        "frame_count_consistent": structure_report["frame_count_consistent"],
        "sequence_metadata_exists": structure_report["sequence_metadata_exists"],
        "reference_camera": structure_report["reference_camera"],
        "phase_offset_c3": structure_report["phase_offset_c3"],
        "scene_errors": structure_report["scene_errors"],
        "epipolar_checks": epipolar_checks,
        "epipolar_check_note": (
            "该指标仅为辅助探索性指标。它基于自然场景跨视角 LK 匹配，"
            "容易受到遮挡、视差、运动和纹理重复影响，不能直接解释为真实几何误差。"
        ),
        "epipolar_check_blocking": False,
        "black_border_check_note": "黑边指标统计的是图像四条边界像素中的纯黑/近黑比例，不等同于整张图的黑边面积比例。",
    }


def run_scene_checks(scene_dir, checks_dir, logger, default_samples, upgraded_samples, black_fail_threshold, near_black_warning_threshold, enable_auto_upgrade):
    structure_report = inspect_scene_structure(scene_dir)
    if structure_report["scene_errors"]:
        report = build_scene_report(
            scene_dir=scene_dir,
            structure_report=structure_report,
            epipolar_checks={},
            initial_indices=[],
            final_indices=[],
            upgraded=False,
            suspicious_reasons=structure_report["scene_errors"],
            scene_status="fail",
            scene_summary="结构检查失败，未进入抽样阶段",
        )
        return report

    frame_count = next(iter(structure_report["view_reports"].values()))["frame_count"]
    initial_indices = sample_frame_indices(frame_count, default_samples)
    epipolar_checks = evaluate_scene_samples(scene_dir, structure_report["view_reports"], initial_indices)
    border_suspicious_reasons = detect_border_suspicious_reasons(
        structure_report["view_reports"],
        black_fail_threshold,
        near_black_warning_threshold,
    )
    epipolar_warnings = collect_epipolar_warnings(epipolar_checks)

    upgraded = False
    final_indices = initial_indices
    if enable_auto_upgrade and border_suspicious_reasons:
        upgraded = True
        logger.warn(f"场景 {scene_dir.name} 初检可疑，升级到 {upgraded_samples} 帧复检")
        final_indices = sample_frame_indices(frame_count, upgraded_samples)
        for payload in structure_report["view_reports"].values():
            payload["border_stats_samples"] = []
        epipolar_checks = evaluate_scene_samples(scene_dir, structure_report["view_reports"], final_indices)
        border_suspicious_reasons = detect_border_suspicious_reasons(
            structure_report["view_reports"],
            black_fail_threshold,
            near_black_warning_threshold,
        )
        epipolar_warnings = collect_epipolar_warnings(epipolar_checks)

    prepare_checks_dir(checks_dir)
    generate_visual_checks(scene_dir, checks_dir, final_indices, logger)
    scene_status, scene_summary = classify_scene(
        structure_report,
        border_suspicious_reasons,
        epipolar_warnings,
        upgraded,
        black_fail_threshold,
        near_black_warning_threshold,
    )
    suspicious_reasons = border_suspicious_reasons + epipolar_warnings
    return build_scene_report(
        scene_dir=scene_dir,
        structure_report=structure_report,
        epipolar_checks=epipolar_checks,
        epipolar_warnings=epipolar_warnings,
        initial_indices=initial_indices,
        final_indices=final_indices,
        upgraded=upgraded,
        suspicious_reasons=suspicious_reasons,
        scene_status=scene_status,
        scene_summary=scene_summary,
    )


def write_scene_report(checks_dir, report):
    report_path = checks_dir / "check_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)
    return report_path


def summarize_batch(scene_reports, batch_name, profile):
    summary = {
        "schema_version": "lfv_export_batch_check_summary_v1",
        "batch_name": batch_name,
        "profile": profile,
        "scene_count": len(scene_reports),
        "status_counts": {"pass": 0, "warning": 0, "fail": 0},
        "scenes": [],
        "recommended_review_scenes": [],
        "worst_border_samples_top20": [],
    }
    worst_samples = []
    for scene_id, report in scene_reports.items():
        status = report["scene_status"]
        summary["status_counts"][status] += 1
        summary["scenes"].append(
            {
                "scene_id": scene_id,
                "scene_status": status,
                "scene_summary": report["scene_summary"],
                "upgraded": report["sampling_plan"]["upgraded"],
                "final_sample_count": report["sampling_plan"]["final_sample_count"],
            }
        )
        if status in {"warning", "fail"}:
            summary["recommended_review_scenes"].append(scene_id)
        for item in report["border_worst_cameras"]:
            worst_samples.append({"scene_id": scene_id, **item})

    worst_samples.sort(key=lambda item: (item["overall_black_ratio"], item["overall_near_black_ratio"]), reverse=True)
    summary["worst_border_samples_top20"] = worst_samples[:20]
    return summary


def iter_scene_dirs(profile_root, scene_name=None):
    if scene_name is not None:
        scene_dir = profile_root / scene_name
        if not scene_dir.exists():
            raise RuntimeError(f"场景目录不存在：{scene_dir}")
        return [scene_dir]
    return sorted([path for path in profile_root.iterdir() if path.is_dir()])


def main():
    parser = argparse.ArgumentParser(description="检查导出结果，支持单场景检查与整批次全量检查，并在可疑时自动升级抽样。")
    parser.add_argument("--config", default=None, help="Optional RealDynLFV batch YAML config.")
    parser.add_argument("--batch-name", default=None, help="Optional named default config. Public use should prefer --config.")
    parser.add_argument("--profile", required=True, choices=["benchmark", "fidelity"])
    parser.add_argument("--scene", default=None)
    parser.add_argument("--all-scenes", action="store_true", help="显式要求遍历当前 batch/profile 下的全部场景。")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    config = get_default_config(args.batch_name)
    if args.config:
        config = apply_pipeline_config(config, args.config)
    output_root = Path(args.output_root or config["paths"]["output_root"])
    profile_root = output_root / args.profile
    logs_dir = output_root / "logs"
    checks_root = output_root / "checks"
    logger_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_check_{args.profile}_{args.scene}.log"
        if args.scene and not args.all_scenes
        else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_check_{args.profile}_all.log"
    )
    logger = CheckLogger(logs_dir / logger_name)

    default_samples = int(config["runtime"].get("export_validation_default_samples", 3))
    upgraded_samples = int(config["runtime"].get("export_validation_upgraded_samples", 9))
    black_fail_threshold = float(config["runtime"].get("export_validation_black_border_fail_threshold", 0.01))
    near_black_warning_threshold = float(config["runtime"].get("export_validation_near_black_warning_threshold", 0.03))
    enable_auto_upgrade = bool(config["runtime"].get("export_validation_enable_auto_upgrade", True))

    logger.info(
        f"开始检查，batch={config['runtime']['batch_name']}，profile={args.profile}，"
        f"scene={args.scene or 'ALL'}，default_samples={default_samples}，upgraded_samples={upgraded_samples}"
    )
    scene_dirs = iter_scene_dirs(profile_root, None if args.all_scenes or args.scene is None else args.scene)
    if args.scene and not args.all_scenes:
        scene_dirs = [profile_root / args.scene]

    scene_reports = {}
    for scene_dir in scene_dirs:
        logger.info(f"开始检查场景 {scene_dir.name}")
        checks_dir = checks_root / scene_dir.name
        report = run_scene_checks(
            scene_dir=scene_dir,
            checks_dir=checks_dir,
            logger=logger,
            default_samples=default_samples,
            upgraded_samples=upgraded_samples,
            black_fail_threshold=black_fail_threshold,
            near_black_warning_threshold=near_black_warning_threshold,
            enable_auto_upgrade=enable_auto_upgrade,
        )
        report_path = write_scene_report(checks_dir, report)
        scene_reports[scene_dir.name] = report
        logger.info(
            f"场景 {scene_dir.name} 检查完成，状态={report['scene_status']}，"
            f"抽样 {report['sampling_plan']['final_sample_count']} 帧，报告={report_path}"
        )

    if args.scene and not args.all_scenes:
        logger.close()
        return

    batch_summary = summarize_batch(scene_reports, config["runtime"]["batch_name"], args.profile)
    summary_path = checks_root / f"_batch_summary_{args.profile}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(batch_summary, handle, indent=4, ensure_ascii=False)
    logger.info(
        f"批次检查完成：pass={batch_summary['status_counts']['pass']}，"
        f"warning={batch_summary['status_counts']['warning']}，"
        f"fail={batch_summary['status_counts']['fail']}，汇总={summary_path}"
    )
    logger.close()


if __name__ == "__main__":
    main()
