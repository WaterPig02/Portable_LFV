import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from pipeline_config import batch_id, config_value, load_pipeline_config

REFERENCE_CAMERA = "CAM_C3"
CHECKERBOARD_COLS = 8
CHECKERBOARD_ROWS = 8
SQUARE_SIZE_MM = 25.0
IMG_W = 3840
IMG_H = 2160
DEFAULTS = {
    "secondsyn": {
        "input_root": Path(r"E:\5x5_LFV\2nd_synced_output"),
        "calibration_json": PROJECT_ROOT / "Output" / "calibration_raw_stereo_locked_secondsyn.json",
    },
    "firstsyn": {
        "input_root": Path(r"E:\5x5_LFV\1st_synced_output"),
        "calibration_json": PROJECT_ROOT / "Output" / "calibration_raw_stereo_locked_firstsyn.json",
    },
}


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)


def find_video(input_root, camera_id, target_video):
    camera_dir = Path(input_root) / camera_id
    candidates = [camera_dir / target_video]
    suffixes = [".mp4", ".MP4"]
    stem = Path(target_video).stem
    for suffix in suffixes:
        candidates.append(camera_dir / f"{stem}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def detect_corners_raw(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS), flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners, gray.shape[::-1]
    return False, None, gray.shape[::-1]


def get_grid_vectors(corners, rows, cols):
    grid = corners.reshape(rows, cols, 2)
    vec_x = np.mean(grid[:, 1:, :] - grid[:, :-1, :], axis=(0, 1))
    vec_y = np.mean(grid[1:, :, :] - grid[:-1, :, :], axis=(0, 1))
    norm_x = np.linalg.norm(vec_x)
    norm_y = np.linalg.norm(vec_y)
    if norm_x < 1e-8 or norm_y < 1e-8:
        return None, None
    return vec_x / norm_x, vec_y / norm_y


def generate_8_orientations(corners, rows, cols):
    grid = corners.reshape(rows, cols, 2)
    orientations = []
    for flipped in [False, True]:
        current_grid = grid if not flipped else np.flip(grid, axis=1)
        for rot in range(4):
            rotated = np.rot90(current_grid, k=rot, axes=(0, 1))
            orientations.append(rotated.reshape(-1, 1, 2))
    return orientations


def align_corners_to_master(target_corners, ref_vec_x, ref_vec_y, rows, cols):
    candidates = generate_8_orientations(target_corners, rows, cols)
    best_score = -np.inf
    best_corners = None
    for candidate in candidates:
        cand_vec_x, cand_vec_y = get_grid_vectors(candidate, rows, cols)
        if cand_vec_x is None:
            continue
        score = float(np.dot(cand_vec_x, ref_vec_x) + np.dot(cand_vec_y, ref_vec_y))
        if score > best_score:
            best_score = score
            best_corners = candidate
    return best_corners, best_score


def make_object_points():
    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp


def sample_frame_indices(video_path, start_time, sample_interval, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    ensure(cap.isOpened(), f"无法打开视频：{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    ensure(fps > 0, f"无法读取 fps：{video_path}")
    start_frame = int(start_time * fps)
    step_frame = max(1, int(sample_interval * fps))
    indices = list(range(start_frame, total_frames, step_frame))
    if max_frames and max_frames > 0:
        indices = indices[:max_frames]
    return indices, fps, total_frames


def collect_camera_detections(camera_id, video_path, frame_indices, ref_vec_x, ref_vec_y, debug_dir):
    cap = cv2.VideoCapture(str(video_path))
    records = []
    detected = []
    img_size = None
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            records.append({"frame_index": int(frame_idx), "detected": False, "reason": "read_failed"})
            continue
        found, corners, size = detect_corners_raw(frame)
        img_size = size
        if not found:
            records.append({"frame_index": int(frame_idx), "detected": False, "reason": "checkerboard_not_found"})
            continue
        aligned, orientation_score = align_corners_to_master(corners, ref_vec_x, ref_vec_y, CHECKERBOARD_ROWS, CHECKERBOARD_COLS)
        if aligned is None:
            records.append({"frame_index": int(frame_idx), "detected": False, "reason": "orientation_align_failed"})
            continue
        pts = aligned.reshape(-1, 2)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        center = pts.mean(axis=0)
        bbox_size = max_xy - min_xy
        area = float(max(0.0, bbox_size[0]) * max(0.0, bbox_size[1]))
        rec = {
            "frame_index": int(frame_idx),
            "detected": True,
            "corners": aligned,
            "center": [float(center[0]), float(center[1])],
            "bbox": [float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])],
            "bbox_area": area,
            "orientation_score": float(orientation_score),
        }
        records.append(rec)
        detected.append(rec)
    cap.release()
    save_debug_samples(camera_id, video_path, detected, debug_dir)
    return records, img_size


def save_debug_samples(camera_id, video_path, detected_records, debug_dir):
    if not detected_records:
        return
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = sorted(set([0, len(detected_records) // 2, len(detected_records) - 1]))
    cap = cv2.VideoCapture(str(video_path))
    for label, rec in [("sample", detected_records[i]) for i in sample_indices]:
        frame_idx = rec["frame_index"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.drawChessboardCorners(frame, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS), rec["corners"], True)
        cv2.putText(frame, f"{camera_id} frame={frame_idx}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        out_path = debug_dir / f"{camera_id}_{label}_{frame_idx}.jpg"
        cv2.imwrite(str(out_path), frame)
    cap.release()


def strip_corners(records):
    clean = []
    for rec in records:
        item = {k: v for k, v in rec.items() if k != "corners"}
        clean.append(item)
    return clean


def compute_mono_errors(calib, detections, img_size, objp):
    summary = {}
    object_points = objp.astype(np.float32)
    for cam, records in detections.items():
        valid = [rec for rec in records if rec.get("detected")]
        if len(valid) < 4:
            summary[cam] = {"status": "insufficient_points", "valid_frames": len(valid)}
            continue
        K = np.array(calib[cam]["K"], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
        errors = []
        for rec in valid:
            imgpoints = rec["corners"].astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(object_points, imgpoints, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
            err = np.linalg.norm(projected.reshape(-1, 2) - imgpoints.reshape(-1, 2), axis=1)
            errors.append({
                "frame_index": rec["frame_index"],
                "mean_error_px": float(np.mean(err)),
                "max_error_px": float(np.max(err)),
            })
        values = [item["mean_error_px"] for item in errors]
        summary[cam] = summarize_error_values(values)
        summary[cam]["valid_frames"] = len(valid)
        summary[cam]["worst_frames"] = sorted(errors, key=lambda x: x["mean_error_px"], reverse=True)[:5]
    return summary


def compute_stereo_errors(calib, detections, objp):
    summary = {}
    master_records = {rec["frame_index"]: rec for rec in detections.get(REFERENCE_CAMERA, []) if rec.get("detected")}
    for cam, records in detections.items():
        if cam == REFERENCE_CAMERA:
            continue
        slave_records = {rec["frame_index"]: rec for rec in records if rec.get("detected")}
        common = sorted(set(master_records.keys()) & set(slave_records.keys()))
        if len(common) < 5:
            summary[cam] = {"status": "insufficient_common_frames", "common_frames": len(common)}
            continue
        objpoints = [objp for _ in common]
        img_slave = [slave_records[idx]["corners"] for idx in common]
        img_master = [master_records[idx]["corners"] for idx in common]
        K_s = np.array(calib[cam]["K"], dtype=np.float64)
        K_m = np.array(calib[REFERENCE_CAMERA]["K"], dtype=np.float64)
        D_s = np.zeros(5, dtype=np.float64)
        D_m = np.zeros(5, dtype=np.float64)
        try:
            ret, *_ = cv2.stereoCalibrate(
                objpoints, img_slave, img_master,
                K_s, D_s, K_m, D_m, (IMG_W, IMG_H),
                flags=cv2.CALIB_FIX_INTRINSIC,
            )
            summary[cam] = {"common_frames": len(common), "stereo_error": float(ret)}
        except cv2.error as exc:
            summary[cam] = {"status": "stereo_calibrate_failed", "common_frames": len(common), "error": str(exc)}
    return summary


def rectify_points(points, K, R_rectify, P):
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    D = np.zeros(5, dtype=np.float64)
    rectified = cv2.undistortPoints(pts, K, D, R=R_rectify, P=P)
    return rectified.reshape(-1, 2)


def compare_rotation_directions(calib, detections):
    master_records = {rec["frame_index"]: rec for rec in detections.get(REFERENCE_CAMERA, []) if rec.get("detected")}
    K_master = np.array(calib[REFERENCE_CAMERA]["K"], dtype=np.float64)
    P = K_master.copy()
    modes = {"direct": [], "inverse": []}
    per_camera = {}
    for cam, records in detections.items():
        if cam == REFERENCE_CAMERA:
            continue
        slave_records = {rec["frame_index"]: rec for rec in records if rec.get("detected")}
        common = sorted(set(master_records.keys()) & set(slave_records.keys()))
        if not common:
            continue
        K_slave = np.array(calib[cam]["K"], dtype=np.float64)
        R_rel = np.array(calib[cam]["R_rel"], dtype=np.float64)
        per_camera[cam] = {}
        for mode, R_rectify in [("direct", R_rel), ("inverse", np.linalg.inv(R_rel))]:
            values = []
            for frame_idx in common:
                master_pts = rectify_points(master_records[frame_idx]["corners"], K_master, np.eye(3), P)
                slave_pts = rectify_points(slave_records[frame_idx]["corners"], K_slave, R_rectify, P)
                residuals = np.abs(slave_pts[:, 1] - master_pts[:, 1])
                values.extend([float(v) for v in residuals])
            stats = summarize_error_values(values)
            stats["valid_corner_count"] = len(values)
            stats["common_frames"] = len(common)
            per_camera[cam][mode] = stats
            modes[mode].extend(values)
    global_stats = {mode: summarize_error_values(values) for mode, values in modes.items()}
    recommendation = recommend_rotation_mode(global_stats)
    return {"global": global_stats, "per_camera": per_camera, "recommendation": recommendation}


def summarize_error_values(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def recommend_rotation_mode(global_stats):
    direct = global_stats.get("direct", {})
    inverse = global_stats.get("inverse", {})
    direct_p95 = direct.get("p95")
    inverse_p95 = inverse.get("p95")
    if direct_p95 is None or inverse_p95 is None:
        return {"mode": "unknown", "reason": "insufficient_data"}
    if direct_p95 <= inverse_p95 * 0.8:
        return {"mode": "direct", "reason": "direct_p95_at_least_20_percent_lower"}
    if inverse_p95 <= direct_p95 * 0.8:
        return {"mode": "inverse", "reason": "inverse_p95_at_least_20_percent_lower"}
    return {"mode": "no_change_recommended", "reason": "p95_difference_below_20_percent"}


def build_detection_summary(detections, sampled_count):
    summary = {}
    for cam, records in detections.items():
        valid = [rec for rec in records if rec.get("detected")]
        centers = np.array([rec["center"] for rec in valid], dtype=np.float64) if valid else np.empty((0, 2))
        areas = [rec["bbox_area"] for rec in valid]
        item = {
            "sampled_frames": sampled_count,
            "detected_frames": len(valid),
            "detection_rate": float(len(valid) / sampled_count) if sampled_count else 0.0,
        }
        if len(valid):
            item.update({
                "center_x_range": [float(np.min(centers[:, 0])), float(np.max(centers[:, 0]))],
                "center_y_range": [float(np.min(centers[:, 1])), float(np.max(centers[:, 1]))],
                "bbox_area_range": [float(np.min(areas)), float(np.max(areas))],
                "orientation_score_range": [float(min(rec["orientation_score"] for rec in valid)), float(max(rec["orientation_score"] for rec in valid))],
            })
        summary[cam] = item
    return summary


def summarize_distortion(calib):
    summary = {}
    for cam, data in calib.items():
        d = np.array(data.get("D", []), dtype=np.float64).flatten()
        summary[cam] = {
            "count": int(d.size),
            "min": float(np.min(d)) if d.size else None,
            "max": float(np.max(d)) if d.size else None,
            "l2_norm": float(np.linalg.norm(d)) if d.size else None,
            "note": "record_only_not_used_in_this_validation",
        }
    return summary


def build_risk_warnings(detection_summary, mono_summary, stereo_summary, rotation_comparison):
    warnings = []
    for cam, item in detection_summary.items():
        if item["detected_frames"] < 5:
            warnings.append({"level": "fail", "camera": cam, "reason": "detected_frames_less_than_5"})
        elif item["detection_rate"] < 0.3:
            warnings.append({"level": "warning", "camera": cam, "reason": "detection_rate_below_30_percent"})
    for cam, item in mono_summary.items():
        p95 = item.get("p95")
        if p95 is not None and p95 > 2.0:
            warnings.append({"level": "warning", "camera": cam, "reason": "mono_reprojection_p95_gt_2px", "p95": p95})
    for cam, item in stereo_summary.items():
        err = item.get("stereo_error")
        if err is not None and err > 2.0:
            warnings.append({"level": "warning", "camera": cam, "reason": "stereo_error_gt_2px", "stereo_error": err})
    rec = rotation_comparison.get("recommendation", {})
    if rec.get("mode") in {"direct", "inverse"}:
        warnings.append({"level": "warning", "reason": "rotation_direction_significant_difference", "recommendation": rec})
    return warnings


def write_text_report(path, report):
    lines = []
    lines.append("=== Calibration Quality Report ===")
    lines.append(f"batch_name: {report['batch_name']}")
    lines.append(f"input_root: {report['input_root']}")
    lines.append(f"calibration_json: {report['calibration_json']}")
    lines.append(f"target_video: {report['target_video']}")
    lines.append("")
    rec = report["rotation_direction_comparison"]["recommendation"]
    lines.append("--- R_rel direction recommendation ---")
    lines.append(f"recommended_rotation_mode: {report['recommended_rotation_mode']}")
    lines.append(f"reason: {rec.get('reason')}")
    lines.append(f"direct_global: {report['rotation_direction_comparison']['global'].get('direct')}")
    lines.append(f"inverse_global: {report['rotation_direction_comparison']['global'].get('inverse')}")
    lines.append("")
    lines.append("--- camera detection summary ---")
    for cam, item in sorted(report["camera_detection_summary"].items()):
        lines.append(f"{cam}: detected={item['detected_frames']}/{item['sampled_frames']} rate={item['detection_rate']:.3f}")
    lines.append("")
    lines.append("--- risk warnings ---")
    if report["risk_warnings"]:
        for warning in report["risk_warnings"]:
            lines.append(json.dumps(warning, ensure_ascii=False))
    else:
        lines.append("none")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate calibration quality and R_rel direction without modifying assets.")
    parser.add_argument("--config", default=None, help="Optional RealDynLFV batch YAML config.")
    parser.add_argument("--batch-name", default=None, help="Legacy named batch default when --config is not provided.")
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--calibration-json", default=None)
    parser.add_argument("--target-video", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-interval", type=float, default=None)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="0 means no cap")
    return parser.parse_args()


def main():
    global REFERENCE_CAMERA, CHECKERBOARD_COLS, CHECKERBOARD_ROWS, SQUARE_SIZE_MM, IMG_W, IMG_H

    args = parse_args()
    if args.config:
        config = load_pipeline_config(args.config)
        selected_batch = args.batch_name or batch_id(config)
        corners = config_value(config, "calibration", "checkerboard_inner_corners", default=[CHECKERBOARD_COLS, CHECKERBOARD_ROWS])
        image_size = config_value(config, "camera", "image_size", default=[IMG_W, IMG_H])
        REFERENCE_CAMERA = str(config_value(config, "camera", "reference", default=REFERENCE_CAMERA))
        CHECKERBOARD_COLS = int(corners[0])
        CHECKERBOARD_ROWS = int(corners[1])
        SQUARE_SIZE_MM = float(config_value(config, "calibration", "square_size_mm", default=SQUARE_SIZE_MM))
        IMG_W = int(image_size[0])
        IMG_H = int(image_size[1])
        input_root = Path(args.input_root or config_value(config, "paths", "synced_root"))
        calibration_json = Path(args.calibration_json or config_value(config, "paths", "calibration_json"))
        output_dir = Path(args.output_dir or config_value(config, "paths", "calibration_report_dir", default=PROJECT_ROOT / "Output" / "Calibration_Reports" / selected_batch))
        target_video = args.target_video or str(config_value(config, "calibration", "target_video", default="0001.mp4"))
        sample_interval = args.sample_interval if args.sample_interval is not None else float(config_value(config, "calibration", "sample_interval_seconds", default=1.0))
        start_time = args.start_time if args.start_time is not None else float(config_value(config, "calibration", "sample_start_seconds", default=0.1))
        max_frames = args.max_frames if args.max_frames is not None else int(config_value(config, "calibration", "max_frames", default=0))
    else:
        if args.batch_name not in DEFAULTS:
            raise SystemExit("--batch-name must be one of firstsyn/secondsyn when --config is not provided")
        selected_batch = args.batch_name
        defaults = DEFAULTS[selected_batch]
        input_root = Path(args.input_root) if args.input_root else defaults["input_root"]
        calibration_json = Path(args.calibration_json) if args.calibration_json else defaults["calibration_json"]
        output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "Output" / "Calibration_Reports" / selected_batch
        target_video = args.target_video or "0001.mp4"
        sample_interval = args.sample_interval if args.sample_interval is not None else 1.0
        start_time = args.start_time if args.start_time is not None else 0.1
        max_frames = args.max_frames if args.max_frames is not None else 0
    debug_dir = output_dir / "debug_frames"

    ensure(input_root.exists(), f"input_root 不存在：{input_root}")
    ensure(calibration_json.exists(), f"calibration_json 不存在：{calibration_json}")
    calib = load_json(calibration_json)
    ensure(REFERENCE_CAMERA in calib, f"calibration json 缺少参考相机：{REFERENCE_CAMERA}")

    master_video = find_video(input_root, REFERENCE_CAMERA, target_video)
    ensure(master_video is not None, f"找不到参考相机视频：{REFERENCE_CAMERA}/{target_video}")
    frame_indices, fps, total_frames = sample_frame_indices(master_video, start_time, sample_interval, max_frames)
    ensure(frame_indices, "没有可用采样帧")

    cap = cv2.VideoCapture(str(master_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
    ok, frame = cap.read()
    cap.release()
    ensure(ok, f"无法读取参考相机起始帧：{master_video}")
    found, master_corners_ref, _ = detect_corners_raw(frame)
    ensure(found, "参考相机起始帧未检测到棋盘格，无法建立角点方向基准")
    ref_vec_x, ref_vec_y = get_grid_vectors(master_corners_ref, CHECKERBOARD_ROWS, CHECKERBOARD_COLS)
    ensure(ref_vec_x is not None, "参考相机角点方向基准无效")

    detections = {}
    image_sizes = {}
    for cam in sorted(calib.keys()):
        video_path = find_video(input_root, cam, target_video)
        if video_path is None:
            detections[cam] = []
            continue
        records, img_size = collect_camera_detections(cam, video_path, frame_indices, ref_vec_x, ref_vec_y, debug_dir)
        detections[cam] = records
        image_sizes[cam] = img_size
        print(f"{cam}: detected {sum(1 for r in records if r.get('detected'))}/{len(records)}")

    objp = make_object_points()
    detection_summary = build_detection_summary(detections, len(frame_indices))
    mono_summary = compute_mono_errors(calib, detections, (IMG_W, IMG_H), objp)
    stereo_summary = compute_stereo_errors(calib, detections, objp)
    rotation_comparison = compare_rotation_directions(calib, detections)
    risk_warnings = build_risk_warnings(detection_summary, mono_summary, stereo_summary, rotation_comparison)

    report = {
        "schema_version": "calibration_quality_report_v1",
        "batch_name": selected_batch,
        "input_root": str(input_root),
        "calibration_json": str(calibration_json),
        "target_video": target_video,
        "checkerboard_config": {
            "cols": CHECKERBOARD_COLS,
            "rows": CHECKERBOARD_ROWS,
            "square_size_mm": SQUARE_SIZE_MM,
            "reference_camera": REFERENCE_CAMERA,
            "distortion_policy": "D_zero_for_validation_matches_current_rectification_flow",
        },
        "sampling": {
            "fps": fps,
            "total_frames": total_frames,
            "sampled_frame_count": len(frame_indices),
            "sample_interval_sec": sample_interval,
            "start_time_sec": start_time,
            "frame_indices": [int(v) for v in frame_indices],
        },
        "camera_detection_summary": detection_summary,
        "mono_reprojection_summary": mono_summary,
        "stereo_error_summary": stereo_summary,
        "rotation_direction_comparison": rotation_comparison,
        "recommended_rotation_mode": rotation_comparison["recommendation"].get("mode"),
        "distortion_coefficients_summary": summarize_distortion(calib),
        "risk_warnings": risk_warnings,
        "detections_light": {cam: strip_corners(records) for cam, records in detections.items()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "calibration_quality_report.json"
    txt_path = output_dir / "calibration_quality_report.txt"
    write_json(json_path, report)
    write_text_report(txt_path, report)
    print(f"JSON report: {json_path}")
    print(f"TXT report : {txt_path}")
    print(f"recommended_rotation_mode: {report['recommended_rotation_mode']}")


if __name__ == "__main__":
    main()
