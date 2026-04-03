import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import get_default_config


CAMERAS = [f"CAM_{row}{col}" for row in "ABCDE" for col in range(1, 6)]
NEIGHBOR_CAMERAS = ["CAM_C2", "CAM_C4", "CAM_B3", "CAM_D3"]
WIDE_BASELINE_CAMERAS = ["CAM_A1", "CAM_E5"]


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


def sample_frame_indices(frame_count):
    if frame_count <= 0:
        return []
    return sorted({0, frame_count // 2, frame_count - 1})


def black_border_ratio(image):
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]], axis=0)
    black = np.all(border <= 2, axis=1)
    return float(np.mean(black))


def estimate_vertical_residual(ref_image, target_image):
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(ref_gray, maxCorners=300, qualityLevel=0.01, minDistance=12)
    if points is None or len(points) < 10:
        return {"count": 0, "vertical_residual_mean": None, "vertical_residual_p95": None, "vertical_residual_max": None}
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(ref_gray, target_gray, points, None)
    if next_points is None or status is None:
        return {"count": 0, "vertical_residual_mean": None, "vertical_residual_p95": None, "vertical_residual_max": None}
    # OpenCV 返回的点坐标通常是 (N, 1, 2)，这里先压平成 (N, 2)，避免索引维度错误。
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


def generate_visual_checks(scene_dir, checks_dir, logger):
    view_dirs = collect_view_dirs(scene_dir)
    ref_frames = list_frame_files(view_dirs["CAM_C3"])
    frame_indices = sample_frame_indices(len(ref_frames))
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


def run_checks(scene_dir, checks_dir, logger):
    report = {"scene_dir": str(scene_dir), "views": {}, "epipolar_checks": {}}
    view_dirs = collect_view_dirs(scene_dir)
    resolution_set = set()
    frame_count_set = set()
    for camera_id, view_dir in view_dirs.items():
        if not view_dir.exists():
            raise RuntimeError(f"缺少视角目录：{view_dir}")
        if (view_dir / ".in_progress").exists():
            raise RuntimeError(f"发现未完成标记：{view_dir / '.in_progress'}")
        metadata_path = view_dir / "view_metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(f"缺少 view_metadata.json：{metadata_path}")
        metadata = load_json(metadata_path)
        frame_files = list_frame_files(view_dir)
        if not frame_files:
            raise RuntimeError(f"视角目录内没有导出帧：{view_dir}")
        resolution_set.add(tuple(metadata["output_resolution"]))
        frame_count_set.add(len(frame_files))
        border_ratios = []
        for frame_index in sample_frame_indices(len(frame_files)):
            image = cv2.imread(str(frame_files[frame_index]), cv2.IMREAD_COLOR)
            border_ratios.append(black_border_ratio(image))
        report["views"][camera_id] = {
            "frame_count": len(frame_files),
            "output_resolution": metadata["output_resolution"],
            "black_border_ratio_samples": border_ratios,
        }
    report["resolution_consistent"] = len(resolution_set) == 1
    report["frame_count_consistent"] = len(frame_count_set) == 1
    sequence_metadata_path = scene_dir / "metadata.json"
    report["sequence_metadata_exists"] = sequence_metadata_path.exists()
    if sequence_metadata_path.exists():
        sequence_metadata = load_json(sequence_metadata_path)
        report["reference_camera"] = sequence_metadata.get("reference_camera")
        report["phase_offset_c3"] = sequence_metadata.get("phase_offset_ms", {}).get("CAM_C3")

    ref_frames = list_frame_files(view_dirs["CAM_C3"])
    for camera_id in NEIGHBOR_CAMERAS + WIDE_BASELINE_CAMERAS:
        metrics = []
        target_frames = list_frame_files(view_dirs[camera_id])
        for frame_index in sample_frame_indices(len(ref_frames)):
            ref_img = cv2.imread(str(ref_frames[frame_index]), cv2.IMREAD_COLOR)
            tgt_img = cv2.imread(str(target_frames[frame_index]), cv2.IMREAD_COLOR)
            metrics.append(estimate_vertical_residual(ref_img, tgt_img))
        report["epipolar_checks"][camera_id] = metrics

    generate_visual_checks(scene_dir, checks_dir, logger)
    return report


def main():
    parser = argparse.ArgumentParser(description="导出结果检查工具")
    parser.add_argument("--batch-name", default=None, choices=["secondsyn", "firstsyn"])
    parser.add_argument("--profile", required=True, choices=["benchmark", "fidelity"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    config = get_default_config(args.batch_name)
    output_root = Path(args.output_root or config["paths"]["output_root"])
    scene_dir = output_root / args.profile / args.scene
    checks_dir = output_root / "checks" / args.scene
    logs_dir = output_root / "logs"
    logger = CheckLogger(logs_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_check_{args.profile}_{args.scene}.log")
    logger.info(f"开始检查场景 {args.scene}，profile={args.profile}，目录={scene_dir}")
    report = run_checks(scene_dir, checks_dir, logger)
    report_path = checks_dir / "check_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)
    logger.info(f"检查完成，报告已写出：{report_path}")
    logger.close()


if __name__ == "__main__":
    main()
