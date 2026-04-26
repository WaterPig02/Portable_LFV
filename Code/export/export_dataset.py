import argparse
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from asset_versioning import (
    REFERENCE_CAMERA,
    ROI_METADATA_FILENAME,
    build_lut_asset_info,
    compute_export_signature,
    normalize_roi_metadata,
    normalize_rectify_metadata,
)
from config import get_default_config
from export_metadata import build_sequence_metadata, build_view_metadata, write_json
from export_profiles import get_export_profile


# 某个视角目录下存在该文件，表示该 view 尚未完整导出。
IN_PROGRESS_FILENAME = ".in_progress"


def ensure(condition, message):
    """断言运行条件；失败时抛出 RuntimeError. / Assert a runtime condition and raise RuntimeError on failure."""
    if not condition:
        raise RuntimeError(message)


class RunLogger:
    """同时写终端和日志文件的轻量日志器。 / Lightweight logger for console and file output."""

    def __init__(self, log_path):
        """打开本次运行的日志文件。 / Open the log file for this run."""
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("a", encoding="utf-8")

    def _write(self, level, message):
        """写一条带时间戳的日志。 / Write one timestamped log line."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def info(self, message):
        """写信息日志。 / Write an info log message."""
        self._write("信息", message)

    def warn(self, message):
        """写警告日志。 / Write a warning log message."""
        self._write("警告", message)

    def error(self, message):
        """写错误日志。 / Write an error log message."""
        self._write("错误", message)

    def close(self):
        """关闭日志文件句柄。 / Close the log file handle."""
        self.handle.close()


def load_json(path):
    """读取 UTF-8 JSON 文件。 / Load a UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_time_config(path, time_key):
    """读取并规范化 time.json 中指定批次的导出片段。 / Load and normalize selected segments from time.json."""
    # time.json 只负责“同步后视频中哪些片段需要导出”，不参与同步或几何校正。
    raw = load_json(path)
    ensure(time_key in raw, f"time.json missing key: {time_key}")
    selected = raw[time_key]
    result = {}
    for scene_id, value in selected.items():
        if value == "del":
            result[scene_id] = []
            continue
        ensure(isinstance(value, list) and len(value) > 0, f"invalid time selection for {scene_id}")
        if isinstance(value[0], list):
            segments = value
        else:
            segments = [value]
        normalized = []
        for segment in segments:
            ensure(len(segment) == 2, f"invalid segment definition for {scene_id}")
            start_time = float(segment[0])
            end_time = float(segment[1])
            ensure(end_time > start_time, f"segment end must be greater than start for {scene_id}")
            normalized.append((start_time, end_time))
        result[scene_id] = normalized
    return result


def resolve_optional_path(cli_value, config_value):
    """解析可选路径，命令行参数优先。 / Resolve an optional path, preferring the CLI value."""
    if cli_value is not None:
        return cli_value
    return config_value


def resolve_required_path(name, cli_value, config_value):
    """解析必需路径，缺失时直接失败。 / Resolve a required path and fail if it is missing."""
    value = resolve_optional_path(cli_value, config_value)
    ensure(value is not None, f"missing required path: {name}")
    return value


def resolve_output_root(cli_value, config_paths, profile_name):
    """解析输出根目录，并判断是否使用 profile 根模式。 / Resolve output root and profile-root mode."""
    if cli_value is not None:
        return cli_value, False

    common_root = config_paths.get("output_root")
    if common_root:
        return common_root, False

    profile_key = f"{profile_name}_output_root"
    profile_root = config_paths.get(profile_key)
    ensure(profile_root is not None, f"missing required output root: {profile_key}")
    return profile_root, True


def resolve_max_workers(cli_value, config_runtime):
    """解析 camera worker 并发数。 / Resolve the number of parallel camera workers."""
    if cli_value is not None:
        ensure(cli_value > 0, "--max-workers must be greater than 0")
        return cli_value
    configured = int(config_runtime.get("max_workers", 4))
    ensure(configured > 0, "runtime.max_workers must be greater than 0")
    return configured


def resolve_lut_hwaccel(args, config_runtime):
    """解析 ffmpeg LUT 的硬件加速设置。 / Resolve ffmpeg LUT hardware-acceleration mode."""
    if args.disable_gpu_lut:
        return "none"
    if args.lut_hwaccel is not None:
        return args.lut_hwaccel
    return config_runtime.get("ffmpeg_lut_hwaccel", "cuda")


def load_rectification_metadata(rectify_dir):
    """读取并兼容旧版 rectification metadata。 / Load rectification metadata with legacy compatibility."""
    rectify_meta_path = Path(rectify_dir) / "rectify_meta.json"
    ensure(rectify_meta_path.exists(), f"missing rectification metadata: {rectify_meta_path}")
    rectification_meta = normalize_rectify_metadata(load_json(rectify_meta_path))
    if rectification_meta["image_size"] is None:
        first_camera_meta = next(iter(rectification_meta["cameras"].values()), None)
        ensure(first_camera_meta is not None, "rectification metadata contains no cameras")
        map_path = Path(rectify_dir) / first_camera_meta["map_file"]
        ensure(map_path.exists(), f"missing rectification map for legacy metadata inference: {map_path}")
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
    return rectification_meta


def load_roi_metadata(rectify_dir, roi_metadata_path=None):
    """读取并规范化 ROI metadata。 / Load and normalize ROI metadata."""
    roi_path = Path(roi_metadata_path) if roi_metadata_path else Path(rectify_dir) / ROI_METADATA_FILENAME
    ensure(roi_path.exists(), f"missing ROI metadata: {roi_path}")
    roi_metadata = normalize_roi_metadata(load_json(roi_path))
    ensure("common_valid_roi" in roi_metadata, "ROI metadata missing common_valid_roi")
    ensure("safe_rect_roi" in roi_metadata, "ROI metadata missing safe_rect_roi")
    ensure("final_release_crop_16_9" in roi_metadata, "ROI metadata missing final_release_crop_16_9")
    ensure("roi_asset_version" in roi_metadata, "ROI metadata missing roi_asset_version")
    return roi_metadata


def validate_assets(profile, rectification_meta, roi_metadata, lut_path):
    """校验 rectification、ROI 和 LUT 资产是否匹配。 / Validate rectification, ROI, and LUT assets."""
    ensure(rectification_meta["master_camera"] == REFERENCE_CAMERA, "reference camera mismatch in rectification metadata")
    ensure(
        roi_metadata["rectification_asset_version"] == rectification_meta["rectification_asset_version"],
        "ROI metadata and rectification assets are incompatible",
    )
    if profile["apply_color_standardization"]:
        ensure(lut_path is not None, f"{profile['profile']} export requires LUT configuration")
        ensure(Path(lut_path).exists(), f"missing LUT file: {lut_path}")


def validate_crop_policy(roi_metadata):
    """校验 ROI 与最终 16:9 裁剪框的几何关系。 / Validate ROI and final 16:9 crop geometry."""
    common_roi = roi_metadata["common_valid_roi"]
    safe_rect_roi = roi_metadata.get("safe_rect_roi", common_roi)
    final_crop = roi_metadata["final_release_crop_16_9"]
    ensure(final_crop["width"] * 9 == final_crop["height"] * 16, "ROI metadata final crop is not exact 16:9")
    ensure(safe_rect_roi["x"] >= common_roi["x"], "safe rect lies outside common ROI")
    ensure(safe_rect_roi["y"] >= common_roi["y"], "safe rect lies outside common ROI")
    ensure(
        safe_rect_roi["x"] + safe_rect_roi["width"] <= common_roi["x"] + common_roi["width"],
        "safe rect exceeds common ROI width",
    )
    ensure(
        safe_rect_roi["y"] + safe_rect_roi["height"] <= common_roi["y"] + common_roi["height"],
        "safe rect exceeds common ROI height",
    )
    ensure(final_crop["x"] >= safe_rect_roi["x"], "final crop lies outside safe_rect_roi")
    ensure(final_crop["y"] >= safe_rect_roi["y"], "final crop lies outside safe_rect_roi")
    ensure(
        final_crop["x"] + final_crop["width"] <= safe_rect_roi["x"] + safe_rect_roi["width"],
        "final crop exceeds safe_rect_roi width",
    )
    ensure(
        final_crop["y"] + final_crop["height"] <= safe_rect_roi["y"] + safe_rect_roi["height"],
        "final crop exceeds safe_rect_roi height",
    )


def validate_phase_offsets(offsets):
    """校验同步 manifest 中的参考相机相位约束。 / Validate reference-camera phase offsets in the sync manifest."""
    ensure(REFERENCE_CAMERA in offsets, "reference camera missing from sync manifest")
    ensure(float(offsets[REFERENCE_CAMERA]["offset_ms"]) == 0.0, "CAM_C3 phase_offset_ms must be 0.0")


def scene_id_from_video(video_name):
    """从视频文件名提取 scene id。 / Extract the scene id from a video filename."""
    return Path(video_name).stem


def build_segment_scene_name(scene_id, segment_index, segment_count):
    """为多片段 scene 生成稳定输出名。 / Build a stable output name for multi-segment scenes."""
    if segment_count <= 1:
        return scene_id
    return f"{scene_id}_seg{segment_index + 1:02d}"


def crop_frame(frame, roi):
    """按 ROI 裁剪图像并防止空结果。 / Crop a frame by ROI and reject empty crops."""
    x = int(roi["x"])
    y = int(roi["y"])
    w = int(roi["width"])
    h = int(roi["height"])
    cropped = frame[y : y + h, x : x + w]
    ensure(cropped.size != 0, "crop produced an empty frame")
    return cropped


def convert_to_bit_depth(frame, bit_depth):
    """转换输出位深。 / Convert frame data to the requested output bit depth."""
    if bit_depth == 8:
        if frame.dtype == np.uint8:
            return frame
        if frame.dtype == np.uint16:
            return (frame / 257.0).round().clip(0, 255).astype(np.uint8)
        return np.clip(frame, 0, 255).astype(np.uint8)
    if bit_depth == 16:
        if frame.dtype == np.uint16:
            return frame
        if frame.dtype == np.uint8:
            return frame.astype(np.uint16) * 257
        return np.clip(frame, 0, 65535).astype(np.uint16)
    raise RuntimeError(f"unsupported bit depth: {bit_depth}")


def escape_ffmpeg_filter_path(path):
    """转义 LUT 路径，供 ffmpeg filter 参数使用。 / Escape a LUT path for ffmpeg filter syntax."""
    normalized = Path(path).resolve().as_posix()
    return normalized.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def build_ffmpeg_lut_command(ffmpeg_executable, lut_path, width, height, hwaccel):
    """构建单帧 rawvideo LUT 命令。 / Build the ffmpeg command for one rawvideo LUT pass."""
    # 当前 LUT 仍按“单帧 rawvideo -> ffmpeg -> rawvideo”执行，优先保证正确性。
    command = [ffmpeg_executable, "-hide_banner", "-loglevel", "error", "-y"]
    if hwaccel and hwaccel != "none":
        command.extend(["-hwaccel", hwaccel])
    command.extend(
        [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            "-vf",
            f"lut3d=file='{escape_ffmpeg_filter_path(lut_path)}'",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
    )
    return command


def apply_lut_with_ffmpeg(image_bgr, ffmpeg_executable, lut_path, hwaccel):
    """通过 ffmpeg lut3d 对单帧应用 LUT。 / Apply the LUT to one frame through ffmpeg lut3d."""
    height, width = image_bgr.shape[:2]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    result = subprocess.run(
        build_ffmpeg_lut_command(ffmpeg_executable, lut_path, width, height, hwaccel),
        input=image_bgr.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        creationflags=creationflags,
    )
    ensure(result.returncode == 0, f"ffmpeg LUT failed: {result.stderr.decode('utf-8', errors='ignore').strip()}")
    expected_size = width * height * 3
    ensure(len(result.stdout) == expected_size, "ffmpeg LUT output size mismatch")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape((height, width, 3))


def build_output_dir(output_root, profile_name, scene_id, camera_id, profile_root_mode):
    """构建某个 camera/view 的输出目录。 / Build the output directory for one camera/view."""
    suffix = camera_id.split("_", 1)[1]
    row = suffix[0]
    col = int(suffix[1:])
    view_index = "ABCDE".index(row) * 5 + (col - 1)
    base_root = Path(output_root)
    if profile_root_mode:
        return base_root / scene_id / f"view_{view_index:02d}"
    return base_root / profile_name / scene_id / f"view_{view_index:02d}"


def open_video(path):
    """打开视频并在失败时明确报错。 / Open a video capture and fail clearly on error."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    return capture


def camera_worker_entry(task, result_queue, message_queue, stop_event):
    """执行单个 camera 的完整导出任务。 / Run the full export task for one camera."""
    # 一个 worker 只负责一个 camera：读视频、校正、裁剪、套 LUT、写图、写 view metadata。
    capture = None
    output_dir = Path(task["output_dir"])
    marker_path = output_dir / IN_PROGRESS_FILENAME
    try:
        if stop_event.is_set():
            raise InterruptedError("export interrupted before worker start")
        message_queue.put({"type": "camera_start", "camera_id": task["camera_id"], "scene_id": task["export_scene_id"]})

        map_path = Path(task["rectify_dir"]) / task["camera_meta"]["map_file"]
        ensure(map_path.exists(), f"missing rectification map: {map_path}")
        data = np.load(map_path)
        map_x = data["map_x"]
        map_y = data["map_y"]

        common_roi = task["roi_metadata"]["common_valid_roi"]
        final_crop = task["roi_metadata"]["final_release_crop_16_9"]
        final_crop_in_common = {
            "x": final_crop["x"] - common_roi["x"],
            "y": final_crop["y"] - common_roi["y"],
            "width": final_crop["width"],
            "height": final_crop["height"],
        }
        ensure(final_crop_in_common["x"] >= 0 and final_crop_in_common["y"] >= 0, "invalid crop relationship in ROI metadata")

        output_dir.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("in_progress\n", encoding="utf-8")

        capture = open_video(task["source_video_path"])
        capture.set(cv2.CAP_PROP_POS_MSEC, float(task["start_time"]) * 1000.0)

        frame_index = 0
        final_resolution = None
        while not stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                break
            current_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
            if current_msec > float(task["end_time"]) * 1000.0:
                break

            rectified = cv2.remap(frame, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
            common = crop_frame(rectified, common_roi)
            final_frame = crop_frame(common, final_crop_in_common)

            resize_target = task["profile"]["resize_target"]
            if resize_target is not None:
                target_w, target_h = resize_target
                final_frame = cv2.resize(final_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            if task["profile"]["apply_color_standardization"]:
                final_frame = apply_lut_with_ffmpeg(
                    image_bgr=final_frame,
                    ffmpeg_executable=task["ffmpeg_executable"],
                    lut_path=task["lut_path"],
                    hwaccel=task["lut_hwaccel"],
                )

            final_frame = convert_to_bit_depth(final_frame, task["profile"]["bit_depth"])
            final_resolution = [int(final_frame.shape[1]), int(final_frame.shape[0])]
            if task["profile"]["profile"] == "benchmark":
                ensure(final_resolution == [1920, 1080], "benchmark output resolution must be exactly 1920x1080")
            if task["profile"]["profile"] == "fidelity":
                ensure(final_resolution == [final_crop["width"], final_crop["height"]], "fidelity output must keep cropped resolution")

            if task["profile"]["output_format"] == "jpeg":
                output_path = output_dir / f"{frame_index + 1:06d}.jpg"
                written = cv2.imwrite(
                    str(output_path),
                    final_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, int(task["profile"]["jpeg_quality"])],
                )
            else:
                output_path = output_dir / f"{frame_index + 1:06d}.png"
                written = cv2.imwrite(str(output_path), final_frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            ensure(written, f"failed to write output frame: {output_path}")
            frame_index += 1

        if stop_event.is_set():
            raise InterruptedError("camera export interrupted")

        ensure(frame_index > 0, f"no frames exported for {task['camera_id']} from {task['source_video_path']}")

        runtime_info = {
            "max_workers": int(task["max_workers"]),
            "lut_hwaccel": task["lut_hwaccel"],
            "interrupted_export": False,
        }
        view_metadata = build_view_metadata(
            camera_id=task["camera_id"],
            source_video=task["source_video"],
            batch_name=task["batch_name"],
            profile=task["profile"],
            rectification_camera_meta=task["camera_meta"],
            phase_offset_ms=task["phase_offset_ms"],
            frame_count=frame_index,
            output_resolution=final_resolution,
            roi_asset_version=task["roi_metadata"]["roi_asset_version"],
            rectification_asset_version=task["rectification_meta"]["rectification_asset_version"],
            crop_policy_version=task["roi_metadata"]["crop_policy_version"],
            segment_info=task["segment_info"],
            runtime_info=runtime_info,
        )
        write_json(output_dir / "view_metadata.json", view_metadata)
        if marker_path.exists():
            marker_path.unlink()

        message_queue.put(
            {
                "type": "camera_done",
                "camera_id": task["camera_id"],
                "scene_id": task["export_scene_id"],
                "frame_count": frame_index,
                "output_resolution": final_resolution,
            }
        )
        result_queue.put(
            {
                "camera_id": task["camera_id"],
                "status": "success",
                "frame_count": frame_index,
                "output_resolution": final_resolution,
            }
        )
    except InterruptedError as exc:
        message_queue.put({"type": "camera_interrupted", "camera_id": task["camera_id"], "message": str(exc)})
        result_queue.put({"camera_id": task["camera_id"], "status": "interrupted", "error": str(exc)})
    except Exception as exc:
        message_queue.put(
            {
                "type": "camera_error",
                "camera_id": task["camera_id"],
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        result_queue.put(
            {
                "camera_id": task["camera_id"],
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if capture is not None:
            capture.release()


class ShutdownManager:
    """管理并清理 camera worker 进程。 / Track and clean up camera worker processes."""

    def __init__(self, timeout_sec):
        """初始化清理超时时间。 / Initialize shutdown timeout."""
        self.timeout_sec = float(timeout_sec)
        self.processes = {}

    def register(self, camera_id, process):
        """登记正在运行的 camera worker。 / Register an active camera worker."""
        self.processes[camera_id] = process

    def unregister(self, camera_id):
        """移除已经结束的 camera worker。 / Remove a finished camera worker."""
        self.processes.pop(camera_id, None)

    def shutdown(self, stop_event):
        """请求停止并强制清理残留进程。 / Request shutdown and force-clean remaining processes."""
        stop_event.set()
        deadline = time.time() + self.timeout_sec
        for process in list(self.processes.values()):
            if process.is_alive():
                process.join(timeout=max(0.0, deadline - time.time()))
        for process in list(self.processes.values()):
            if process.is_alive():
                process.terminate()
        time.sleep(0.5)
        for process in list(self.processes.values()):
            if process.is_alive():
                self.kill_process_tree(process.pid)
        for process in list(self.processes.values()):
            process.join(timeout=1)

    @staticmethod
    def kill_process_tree(pid):
        """按平台杀掉 worker 进程树。 / Kill a worker process tree by platform."""
        if pid is None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            try:
                os.kill(pid, 9)
            except OSError:
                pass


def drain_queue_items(raw_queue):
    """尽量取空 multiprocessing queue 中的现有消息。 / Drain currently available queue items."""
    drained = []
    while True:
        try:
            drained.append(raw_queue.get_nowait())
        except queue.Empty:
            break
        except (ValueError, OSError):
            break
    return drained


def process_message_queue(message_queue, logger, progress_state):
    """处理 worker 状态消息并更新进度日志。 / Process worker messages and update progress logs."""
    for message in drain_queue_items(message_queue):
        message_type = message.get("type")
        camera_id = message.get("camera_id", "unknown")
        if message_type == "camera_start":
            logger.info(f"开始处理视角 {camera_id}，场景 {message.get('scene_id')}")
        elif message_type == "camera_done":
            progress_state["done"] += 1
            logger.info(
                f"视角 {camera_id} 完成，帧数 {message.get('frame_count')}，分辨率 {message.get('output_resolution')}，"
                f"进度 {progress_state['done']}/{progress_state['total']}"
            )
        elif message_type == "camera_interrupted":
            logger.warn(f"视角 {camera_id} 被中断：{message.get('message')}")
        elif message_type == "camera_error":
            logger.error(f"视角 {camera_id} 失败：{message.get('message')}")
            if message.get("traceback"):
                logger.error(message["traceback"])

def run_camera_tasks(camera_tasks, max_workers, shutdown_timeout_sec, logger):
    """并行调度 camera 任务并汇总结果。 / Run camera tasks in parallel and collect results."""
    # 主进程统一调度 camera 级并行，并负责在 Ctrl+C 时回收整棵子进程树。
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    result_queue = ctx.Queue()
    message_queue = ctx.Queue()
    shutdown_manager = ShutdownManager(shutdown_timeout_sec)
    pending = deque(camera_tasks)
    results = {}
    clean_exit_waiting = {}
    clean_exit_grace_sec = 2.0
    progress_state = {"done": 0, "total": len(camera_tasks)}

    try:
        while pending or shutdown_manager.processes:
            while pending and len(shutdown_manager.processes) < max_workers and not stop_event.is_set():
                task = pending.popleft()
                process = ctx.Process(target=camera_worker_entry, args=(task, result_queue, message_queue, stop_event))
                process.start()
                shutdown_manager.register(task["camera_id"], process)

            process_message_queue(message_queue, logger, progress_state)

            try:
                result = result_queue.get(timeout=0.5)
                fresh_results = [result] + drain_queue_items(result_queue)
            except queue.Empty:
                fresh_results = drain_queue_items(result_queue)

            for result in fresh_results:
                camera_id = result["camera_id"]
                process = shutdown_manager.processes.get(camera_id)
                if process is not None:
                    process.join(timeout=0.1)
                    shutdown_manager.unregister(camera_id)
                clean_exit_waiting.pop(camera_id, None)
                results[camera_id] = result
                if result["status"] != "success":
                    stop_event.set()

            for camera_id, process in list(shutdown_manager.processes.items()):
                if not process.is_alive():
                    process.join(timeout=0.1)
                    shutdown_manager.unregister(camera_id)
                    if camera_id in results:
                        clean_exit_waiting.pop(camera_id, None)
                        continue
                    if process.exitcode == 0:
                        clean_exit_waiting[camera_id] = time.time()
                        continue
                    if camera_id not in results:
                        results[camera_id] = {
                            "camera_id": camera_id,
                            "status": "error",
                            "error": f"worker exited unexpectedly with code {process.exitcode}",
                        }
                        stop_event.set()

            for camera_id, exited_at in list(clean_exit_waiting.items()):
                if camera_id in results:
                    clean_exit_waiting.pop(camera_id, None)
                    continue
                if time.time() - exited_at < clean_exit_grace_sec:
                    continue
                results[camera_id] = {
                    "camera_id": camera_id,
                    "status": "error",
                    "error": "worker exited with code 0 but did not return a result",
                }
                clean_exit_waiting.pop(camera_id, None)
                stop_event.set()

            if stop_event.is_set() and shutdown_manager.processes:
                shutdown_manager.shutdown(stop_event)
                process_message_queue(message_queue, logger, progress_state)
                for result in drain_queue_items(result_queue):
                    results[result["camera_id"]] = result
                break

    except KeyboardInterrupt:
        shutdown_manager.shutdown(stop_event)
        process_message_queue(message_queue, logger, progress_state)
        for result in drain_queue_items(result_queue):
            results[result["camera_id"]] = result
        raise
    process_message_queue(message_queue, logger, progress_state)
    for result in drain_queue_items(result_queue):
        results[result["camera_id"]] = result

    for camera_id in list(clean_exit_waiting.keys()):
        if camera_id not in results:
            results[camera_id] = {
                "camera_id": camera_id,
                "status": "error",
                "error": "worker exited with code 0 but no result was collected",
            }
            clean_exit_waiting.pop(camera_id, None)

    errors = [item for item in results.values() if item["status"] == "error"]
    if errors:
        first = errors[0]
        raise RuntimeError(first.get("traceback") or first.get("error") or "camera export failed")

    interruptions = [item for item in results.values() if item["status"] == "interrupted"]
    if interruptions:
        first = interruptions[0]
        raise KeyboardInterrupt(first.get("error", "camera export interrupted"))

    try:
        result_queue.close()
    except Exception:
        pass
    try:
        message_queue.close()
    except Exception:
        pass

    ensure(len(results) == len(camera_tasks), "missing camera results after export")
    return results


def build_camera_task(
    camera_id,
    camera_meta,
    source_video,
    export_scene_id,
    source_video_path,
    profile,
    rectify_dir,
    rectification_meta,
    roi_metadata,
    lut_path,
    ffmpeg_executable,
    lut_hwaccel,
    output_dir,
    start_time,
    end_time,
    phase_offset_ms,
    batch_name,
    segment_info,
    max_workers,
):
    """打包传给 worker 的单 camera 任务配置。 / Pack one camera task payload for a worker."""
    return {
        "camera_id": camera_id,
        "camera_meta": camera_meta,
        "source_video": source_video,
        "source_video_path": str(source_video_path),
        "profile": dict(profile),
        "rectify_dir": str(rectify_dir),
        "rectification_meta": rectification_meta,
        "roi_metadata": roi_metadata,
        "lut_path": lut_path,
        "ffmpeg_executable": ffmpeg_executable,
        "lut_hwaccel": lut_hwaccel,
        "output_dir": str(output_dir),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "phase_offset_ms": float(phase_offset_ms),
        "batch_name": batch_name,
        "export_scene_id": export_scene_id,
        "segment_info": segment_info,
        "max_workers": int(max_workers),
    }


def build_run_paths(output_root, batch_name, profile_name, scene_name):
    """生成本次运行的 run id 与日志路径。 / Build run id and log paths for this export run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{profile_name}_{scene_name}_{batch_name}"
    logs_dir = Path(output_root) / "logs"
    checks_dir = Path(output_root) / "checks"
    return {
        "run_id": run_id,
        "log_path": logs_dir / f"{run_id}.log",
        "checks_root": checks_dir,
    }


def verify_sequence_outputs(camera_tasks, sequence_output_root):
    """写 sequence metadata 前检查 25 视角输出完整性。 / Verify per-view outputs before sequence metadata commit."""
    resolutions = []
    for task in camera_tasks:
        view_dir = Path(task["output_dir"])
        ensure(not (view_dir / IN_PROGRESS_FILENAME).exists(), f"视角目录仍残留 {IN_PROGRESS_FILENAME}: {view_dir}")
        ensure((view_dir / "view_metadata.json").exists(), f"缺少 view_metadata.json: {view_dir}")
        meta = load_json(view_dir / "view_metadata.json")
        resolutions.append(tuple(meta["output_resolution"]))
    ensure(len(set(resolutions)) == 1, "视角输出分辨率不一致")
    ensure(not (sequence_output_root / "metadata.json").exists(), "sequence metadata 写出前发现旧 metadata，请先清理后重跑")


def main():
    """命令行入口，组织批次导出流程。 / CLI entry point for batch export orchestration."""
    # 主入口：解析配置、组织 scene/segment/camera 任务，并负责最终提交 metadata。
    parser = argparse.ArgumentParser(description="Unified benchmark/fidelity exporter from synchronized MP4 clips.")
    parser.add_argument("--batch-name", default=None, choices=["secondsyn", "firstsyn"], help="Named batch config to use.")
    parser.add_argument("--input-root", default=None, help="Root containing synchronized MP4 clips organized by camera.")
    parser.add_argument("--sync-manifest", default=None, help="Path to sync_manifest.json.")
    parser.add_argument("--time-json", default=None, help="Path to time.json containing selected synced segments.")
    parser.add_argument("--rectify-dir", default=None, help="Directory containing rectify_meta.json and rectification maps.")
    parser.add_argument("--output-root", default=None, help="Output dataset root.")
    parser.add_argument("--profile", required=True, choices=["benchmark", "fidelity"], help="Export profile.")
    parser.add_argument("--roi-metadata", default=None, help="Optional explicit path to release_roi_metadata.json.")
    parser.add_argument("--lut-path", default=None, help="LUT path. Overrides config default when provided.")
    parser.add_argument("--scene", default=None, help="Optional single scene/video stem to export.")
    parser.add_argument("--max-workers", type=int, default=None, help="Camera workers to run in parallel.")
    parser.add_argument("--disable-gpu-lut", action="store_true", help="Disable ffmpeg LUT hardware acceleration.")
    parser.add_argument("--lut-hwaccel", choices=["cuda", "none"], default=None, help="ffmpeg LUT hardware acceleration mode.")
    args = parser.parse_args()

    config = get_default_config(args.batch_name)
    config_paths = config["paths"]
    config_environment = config["environment"]
    config_profiles = config["profiles"]
    config_runtime = config["runtime"]

    profile = get_export_profile(args.profile)
    profile_overrides = config_profiles.get(args.profile, {})
    for key, value in profile_overrides.items():
        profile[key] = value

    input_root_value = resolve_required_path("input_root", args.input_root, config_paths.get("input_root"))
    sync_manifest_value = resolve_required_path("sync_manifest", args.sync_manifest, config_paths.get("sync_manifest"))
    time_json_value = resolve_required_path("time_json", args.time_json, config_paths.get("time_json"))
    rectify_dir_value = resolve_required_path("rectify_dir", args.rectify_dir, config_paths.get("rectify_dir"))
    output_root_value, profile_root_mode = resolve_output_root(args.output_root, config_paths, args.profile)
    lut_path_value = resolve_required_path("lut_path", args.lut_path, config_paths.get("lut_path"))
    ffmpeg_executable_value = config_environment.get("ffmpeg_executable", "ffmpeg")

    roi_metadata_value = resolve_optional_path(args.roi_metadata, config_paths.get("roi_metadata"))
    if roi_metadata_value is None:
        roi_metadata_value = str(Path(rectify_dir_value) / ROI_METADATA_FILENAME)
    scene_value = args.scene if args.scene is not None else config_runtime.get("default_scene")
    time_key_value = config_runtime.get("time_key")
    batch_name_value = config_runtime.get("batch_name")
    max_workers_value = resolve_max_workers(args.max_workers, config_runtime)
    lut_hwaccel_value = resolve_lut_hwaccel(args, config_runtime)
    shutdown_timeout_sec = float(config_runtime.get("shutdown_timeout_sec", 5))
    ensure(time_key_value is not None, "missing runtime.time_key in config")
    ensure(batch_name_value is not None, "missing runtime.batch_name in config")

    rectification_meta = load_rectification_metadata(rectify_dir_value)
    roi_metadata = load_roi_metadata(rectify_dir_value, roi_metadata_value)
    validate_assets(profile, rectification_meta, roi_metadata, lut_path_value)
    validate_crop_policy(roi_metadata)
    lut_info = build_lut_asset_info(lut_path_value, "D-Log M", "Rec.709") if profile["apply_color_standardization"] else None

    sync_manifest = load_json(sync_manifest_value)
    time_config = load_time_config(time_json_value, time_key_value)
    input_root = Path(input_root_value)
    output_root = Path(output_root_value)
    run_scene_name = scene_value if scene_value is not None else "all"
    run_paths = build_run_paths(output_root, batch_name_value, profile["profile"], run_scene_name)
    logger = RunLogger(run_paths["log_path"])
    logger.info(
        f"开始导出，batch={batch_name_value}，profile={profile['profile']}，scene={run_scene_name}，"
        f"time_key={time_key_value}，max_workers={max_workers_value}，lut_hwaccel={lut_hwaccel_value}"
    )
    logger.info(f"输入根目录：{input_root}")
    logger.info(f"sync_manifest：{sync_manifest_value}")
    logger.info(f"rectify_dir：{rectify_dir_value}")
    logger.info(f"roi_metadata：{roi_metadata_value}")
    logger.info(f"LUT：{lut_path_value}")

    try:
        for source_video, group_data in sorted(sync_manifest.items()):
            current_scene_id = scene_id_from_video(source_video)
            if scene_value is not None and current_scene_id != scene_value:
                continue
            if current_scene_id not in time_config:
                continue
            selected_segments = time_config[current_scene_id]
            if not selected_segments:
                continue

            offsets = group_data.get("offsets", {})
            validate_phase_offsets(offsets)
            phase_offsets = {camera_id: float(payload["offset_ms"]) for camera_id, payload in offsets.items()}
            ensure(phase_offsets[REFERENCE_CAMERA] == 0.0, "CAM_C3 phase_offset_ms must remain 0.0")

            for segment_index, (start_time, end_time) in enumerate(selected_segments):
                export_scene_id = build_segment_scene_name(current_scene_id, segment_index, len(selected_segments))
                sequence_output_root = (output_root / export_scene_id) if profile_root_mode else (output_root / profile["profile"] / export_scene_id)
                sequence_output_root.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"开始处理场景 {export_scene_id}，源视频 {source_video}，片段 {segment_index + 1}/{len(selected_segments)}，"
                    f"时间范围 {start_time:.3f}s -> {end_time:.3f}s"
                )
                segment_info = {
                    "time_key": time_key_value,
                    "segment_index": segment_index,
                    "segment_count": len(selected_segments),
                    "start_time": start_time,
                    "end_time": end_time,
                    "source_scene_id": current_scene_id,
                }

                camera_tasks = []
                for camera_id, camera_meta in sorted(rectification_meta["cameras"].items()):
                    source_video_path = input_root / camera_id / source_video
                    ensure(source_video_path.exists(), f"missing synchronized MP4: {source_video_path}")
                    ensure(camera_id in phase_offsets, f"missing phase offset for {camera_id} in {source_video}")
                    view_dir = build_output_dir(output_root, profile["profile"], export_scene_id, camera_id, profile_root_mode)
                    camera_tasks.append(
                        build_camera_task(
                            camera_id=camera_id,
                            camera_meta=camera_meta,
                            source_video=source_video,
                            export_scene_id=export_scene_id,
                            source_video_path=source_video_path,
                            profile=profile,
                            rectify_dir=rectify_dir_value,
                            rectification_meta=rectification_meta,
                            roi_metadata=roi_metadata,
                            lut_path=lut_path_value,
                            ffmpeg_executable=ffmpeg_executable_value,
                            lut_hwaccel=lut_hwaccel_value,
                            output_dir=view_dir,
                            start_time=start_time,
                            end_time=end_time,
                            phase_offset_ms=phase_offsets[camera_id],
                            batch_name=batch_name_value,
                            segment_info=segment_info,
                            max_workers=max_workers_value,
                        )
                    )

                results = run_camera_tasks(
                    camera_tasks=camera_tasks,
                    max_workers=max_workers_value,
                    shutdown_timeout_sec=shutdown_timeout_sec,
                    logger=logger,
                )

                final_output_resolution = None
                for camera_id in sorted(results.keys()):
                    output_resolution = results[camera_id]["output_resolution"]
                    if final_output_resolution is None:
                        final_output_resolution = output_resolution
                    else:
                        ensure(final_output_resolution == output_resolution, "inconsistent output resolution across cameras")

                ensure(final_output_resolution is not None, f"no cameras exported for {source_video} segment {segment_index}")
                verify_sequence_outputs(camera_tasks, sequence_output_root)
                runtime_info = {
                    "max_workers": max_workers_value,
                    "lut_hwaccel": lut_hwaccel_value,
                    "interrupted_export": False,
                    "run_id": run_paths["run_id"],
                    "run_log": str(run_paths["log_path"]),
                }
                sequence_metadata = build_sequence_metadata(
                    scene_id=export_scene_id,
                    source_video=source_video,
                    batch_name=batch_name_value,
                    profile=profile,
                    roi_metadata=roi_metadata,
                    rectification_meta=rectification_meta,
                    phase_offsets=phase_offsets,
                    final_output_resolution=final_output_resolution,
                    lut_info=lut_info,
                    source_sync_manifest=str(Path(sync_manifest_value)),
                    segment_info=segment_info,
                    runtime_info=runtime_info,
                )
                write_json(sequence_output_root / "metadata.json", sequence_metadata)
                logger.info(f"场景 {export_scene_id} 完成，sequence metadata 已写出：{sequence_output_root / 'metadata.json'}")
    except KeyboardInterrupt:
        logger.warn("已收到 Ctrl+C，导出已中断，子进程清理完成。请检查残留的 .in_progress 并按需清理后重跑。")
        logger.close()
        raise SystemExit(130)
    except Exception as exc:
        logger.error(f"导出失败：{exc}")
        logger.error(traceback.format_exc())
        logger.error(f"详细日志见：{run_paths['log_path']}")
        logger.close()
        raise
    logger.info(f"导出结束，运行日志：{run_paths['log_path']}")
    logger.close()


if __name__ == "__main__":
    main()
