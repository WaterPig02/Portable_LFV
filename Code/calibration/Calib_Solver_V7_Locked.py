import cv2
import numpy as np
import json
import os
import multiprocessing
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from pipeline_config import config_value, load_pipeline_config

# ================= 配置区域 =================
CLIPPED_VIDEO_ROOT = r"E:\5x5_LFV\1st_synced_output"
OUTPUT_JSON = r"D:\Project\LF_dataset\Calibration\Output\calibration_raw_stereo_locked_firstsyn.json"
DEBUG_IMG_DIR = r"D:\Project\LF_dataset\Calibration\Output\Debug_Images_V7_Firstsyn"

# 标定板参数
CHECKERBOARD_COLS = 8
CHECKERBOARD_ROWS = 8
SQUARE_SIZE_MM = 25.0 

# 采样设置
TARGET_VIDEO_NAME = "0001.mp4" 
START_TIME_SEC = 0.1          
SAMPLE_INTERVAL = 1.0
MASTER_CAM = "CAM_C3"
MAX_WORKERS = 4
# ===========================================

# --- 辅助函数：检测与对齐 ---
def detect_corners_raw(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS), flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners, gray.shape[::-1]
    return False, None, None

def get_grid_vectors(corners, rows, cols):
    grid = corners.reshape(rows, cols, 2)
    vec_x = np.mean(grid[:, 1:, :] - grid[:, :-1, :], axis=(0, 1))
    vec_y = np.mean(grid[1:, :, :] - grid[:-1, :, :], axis=(0, 1))
    return vec_x / np.linalg.norm(vec_x), vec_y / np.linalg.norm(vec_y)

def generate_8_orientations(corners, rows, cols):
    grid = corners.reshape(rows, cols, 2)
    orientations = []
    base_grid = grid
    for flipped in [False, True]:
        current_grid = base_grid if not flipped else np.flip(base_grid, axis=1)
        for rot in range(4):
            rotated = np.rot90(current_grid, k=rot, axes=(0, 1))
            orientations.append(rotated.reshape(-1, 1, 2))
    return orientations

def align_corners_to_master(target_corners, ref_vec_x, ref_vec_y, rows, cols):
    candidates = generate_8_orientations(target_corners, rows, cols)
    best_score = -np.inf
    best_corners = None
    for cand in candidates:
        cand_vec_x, cand_vec_y = get_grid_vectors(cand, rows, cols)
        score = np.dot(cand_vec_x, ref_vec_x) + np.dot(cand_vec_y, ref_vec_y)
        if score > best_score:
            best_score = score
            best_corners = cand
    return best_corners

def save_debug_image(cam_name, image, corners):
    """保存调试图片"""
    if not os.path.exists(DEBUG_IMG_DIR):
        os.makedirs(DEBUG_IMG_DIR, exist_ok=True)
    debug_img = image.copy()
    cv2.drawChessboardCorners(debug_img, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS), corners, True)
    cv2.putText(debug_img, cam_name, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"{cam_name}_v7_debug.jpg"), debug_img)

# --- 工作进程 ---
def worker_extract_data(args):
    cam_name, video_path, ref_vec_x, ref_vec_y = args
    if not os.path.exists(video_path):
        return {"camera": cam_name, "ok": False, "reason": "video_missing", "video_path": video_path}
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return {"camera": cam_name, "ok": False, "reason": "invalid_video", "video_path": video_path}
    
    start_frame = int(START_TIME_SEC * fps)
    step_frame = int(SAMPLE_INTERVAL * fps)
    frame_indices = list(range(start_frame, total_frames, step_frame))
    
    extracted_data = {} 
    img_size = None
    first_frame_saved = False # 每个机位只存一张调试图
    sampled_frames = 0
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        sampled_frames += 1
        
        found, corners, size = detect_corners_raw(frame)
        if found:
            aligned = align_corners_to_master(corners, ref_vec_x, ref_vec_y, CHECKERBOARD_ROWS, CHECKERBOARD_COLS)
            extracted_data[idx] = aligned
            img_size = size
            
            # 只有在第一帧有效时保存调试图
            if not first_frame_saved:
                save_debug_image(cam_name, frame, aligned)
                first_frame_saved = True
            
    cap.release()
    result = {
        "camera": cam_name,
        "ok": len(extracted_data) >= 5,
        "reason": "ok" if len(extracted_data) >= 5 else "too_few_detections",
        "data": extracted_data,
        "img_size": img_size,
        "sampled_frames": sampled_frames,
        "detected_frames": len(extracted_data),
        "total_frames": total_frames,
        "fps": fps,
        "video_path": video_path,
    }
    return result

def init_worker_runtime(start_time, sample_interval, checkerboard_cols, checkerboard_rows, debug_img_dir):
    """Propagate runtime overrides to Windows multiprocessing workers."""
    global START_TIME_SEC, SAMPLE_INTERVAL, CHECKERBOARD_COLS, CHECKERBOARD_ROWS, DEBUG_IMG_DIR
    START_TIME_SEC = start_time
    SAMPLE_INTERVAL = sample_interval
    CHECKERBOARD_COLS = checkerboard_cols
    CHECKERBOARD_ROWS = checkerboard_rows
    DEBUG_IMG_DIR = debug_img_dir


def run_extract_pool(tasks):
    """并行提取角点；Ctrl+C 时强制清理 worker，避免残留进程继续占满 CPU。"""
    num_workers = min(MAX_WORKERS, multiprocessing.cpu_count(), len(tasks))
    print(f"正在提取角点：{len(tasks)} 个机位，workers={num_workers}，调试图目录={DEBUG_IMG_DIR}")
    pool = multiprocessing.Pool(
        num_workers,
        initializer=init_worker_runtime,
        initargs=(START_TIME_SEC, SAMPLE_INTERVAL, CHECKERBOARD_COLS, CHECKERBOARD_ROWS, DEBUG_IMG_DIR),
    )
    results = []
    try:
        with tqdm(total=len(tasks), desc="camera extraction", unit="cam") as progress:
            for result in pool.imap_unordered(worker_extract_data, tasks):
                results.append(result)
                cam = result.get("camera", "UNKNOWN")
                detected = result.get("detected_frames", 0)
                sampled = result.get("sampled_frames", 0)
                reason = result.get("reason", "")
                tqdm.write(f"[{cam}] {reason}: detected={detected}/{sampled}")
                progress.update(1)
        pool.close()
        pool.join()
        return results
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在终止标定提取 worker...")
        pool.terminate()
        pool.join()
        raise
    except Exception:
        pool.terminate()
        pool.join()
        raise

def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate the RealDynLFV 5x5 camera array.")
    parser.add_argument("--config", default=None, help="Optional RealDynLFV batch YAML config.")
    parser.add_argument("--input-root", default=None, help="Root containing synchronized camera videos.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--target-video", default=None)
    parser.add_argument("--reference-camera", default=None)
    parser.add_argument("--checkerboard-cols", type=int, default=None)
    parser.add_argument("--checkerboard-rows", type=int, default=None)
    parser.add_argument("--square-size-mm", type=float, default=None)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--sample-interval", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def apply_runtime_config(args):
    """Apply CLI/config overrides without changing calibration calculations."""
    global CLIPPED_VIDEO_ROOT, OUTPUT_JSON, DEBUG_IMG_DIR, TARGET_VIDEO_NAME
    global MASTER_CAM, CHECKERBOARD_COLS, CHECKERBOARD_ROWS, SQUARE_SIZE_MM
    global START_TIME_SEC, SAMPLE_INTERVAL, MAX_WORKERS

    config = load_pipeline_config(args.config) if args.config else {}
    corners = config_value(config, "calibration", "checkerboard_inner_corners", default=[CHECKERBOARD_COLS, CHECKERBOARD_ROWS])
    CLIPPED_VIDEO_ROOT = args.input_root or config_value(config, "paths", "synced_root", default=CLIPPED_VIDEO_ROOT)
    OUTPUT_JSON = args.output_json or config_value(config, "paths", "calibration_json", default=OUTPUT_JSON)
    DEBUG_IMG_DIR = args.debug_dir or config_value(config, "paths", "calibration_debug_dir", default=DEBUG_IMG_DIR)
    TARGET_VIDEO_NAME = args.target_video or str(config_value(config, "calibration", "target_video", default=TARGET_VIDEO_NAME))
    MASTER_CAM = args.reference_camera or str(config_value(config, "camera", "reference", default=MASTER_CAM))
    CHECKERBOARD_COLS = args.checkerboard_cols if args.checkerboard_cols is not None else int(corners[0])
    CHECKERBOARD_ROWS = args.checkerboard_rows if args.checkerboard_rows is not None else int(corners[1])
    SQUARE_SIZE_MM = args.square_size_mm if args.square_size_mm is not None else float(config_value(config, "calibration", "square_size_mm", default=SQUARE_SIZE_MM))
    START_TIME_SEC = args.start_time if args.start_time is not None else float(config_value(config, "calibration", "sample_start_seconds", default=START_TIME_SEC))
    SAMPLE_INTERVAL = args.sample_interval if args.sample_interval is not None else float(config_value(config, "calibration", "sample_interval_seconds", default=SAMPLE_INTERVAL))
    MAX_WORKERS = args.workers if args.workers is not None else int(config_value(config, "calibration", "workers", default=MAX_WORKERS))


def preflight():
    if MASTER_CAM != "CAM_C3":
        raise RuntimeError(f"reference camera must be CAM_C3, got {MASTER_CAM}")
    if not os.path.isdir(CLIPPED_VIDEO_ROOT):
        raise FileNotFoundError(f"synchronized video root not found: {CLIPPED_VIDEO_ROOT}")
    cameras = sorted(d for d in os.listdir(CLIPPED_VIDEO_ROOT) if d.startswith("CAM_"))
    expected = [f"CAM_{row}{col}" for row in "ABCDE" for col in range(1, 6)]
    if cameras != expected:
        raise RuntimeError(f"expected 25 camera directories CAM_A1..CAM_E5, got {len(cameras)}")
    missing = [cam for cam in expected if not os.path.isfile(os.path.join(CLIPPED_VIDEO_ROOT, cam, TARGET_VIDEO_NAME))]
    if missing:
        raise FileNotFoundError(f"target video {TARGET_VIDEO_NAME} missing for: {', '.join(missing)}")


def main():
    preflight()
    print(f"=== 开始 V7.1 锁定内参双目标定 (带可视化) ===")
    
    # 1. 建立方向基准
    master_video_path = os.path.join(CLIPPED_VIDEO_ROOT, MASTER_CAM, TARGET_VIDEO_NAME)
    cap = cv2.VideoCapture(master_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME_SEC * fps))
    _, master_frame = cap.read()
    cap.release()
    _, master_corners_ref, _ = detect_corners_raw(master_frame)
    ref_vec_x, ref_vec_y = get_grid_vectors(master_corners_ref, CHECKERBOARD_ROWS, CHECKERBOARD_COLS)

    # 2. 并行提取数据
    cam_list = sorted([d for d in os.listdir(CLIPPED_VIDEO_ROOT) if d.startswith("CAM_")])
    tasks = [(cam, os.path.join(CLIPPED_VIDEO_ROOT, cam, TARGET_VIDEO_NAME), ref_vec_x, ref_vec_y) for cam in cam_list]
    
    try:
        results = run_extract_pool(tasks)
    except KeyboardInterrupt:
        print("标定已被用户中断，worker 已清理。本次不会写出 calibration json。")
        return
    
    failed = [res for res in results if not res.get("ok")]
    if failed:
        print("\n以下机位角点提取失败或数量不足：")
        for res in failed:
            print(f"  {res.get('camera')}: {res.get('reason')} detected={res.get('detected_frames', 0)}/{res.get('sampled_frames', 0)} path={res.get('video_path')}")
    all_data = {res["camera"]: (res["data"], res["img_size"]) for res in results if res.get("ok")}
    print(f"\n角点提取完成：成功 {len(all_data)}/{len(tasks)} 个机位。")
    if MASTER_CAM not in all_data:
        print("Fatal: Master 数据缺失")
        return
    master_frames, master_size = all_data[MASTER_CAM]
    
    # 3. 准备 3D 点
    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    
    final_json = {}
    
    # --- 步骤 A: 计算 Master 的高精度内参 ---
    print(f"\n正在计算 Master ({MASTER_CAM}) 的基准内参...")
    objpoints_m = [objp for _ in master_frames]
    imgpoints_m = list(master_frames.values())
    _, K_m_fixed, D_m_fixed, _, _ = cv2.calibrateCamera(objpoints_m, imgpoints_m, master_size, None, None)
    
    final_json[MASTER_CAM] = {
        "K": K_m_fixed.tolist(), "D": D_m_fixed.tolist(),
        "R_rel": np.eye(3).tolist(), "T_rel": [0.0, 0.0, 0.0]
    }

    print("\n=== 开始联合优化 (锁定内参) ===")
    for cam in tqdm(cam_list, desc="stereo calibration", unit="cam"):
        if cam == MASTER_CAM: continue
        if cam not in all_data: continue
            
        slave_frames, slave_size = all_data[cam]
        
        # 1. 计算该相机的锁定内参
        objpoints_s_all = [objp for _ in slave_frames]
        imgpoints_s_all = list(slave_frames.values())
        _, K_s_fixed, D_s_fixed, _, _ = cv2.calibrateCamera(objpoints_s_all, imgpoints_s_all, slave_size, None, None)
        
        # 2. 寻找公共帧进行双目解算
        common_frames = sorted(list(set(master_frames.keys()) & set(slave_frames.keys())))
        if len(common_frames) < 5: continue
        
        objpoints_stereo, imgpoints_m_stereo, imgpoints_s_stereo = [], [], []
        for fid in common_frames:
            objpoints_stereo.append(objp)
            imgpoints_m_stereo.append(master_frames[fid])
            imgpoints_s_stereo.append(slave_frames[fid])
            
        # 3. 双目优化 (锁定内参)
        ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            objpoints_stereo, imgpoints_s_stereo, imgpoints_m_stereo,
            K_s_fixed, D_s_fixed, K_m_fixed, D_m_fixed, master_size,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        final_json[cam] = {
            "K": K_s_fixed.tolist(), "D": D_s_fixed.tolist(),
            "R_rel": R.tolist(), "T_rel": T.flatten().tolist()
        }
        print(f"{cam} -> Master: T=[{T[0][0]:6.1f}, {T[1][0]:6.1f}, {T[2][0]:6.1f}] | Err: {ret:.4f}")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_json, f, indent=4)
    print(f"\n结果已保存。请务必检查 {DEBUG_IMG_DIR} 确认方向一致。")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    apply_runtime_config(parse_args())
    main()
