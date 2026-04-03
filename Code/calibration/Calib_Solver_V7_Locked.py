import cv2
import numpy as np
import json
import os
import multiprocessing
from tqdm import tqdm

# ================= 配置区域 =================
CLIPPED_VIDEO_ROOT = r"E:\5x5_LFV\2nd_synced_output" 
OUTPUT_JSON = r"D:\Project\LF_dataset\Calibration\Output\calibration_raw_stereo_locked.json"
DEBUG_IMG_DIR = r"D:\Project\LF_dataset\Calibration\Output\Debug_Images_V7"

# 标定板参数
CHECKERBOARD_COLS = 8
CHECKERBOARD_ROWS = 8
SQUARE_SIZE_MM = 25.0 

# 采样设置
TARGET_VIDEO_NAME = "0001.mp4" 
START_TIME_SEC = 0.1          
SAMPLE_INTERVAL = 1.0
MASTER_CAM = "CAM_C3"
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
    if not os.path.exists(video_path): return None
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(START_TIME_SEC * fps)
    step_frame = int(SAMPLE_INTERVAL * fps)
    frame_indices = range(start_frame, total_frames, step_frame)
    
    extracted_data = {} 
    img_size = None
    first_frame_saved = False # 每个机位只存一张调试图
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        
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
    if len(extracted_data) < 5: return None
    return cam_name, extracted_data, img_size

def main():
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
    
    print(f"正在提取数据并保存调试图至 {DEBUG_IMG_DIR}...")
    with multiprocessing.Pool(min(12, multiprocessing.cpu_count())) as pool:
        results = list(tqdm(pool.imap(worker_extract_data, tasks), total=len(tasks)))
    
    all_data = {res[0]: (res[1], res[2]) for res in results if res is not None}
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
    for cam in cam_list:
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
    main()