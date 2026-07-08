import cv2
import numpy as np
import json
import os
import hashlib
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from pipeline_config import config_value, load_pipeline_config

# ================= 配置区域 =================
BATCH_NAME = "firstsyn"
INPUT_JSON = rf"D:\Project\LF_dataset\Calibration\Output\calibration_raw_stereo_locked_{BATCH_NAME}.json"
OUTPUT_MAP_DIR = rf"D:\Project\LF_dataset\Calibration\Output\Rectify_Maps\{BATCH_NAME}"
MASTER_CAM = "CAM_C3"

# 图像分辨率
IMG_W, IMG_H = 3840, 2160
# 缩放系数：1.0 表示保留所有有效像素（会有黑边），0.0 表示强制裁剪掉所有黑边（画面会放大）
# 针对你的需求，建议设为 0.0，这样可以得到干净的共同画面
CROP_ALPHA = 0.0 
# ===========================================


def compute_rectification_asset_version(raw_calib):
    payload = {
        "master_cam": MASTER_CAM,
        "image_size": [IMG_W, IMG_H],
        "crop_alpha": CROP_ALPHA,
        "raw_calib": raw_calib,
    }
    raw_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw_bytes).hexdigest()[:16]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate RealDynLFV rectification maps.")
    parser.add_argument("--config", default=None, help="Optional RealDynLFV batch YAML config.")
    parser.add_argument("--batch-name", default=None)
    parser.add_argument("--calibration-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reference-camera", default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--crop-alpha", type=float, default=None)
    return parser.parse_args()


def apply_runtime_config(args):
    """Apply CLI/config overrides while preserving the existing map algorithm."""
    global BATCH_NAME, INPUT_JSON, OUTPUT_MAP_DIR, MASTER_CAM, IMG_W, IMG_H, CROP_ALPHA
    config = load_pipeline_config(args.config) if args.config else {}
    image_size = config_value(config, "camera", "image_size", default=[IMG_W, IMG_H])
    BATCH_NAME = args.batch_name or str(config_value(config, "batch_name", default=BATCH_NAME))
    INPUT_JSON = args.calibration_json or config_value(
        config, "paths", "calibration_json",
        default=rf"D:\Project\LF_dataset\Calibration\Output\calibration_raw_stereo_locked_{BATCH_NAME}.json",
    )
    OUTPUT_MAP_DIR = args.output_dir or config_value(
        config, "paths", "rectification_dir",
        default=rf"D:\Project\LF_dataset\Calibration\Output\Rectify_Maps\{BATCH_NAME}",
    )
    MASTER_CAM = args.reference_camera or str(config_value(config, "camera", "reference", default=MASTER_CAM))
    IMG_W = args.image_width if args.image_width is not None else int(image_size[0])
    IMG_H = args.image_height if args.image_height is not None else int(image_size[1])
    CROP_ALPHA = args.crop_alpha if args.crop_alpha is not None else float(config_value(config, "rectification", "crop_alpha", default=CROP_ALPHA))


def preflight(raw_calib):
    if MASTER_CAM != "CAM_C3":
        raise RuntimeError(f"reference camera must be CAM_C3, got {MASTER_CAM}")
    expected = {f"CAM_{row}{col}" for row in "ABCDE" for col in range(1, 6)}
    if set(raw_calib) != expected:
        raise RuntimeError(f"calibration JSON must contain CAM_A1..CAM_E5, got {len(raw_calib)} cameras")
    if IMG_W <= 0 or IMG_H <= 0:
        raise ValueError(f"invalid image size: {IMG_W}x{IMG_H}")


def main():
    if not os.path.exists(INPUT_JSON):
        print(f"错误：找不到文件 {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r') as f:
        raw_calib = json.load(f)

    preflight(raw_calib)

    os.makedirs(OUTPUT_MAP_DIR, exist_ok=True)
    rectification_asset_version = compute_rectification_asset_version(raw_calib)
    
    # 1. 预计算：找到 25 个机位的“最小公约 ROI”
    print("正在计算全阵列共同视场...")
    all_rois = []
    
    # 我们先为每个相机计算它在“理想旋转”下能看到的区域
    for cam in raw_calib:
        K = np.array(raw_calib[cam]['K'])
        D = np.zeros(5) # 依然强制畸变归零
        R_rel = np.array(raw_calib[cam]['R_rel'])
        R_rectify = np.linalg.inv(R_rel)
        
        # 获取该相机在校正后的最优内参和 ROI
        # 这个 ROI 告诉我们校正后的图像里，哪些像素是有内容的（非黑边）
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (IMG_W, IMG_H), CROP_ALPHA, (IMG_W, IMG_H))
        all_rois.append(roi)

    # 计算所有机位 ROI 的交集（Intersection）
    # ROI 格式: (x, y, w, h)
    roi_arr = np.array(all_rois)
    # 取所有左上角的最大值，所有右下角的最小值
    final_x = np.max(roi_arr[:, 0])
    final_y = np.max(roi_arr[:, 1])
    final_w = np.min(roi_arr[:, 0] + roi_arr[:, 2]) - final_x
    final_h = np.min(roi_arr[:, 1] + roi_arr[:, 3]) - final_y

    print(f"  [Info] 最大公共矩形选定为: x={final_x}, y={final_y}, w={final_w}, h={final_h}")

    # 2. 构造一个全局统一的虚拟相机矩阵 P
    # 我们以 Master 的 K 为基础，并把中心点偏移到我们选定的公共矩形中心
    K_master = np.array(raw_calib[MASTER_CAM]['K'])
    P_final = K_master.copy()
    
    # 调整焦距，使得公共矩形填满 4K 画布
    scale_x = IMG_W / final_w
    scale_y = IMG_H / final_h
    scale = min(scale_x, scale_y) # 保持纵横比
    
    P_final[0, 0] *= scale
    P_final[1, 1] *= scale
    # 重新定位中心点
    P_final[0, 2] = (P_final[0, 2] - final_x) * scale
    P_final[1, 2] = (P_final[1, 2] - final_y) * scale

    # 3. 生成 25 路映射表
    rectify_meta = {}
    for cam in sorted(raw_calib.keys()):
        K = np.array(raw_calib[cam]['K'])
        R_rel = np.array(raw_calib[cam]['R_rel'])
        R_rectify = np.linalg.inv(R_rel)

        # 这里的 P_final 就是我们要的“强制转回来”后的新视野
        map_x, map_y = cv2.initUndistortRectifyMap(
            K, np.zeros(5), R_rectify, P_final, (IMG_W, IMG_H), cv2.CV_32FC1
        )

        map_filename = f"{cam}_rect_map.npz"
        np.savez_compressed(os.path.join(OUTPUT_MAP_DIR, map_filename), map_x=map_x, map_y=map_y)

        rectify_meta[cam] = {
            "map_file": map_filename,
            "new_K": P_final.tolist(),
            "applied_scale": scale,
            "image_size": [IMG_W, IMG_H],
            "rectification_asset_version": rectification_asset_version
        }
        print(f"  [Done] {cam}: 视轴已对齐并重采样。")

    output_payload = {
        "schema_version": "rectification_meta_v2",
        "master_camera": MASTER_CAM,
        "image_size": [IMG_W, IMG_H],
        "crop_alpha": CROP_ALPHA,
        "rectification_asset_version": rectification_asset_version,
        "cameras": rectify_meta
    }
    with open(os.path.join(OUTPUT_MAP_DIR, "rectify_meta.json"), "w") as f:
        json.dump(output_payload, f, indent=4)

    print(f"\n全阵列校正映射表已更新。请运行 Apply_Rectification 查看效果。")

if __name__ == "__main__":
    apply_runtime_config(parse_args())
    main()
