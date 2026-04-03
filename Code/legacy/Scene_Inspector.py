import cv2
import numpy as np
import json
import os

# ================= 配置区域 =================
# 标定参数文件 (绝对真理)
INPUT_JSON = r"D:\Project\LF_dataset\Calibration\Output\calibration_raw_stereo_locked.json"

# 你想检查的视频目录
CLIPPED_VIDEO_ROOT = r"E:\5x5_LFV\2nd_synced_output" 

# 你想检查的视频和时间点
TARGET_VIDEO_NAME = "0021.mp4"  # 改成你想看的任意场景
CHECK_TIME_SEC = 1            # 改成你想看的任意时间 (秒)

MASTER_CAM = "CAM_C3"
CAM_1 = "CAM_A5"  # 对比机位 1
CAM_2 = "CAM_D5"  # 对比机位 2

IMG_W, IMG_H = 3840, 2160
# ===========================================

def get_frame(cam_name):
    video_path = os.path.join(CLIPPED_VIDEO_ROOT, cam_name, TARGET_VIDEO_NAME)
    if not os.path.exists(video_path):
        print(f"找不到视频: {video_path}")
        return None
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, CHECK_TIME_SEC * 1000)
    ret, frame = cap.read()
    cap.release()
    return frame

def draw_grid_lines(img, color=(0, 255, 0), interval=150):
    out = img.copy()
    for y in range(0, IMG_H, interval):
        cv2.line(out, (0, y), (IMG_W, y), color, 1)
    for x in range(0, IMG_W, interval):
        cv2.line(out, (x, 0), (x, IMG_H), color, 1)
    return out

def rectify_cam(cam_name, img, calib_data, K_master):
    K = np.array(calib_data[cam_name]['K'])
    if cam_name == MASTER_CAM:
        R_rectify = np.eye(3)
    else:
        R_rectify = np.array(calib_data[cam_name]['R_rel'])
        
    map_x, map_y = cv2.initUndistortRectifyMap(K, np.zeros(5), R_rectify, K_master, (IMG_W, IMG_H), cv2.CV_32FC1)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4)

def main():
    print(f"正在加载标定参数...")
    with open(INPUT_JSON, 'r') as f:
        calib = json.load(f)

    K_master = np.array(calib[MASTER_CAM]['K'])

    print(f"正在提取 {TARGET_VIDEO_NAME} 第 {CHECK_TIME_SEC} 秒的画面...")
    img_1 = get_frame(CAM_1)
    img_2 = get_frame(CAM_2)
    
    if img_1 is None or img_2 is None: return

    print(f"正在应用立体校正...")
    rect_1 = rectify_cam(CAM_1, img_1, calib, K_master)
    rect_2 = rectify_cam(CAM_2, img_2, calib, K_master)

    rect_1_lines = draw_grid_lines(rect_1, color=(0, 255, 0), interval=150)
    rect_2_lines = draw_grid_lines(rect_2, color=(0, 0, 255), interval=150)

    scale = 0.5
    small_1 = cv2.resize(rect_1_lines, (0,0), fx=scale, fy=scale)
    small_2 = cv2.resize(rect_2_lines, (0,0), fx=scale, fy=scale)
    
    cv2.putText(small_1, f"{CAM_1} ({TARGET_VIDEO_NAME} @ {CHECK_TIME_SEC}s)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
    cv2.putText(small_2, f"{CAM_2} ({TARGET_VIDEO_NAME} @ {CHECK_TIME_SEC}s)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)

    vis_img = np.vstack((small_1, small_2))
    
    out_path = rf"D:\Project\LF_dataset\Calibration\Output\Scene_Inspect\Scene_Inspect_{TARGET_VIDEO_NAME.split('.')[0]}_{CAM_1}_{CAM_2}.jpg"
    cv2.imwrite(out_path, vis_img)
    print(f"巡检图已生成：{out_path}")
    print(f"请检查远处的静止背景（如建筑、电线杆）是否对齐。")

if __name__ == "__main__":
    main()