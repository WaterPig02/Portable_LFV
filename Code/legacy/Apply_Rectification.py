import cv2
import numpy as np
import os
import glob
import time
import signal
import sys
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 输入：待校正的图片根目录
INPUT_IMAGES_DIR = r"E:\5x5_LFV\Extracted"

# 2. 指定场景：例如 "0029"，若留空 "" 则处理所有场景
TARGET_SCENE = "0029" 

# 3. 输入：校正映射表目录 (npz文件所在处)
MAPS_DIR = r"D:\Project\LF_dataset\Calibration\Output\Rectify_Maps"

# 4. 输出：校正后的图片存放位置
OUTPUT_DIR = r"E:\5x5_LFV\Rectified_Dataset"

# 5. 性能设置
NUM_PROCESSES = 5  
# ===========================================

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def process_camera_scene_task(args):
    cam_name, scene_id, input_dir, output_dir, map_path = args
    cv2.setNumThreads(0) 
    
    if not os.path.exists(map_path):
        return (f"{cam_name}/{scene_id}", 0, f"Map not found: {map_path}")

    try:
        data = np.load(map_path)
        map_x, map_y = data['map_x'], data['map_y']
    except Exception as e:
        return (f"{cam_name}/{scene_id}", 0, f"Map load error: {e}")

    # 兼容常用图片格式
    img_files = []
    for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG"]:
        img_files.extend(glob.glob(os.path.join(input_dir, ext)))
    img_files = sorted(img_files)
    
    if not img_files:
        return (f"{cam_name}/{scene_id}", 0, "No images found")

    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    for img_path in img_files:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None: continue

            # 执行校正 (LANCZOS4 插值)
            rectified_img = cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

            # 统一输出为无损 PNG，压缩级 1 兼顾速度与体积
            fname = os.path.splitext(os.path.basename(img_path))[0] + ".png"
            save_path = os.path.join(output_dir, fname)
            cv2.imwrite(save_path, rectified_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            processed_count += 1
        except Exception:
            continue

    return (f"{cam_name}/{scene_id}", processed_count, "OK")

def main():
    print(f"=== 立体校正流水线 V3 ===")
    if TARGET_SCENE:
        print(f"[模式] 靶向处理场景: {TARGET_SCENE}")
    else:
        print(f"[模式] 全自动扫描模式")

    # 1. 扫描任务
    tasks = []
    cam_folders = sorted([d for d in os.listdir(INPUT_IMAGES_DIR) if d.startswith("CAM_")])
    
    if not cam_folders:
        print(f"错误: 在 {INPUT_IMAGES_DIR} 下未找到 CAM_xx 文件夹")
        return

    print("正在构建任务清单...")
    for cam in cam_folders:
        cam_path = os.path.join(INPUT_IMAGES_DIR, cam)
        
        # 寻找子文件夹
        scenes = [s for s in os.listdir(cam_path) if os.path.isdir(os.path.join(cam_path, s))]
        
        # 过滤场景
        if TARGET_SCENE:
            if TARGET_SCENE in scenes:
                scenes_to_process = [TARGET_SCENE]
            else:
                # tqdm.write(f"  [跳过] {cam} 目录下找不到场景 {TARGET_SCENE}")
                continue
        else:
            scenes_to_process = scenes
        
        map_file = os.path.join(MAPS_DIR, f"{cam}_rect_map.npz")
        
        for scene in scenes_to_process:
            input_scene_dir = os.path.join(cam_path, scene)
            output_scene_dir = os.path.join(OUTPUT_DIR, cam, scene)
            tasks.append((cam, scene, input_scene_dir, output_scene_dir, map_file))

    if not tasks:
        print(f"没有找到满足条件的任务。请检查 TARGET_SCENE ('{TARGET_SCENE}') 是否正确。")
        return

    print(f"待处理序列总数: {len(tasks)}")

    # 2. 启动进程池
    pool = multiprocessing.Pool(processes=NUM_PROCESSES, initializer=init_worker)
    
    start_time = time.time()
    total_processed_frames = 0

    try:
        with tqdm(total=len(tasks), unit="seq", desc="Processing Progress") as pbar:
            for res in pool.imap_unordered(process_camera_scene_task, tasks):
                label, count, status = res
                if status == "OK":
                    total_processed_frames += count
                else:
                    tqdm.write(f"[!] {label} 失败: {status}")
                pbar.update(1)

    except KeyboardInterrupt:
        print("\n[!] 用户中断，正在强制停止子进程...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    pool.close()
    pool.join()

    duration = time.time() - start_time
    print(f"\n=== 处理报告 ===")
    print(f"耗时: {duration:.2f}s")
    print(f"总计校正帧数: {total_processed_frames}")
    if duration > 0:
        print(f"吞吐速度: {total_processed_frames/duration:.2f} fps")
    print(f"结果存放在: {OUTPUT_DIR}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()