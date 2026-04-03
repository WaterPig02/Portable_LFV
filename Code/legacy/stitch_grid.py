
import os
import cv2
import numpy as np
import glob
import sys
import multiprocessing

# ================= 配置区域 =================
CONFIG = {
    "input_dir": r"E:\5x5_LFV\Rectified_Dataset",      # 抽帧脚本的输出主目录
    "output_dir": "./Stitched",      # 拼接后5x5图片的输出目录
    "target_name": "0029",           # 对应的视频集名称
    "target_frames": "all",          # "all" 表示拼接文件夹内所有帧，或传入列表如 ["000001.jpg", "000002.jpg"]
    "scale_mode": "factor",          # "factor" (按倍数缩放) 或 "resolution" (指定绝对分辨率)
    "scale_factor": 0.2,            # 每张图的缩放倍数（当scale_mode="factor"时生效）
    "target_res": (640, 360),        # 每张图的绝对分辨率 (宽, 高)（当scale_mode="resolution"时生效）
    "max_workers": 8                 # 多进程加速的进程数
}
# ============================================

def get_camera_names():
    return [f"CAM_{row}{col}" for row in "ABCDE" for col in range(1, 6)]

def add_label(img, text):
    """在图片左上角添加带有黑色描边的机位文字"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (15, 40)
    scale = 1.0
    thickness = 2
    cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img

def stitch_single_frame(frame_name):
    cameras = get_camera_names()
    grid_rows = []
    
    tile_h, tile_w = CONFIG["target_res"][1], CONFIG["target_res"][0]
    
    for row_letter in "ABCDE":
        row_imgs = []
        for col_idx in range(1, 6):
            cam = f"CAM_{row_letter}{col_idx}"
            img_path = os.path.join(CONFIG["input_dir"], cam, CONFIG["target_name"], frame_name)
            
            img = None
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                
            if img is not None:
                if CONFIG["scale_mode"] == "factor":
                    img = cv2.resize(img, (0, 0), fx=CONFIG["scale_factor"], fy=CONFIG["scale_factor"])
                else:
                    img = cv2.resize(img, CONFIG["target_res"])
                tile_h, tile_w = img.shape[:2]
            else:
                img = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            
            img = add_label(img, cam)
            row_imgs.append(img)
            
        row_concat = np.hstack(row_imgs)
        grid_rows.append(row_concat)
        
    final_grid = np.vstack(grid_rows)
    
    out_dir = os.path.join(CONFIG["output_dir"], CONFIG["target_name"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"grid_{frame_name}")
    cv2.imwrite(out_path, final_grid)
    print(f"拼接完成: {out_path}")

def main():
    # 【性能核心】禁用 OpenCV 内部的多线程，防止与 Python 多进程起冲突导致 CPU 假死
    cv2.setNumThreads(0)

    if CONFIG["target_frames"] == "all":
        ref_dir = os.path.join(CONFIG["input_dir"], "CAM_A1", CONFIG["target_name"])
        if not os.path.exists(ref_dir):
            print(f"错误：参考目录不存在 {ref_dir}")
            return
        frames = [os.path.basename(f) for f in glob.glob(os.path.join(ref_dir, "*.*"))]
    else:
        frames = CONFIG["target_frames"]

    if not frames:
        print("没有找到需要拼接的帧。")
        return

    print(f"开始拼接，共计 {len(frames)} 帧，最大并发数: {CONFIG['max_workers']}")
    print("提示：随时可以按下 Ctrl+C 安全终止任务。\n")
    
    # 使用更底层的 multiprocessing Pool 以便更好地响应终端信号
    pool = multiprocessing.Pool(processes=CONFIG["max_workers"])
    
    try:
        # map_async 配合 get 设定超时，是 Python 多进程接收 Ctrl+C 的标准写法
        result = pool.map_async(stitch_single_frame, frames)
        result.get(0xFFFF)  # 设置极大超时时间，使主进程保持监听状态
    except KeyboardInterrupt:
        print("\n\n[警告] 接收到 Ctrl+C 中断信号！")
        print("正在紧急清理内存并终止所有拼接进程...")
        pool.terminate()
        pool.join()
        print("所有拼接任务已安全终止。系统资源已释放。")
        sys.exit(0)
    else:
        pool.close()
        pool.join()
        print("\n所有网格拼接任务完成！")

if __name__ == "__main__":
    main()