import json
import os
import subprocess
import multiprocessing
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from pipeline_config import config_value, load_pipeline_config

# ================= 配置区域 =================
DATA_ROOT = str(PROJECT_ROOT / "data" / "raw" / "sample_batch")
OUTPUT_ROOT = str(PROJECT_ROOT / "data" / "synced" / "sample_batch")
JSON_FILE = str(PROJECT_ROOT / "metadata" / "sample_batch" / "sync_manifest.json")

# 剪辑设置
START_DELAY = 1.5    # 打板后延迟几秒开始剪 (避开手部动作)
FPS_NOMINAL = 59.94
NUM_WORKERS = 4
FFMPEG_EXECUTABLE = "ffmpeg"
FFPROBE_EXECUTABLE = "ffprobe"
SCENE_FILTER = None
# 注意：CLIP_DURATION 被移除了，现在由脚本自动计算最大时长
# ===========================================

def get_video_duration(file_path):
    """使用 ffprobe 获取视频总时长(秒)"""
    try:
        cmd = [
            FFPROBE_EXECUTABLE, '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path)
        ]
        # 增加 timeout 防止卡死
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[Warn] 无法获取时长: {file_path} ({e})")
        return 0.0

def process_single_video(args):
    """工作进程：GPU加速 + 4K原画 + 高精度剪辑"""
    cam, video_name, offset_data, master_clap_time, common_duration, out_dir = args
    
    src_path = os.path.join(DATA_ROOT, cam, video_name)
    if not os.path.exists(src_path): return f"[Skip] {cam} 不存在"
    
    # 1. 计算精确剪辑点
    FPS = FPS_NOMINAL
    FRAME_INTERVAL = 1.0 / FPS
    theoretical_start_time = master_clap_time + (offset_data['offset_ms'] / 1000.0) + START_DELAY
    
    # 最近帧对齐
    start_frame_idx = int(round(theoretical_start_time * FPS))
    actual_cut_time = start_frame_idx * FRAME_INTERVAL
    phase_error_ms = (actual_cut_time - theoretical_start_time) * 1000.0
    
    # 2. 构建输出路径
    cam_out_dir = os.path.join(out_dir, cam)
    os.makedirs(cam_out_dir, exist_ok=True)
    dst_path = os.path.join(cam_out_dir, video_name)
    
    # 3. FFmpeg GPU 加速指令
    cmd = [
        FFMPEG_EXECUTABLE, '-y',
        '-hwaccel', 'cuda',             # 硬件加速解码
        # '-hwaccel_output_format', 'cuda',
        '-ss', f"{actual_cut_time:.6f}", # 放在 -i 前是快速定位，放在后是精确对齐
        '-i', src_path,
        '-t', f"{common_duration:.6f}",
        # 移除 scale 滤镜，保持 4K
        '-c:v', 'hevc_nvenc',           # NVIDIA 硬件编码器
        '-preset', 'p7',                # p7 是 NVENC 最高画质预设 (Slowest/Best Quality)
        '-tune', 'hq',                  # 高画质微调
        '-rc', 'constqp',               # 恒定质量模式 (类似 CRF)
        '-qp', '18',                    # 18 对应视觉无损
        '-pix_fmt', 'p010le',           # 强制指定 10-bit 输出格式
        '-an',                          # 移除音频 (科研数据集通常不需要音频，减小体积)
        dst_path
    ]
    
    try:
        # 运行并捕获错误
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 4. 写入元数据 (保持不变)
        meta = {
            "source_video": video_name,
            "camera": cam,
            "fps": FPS,
            "sync_info": {
                "master_clap_time": master_clap_time,
                "offset_ms": offset_data['offset_ms'],
                "phase_error_ms": phase_error_ms
            },
            "clip_info": {
                "start_time": actual_cut_time,
                "duration": common_duration
            }
        }
        with open(dst_path.replace('.mp4', '.json'), 'w') as f:
            json.dump(meta, f, indent=4)
            
        return f"[OK] {cam} (GPU Accel)"
    except subprocess.CalledProcessError as e:
        return f"[Error] {cam}: {e.stderr.decode()}"

def apply_runtime_config(args):
    """Apply CLI/config overrides while preserving the original constants as defaults."""
    global DATA_ROOT, OUTPUT_ROOT, JSON_FILE, START_DELAY, FPS_NOMINAL
    global NUM_WORKERS, FFMPEG_EXECUTABLE, FFPROBE_EXECUTABLE, SCENE_FILTER

    config = load_pipeline_config(args.config) if args.config else {}
    DATA_ROOT = args.data_root or config_value(config, "paths", "raw_root", default=DATA_ROOT)
    OUTPUT_ROOT = args.output_root or config_value(config, "paths", "synced_root", default=OUTPUT_ROOT)
    JSON_FILE = args.sync_manifest or config_value(config, "paths", "sync_manifest", default=JSON_FILE)
    START_DELAY = args.start_delay if args.start_delay is not None else float(config_value(config, "sync", "start_delay_seconds", default=START_DELAY))
    FPS_NOMINAL = args.fps if args.fps is not None else float(config_value(config, "camera", "nominal_fps", default=FPS_NOMINAL))
    NUM_WORKERS = args.workers if args.workers is not None else int(config_value(config, "sync", "workers", default=NUM_WORKERS))
    FFMPEG_EXECUTABLE = args.ffmpeg or str(config_value(config, "tools", "ffmpeg", default=FFMPEG_EXECUTABLE))
    FFPROBE_EXECUTABLE = args.ffprobe or str(config_value(config, "tools", "ffprobe", default=FFPROBE_EXECUTABLE))
    SCENE_FILTER = args.scene


def parse_args():
    parser = argparse.ArgumentParser(description="Create synchronized MP4 clips from a sync manifest.")
    parser.add_argument("--config", default=None, help="Optional RealDynLFV batch YAML config.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--sync-manifest", default=None)
    parser.add_argument("--start-delay", type=float, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--ffprobe", default=None)
    parser.add_argument("--scene", default=None, help="Optional single scene/video stem.")
    return parser.parse_args()


def init_worker_runtime(data_root, fps, ffmpeg_executable, ffprobe_executable):
    """Propagate runtime overrides to Windows multiprocessing workers."""
    global DATA_ROOT, FPS_NOMINAL, FFMPEG_EXECUTABLE, FFPROBE_EXECUTABLE
    DATA_ROOT = data_root
    FPS_NOMINAL = fps
    FFMPEG_EXECUTABLE = ffmpeg_executable
    FFPROBE_EXECUTABLE = ffprobe_executable


def main():
    if not os.path.exists(JSON_FILE):
        print("未找到 sync_manifest.json！")
        return

    with open(JSON_FILE, 'r') as f:
        sync_data = json.load(f)

    # 检查 ffprobe
    try:
        subprocess.run([FFPROBE_EXECUTABLE, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("[Error] 未找到 ffprobe，请安装 FFmpeg！")
        return

    all_tasks = []
    print(f"=== 开始批量处理 (共 {len(sync_data)} 组) ===")
    
    # 遍历每一组视频 (例如 0001.mp4)
    for video_name, group_data in sync_data.items():
        if SCENE_FILTER is not None and Path(video_name).stem != SCENE_FILTER:
            continue
        print(f"\n>>> 分析组: {video_name}")
        
        master_time = group_data['master_clap_time']
        offsets = group_data['offsets']
        
        # --- 步骤 A: 计算该组的最大公共时长 (Intersection) ---
        valid_durations = []
        
        # 预扫描该组所有存在的视频
        for cam, off_info in offsets.items():
            src_path = os.path.join(DATA_ROOT, cam, video_name)
            if os.path.exists(src_path):
                # 获取总时长
                total_dur = get_video_duration(src_path)
                if total_dur > 0:
                    # 计算剪辑起点
                    start_t = master_time + (off_info['offset_ms']/1000.0) + START_DELAY
                    # 计算剩余可用时长
                    remaining = total_dur - start_t
                    if remaining > 0:
                        valid_durations.append(remaining)
        
        if not valid_durations:
            print(f"  [Skip] 该组没有有效视频或时长不足")
            continue
            
        # 取最小值，作为该组所有视频的统一时长
        # 减去 0.5秒 作为安全余量，防止浮点数误差导致最后一帧读取失败
        common_duration = min(valid_durations) - 0.1
        
        if common_duration < 1.0:
            print(f"  [Skip] 有效时长太短 ({common_duration:.2f}s)")
            continue
            
        print(f"  -> 统一剪辑时长: {common_duration:.2f} 秒")
        
        # --- 步骤 B: 生成任务 ---
        # out_subdir = os.path.join(OUTPUT_ROOT, video_name.replace('.MP4', '')) # 可选：按视频名分子文件夹
        out_subdir = OUTPUT_ROOT 
        
        for cam, off_info in offsets.items():
            task = (cam, video_name, off_info, master_time, common_duration, out_subdir)
            all_tasks.append(task)

    # --- 步骤 C: 并行执行 ---
    if not all_tasks:
        print("没有任务需要处理。")
        return

    # num_workers = min(12, os.cpu_count())
    num_workers = NUM_WORKERS
    print(f"\n>>> 启动 {num_workers} 个进程处理 {len(all_tasks)} 个剪辑任务...")
    
    with multiprocessing.Pool(
        num_workers,
        initializer=init_worker_runtime,
        initargs=(DATA_ROOT, FPS_NOMINAL, FFMPEG_EXECUTABLE, FFPROBE_EXECUTABLE),
    ) as pool:
        for res in pool.imap_unordered(process_single_video, all_tasks):
            print(res)

    print("\n=== 全部完成 ===")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    apply_runtime_config(parse_args())
    main()
