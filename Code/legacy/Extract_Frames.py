import os
import subprocess
import time
import sys
import json


# ================= 配置区域 =================
CONFIG = {
    # 每个 sync_key 绑定自己的输入目录和输出目录，互不干扰
    "sync": {
        "firstsyn": {
            "input_dir":  "E:/5x5_LFV/1st_synced_output",
            "output_dir": "E:/5x5_LFV/Extracted/1st_synced_output",
        },
        "secondsyn": {
            "input_dir":  "E:/5x5_LFV/2nd_synced_output",
            "output_dir": "E:/5x5_LFV/Extracted/2nd_synced_output",
        },
    },
    "sync_key": "firstsyn",  # ← 只改这里：处理 firstsyn 还是 secondsyn
    "interval": 1,           # 抽帧间隔：1为全部抽帧，2为隔1帧抽1帧，依此类推
    "format": "png",         # 图片格式：jpg 或 png
    "quality": 9,            # 画面质量（JPG: 2-31，越小越清晰；PNG: 压缩率0-9）
    "use_gpu": True,         # 是否尝试使用Nvidia GPU加速解码 (-hwaccel cuda)
    "max_workers": 4         # 同时处理的视频进程数（建议设为CPU核心数）
}
# ============================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIME_JSON = os.path.join(SCRIPT_DIR, "time.json")


def get_sync_dirs():
    """根据 sync_key 返回当前的 (input_dir, output_dir)。"""
    key = CONFIG["sync_key"]
    cfg = CONFIG["sync"][key]
    return cfg["input_dir"], cfg["output_dir"]


def load_time_config():
    """读取 time.json，返回当前 sync_key 对应的 {文件名: 时间段列表} 字典。
    时间段列表格式统一为 [[start, end], [start, end], ...]。
    "del" 条目返回空列表（跳过）。
    """
    with open(TIME_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get(CONFIG["sync_key"], {})
    result = {}

    for file_id, val in raw.items():
        if val == "del":
            result[file_id] = []          # 标记为删除，抽帧时跳过
        elif isinstance(val[0], list):
            result[file_id] = val         # 已经是多段：[[s,e],[s,e]]
        else:
            result[file_id] = [val]       # 单段 [s,e] → 包装为 [[s,e]]

    return result


def get_camera_names():
    return [f"CAM_{row}{col}" for row in "ABCDE" for col in range(1, 6)]


def build_ffmpeg_cmd(cam, file_id, start, end, segment_index=None):
    """构建 ffmpeg 命令。
    segment_index: 多段时间时的段序号（0-based），用于区分输出子目录；None 表示单段。

    输出目录结构示例：
      单段：Extracted/1nd_synced_output/CAM_A1/0002/
      多段：Extracted/1nd_synced_output/CAM_A1/0018/seg01/
            Extracted/1nd_synced_output/CAM_A1/0018/seg02/
    """
    input_dir, output_dir = get_sync_dirs()
    input_file = os.path.join(input_dir, cam, f"{file_id}.mp4")
    if not os.path.exists(input_file):
        return None, None

    if segment_index is not None:
        out_cam_dir = os.path.join(output_dir, cam, file_id, f"seg{segment_index + 1:02d}")
    else:
        out_cam_dir = os.path.join(output_dir, cam, file_id)
    os.makedirs(out_cam_dir, exist_ok=True)

    cmd = ["ffmpeg", "-y"]
    if CONFIG["use_gpu"]:
        cmd.extend(["-hwaccel", "cuda"])

    cmd.extend(["-ss", str(start), "-to", str(end), "-i", input_file])

    if CONFIG["interval"] > 1:
        interval = CONFIG["interval"]
        cmd.extend(["-vf", f"select='not(mod(n\\,{interval}))'", "-fps_mode", "vfr"])

    if CONFIG["format"].lower() == "jpg":
        cmd.extend(["-q:v", str(CONFIG["quality"])])
    elif CONFIG["format"].lower() == "png":
        cmd.extend(["-compression_level", str(CONFIG["quality"])])

    output_pattern = os.path.join(out_cam_dir, f"%06d.{CONFIG['format']}")
    cmd.append(output_pattern)
    return cmd, out_cam_dir


def build_task_list(time_config, cameras):
    """展开所有 (cam, file_id, start, end, seg_idx) 任务。"""
    tasks = []
    for file_id, segments in time_config.items():
        if not segments:
            print(f"[跳过-del] {file_id}")
            continue
        for cam in cameras:
            multi = len(segments) > 1
            for idx, (start, end) in enumerate(segments):
                seg_idx = idx if multi else None
                tasks.append((cam, file_id, start, end, seg_idx))
    return tasks


def main():
    if not os.path.exists(TIME_JSON):
        print(f"[错误] 找不到 time.json：{TIME_JSON}")
        sys.exit(1)

    input_dir, output_dir = get_sync_dirs()
    time_config = load_time_config()
    cameras = get_camera_names()
    tasks = build_task_list(time_config, cameras)

    print(f"同步版本  : {CONFIG['sync_key']}")
    print(f"输入目录  : {input_dir}")
    print(f"输出目录  : {output_dir}")
    print(f"文件数量  : {len(time_config)} 个（已过滤 del）")
    print(f"总任务数  : {len(tasks)} 个（{len(cameras)} 机位 × 文件 × 时间段）")
    print(f"最大并发数: {CONFIG['max_workers']}")
    print("提示：随时可以按下 Ctrl+C 安全终止所有任务。\n")

    active_procs = []   # [(proc, label), ...]
    task_idx = 0

    try:
        while task_idx < len(tasks) or active_procs:
            active_procs = [(p, lbl) for p, lbl in active_procs if p.poll() is None]

            while len(active_procs) < CONFIG["max_workers"] and task_idx < len(tasks):
                cam, file_id, start, end, seg_idx = tasks[task_idx]
                cmd, out_dir = build_ffmpeg_cmd(cam, file_id, start, end, seg_idx)

                seg_label = f" seg{seg_idx + 1:02d}" if seg_idx is not None else ""
                task_label = f"{cam}/{file_id}{seg_label} [{start}s~{end}s]"

                if cmd:
                    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    active_procs.append((p, task_label))
                    print(f"[{task_idx + 1}/{len(tasks)}] 处理中: {task_label}")
                else:
                    print(f"[跳过-无文件] {task_label}")

                task_idx += 1

            time.sleep(0.5)

        print("\n所有机位抽帧完成！")

    except KeyboardInterrupt:
        print("\n\n[警告] 接收到 Ctrl+C 中断信号！")
        print(f"正在紧急终止 {len(active_procs)} 个后台 FFmpeg 进程...")

        for p, _ in active_procs:
            try:
                p.terminate()
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
            except Exception:
                pass

        print("所有抽帧任务已安全终止。系统资源已释放。")
        sys.exit(0)


if __name__ == "__main__":
    main()