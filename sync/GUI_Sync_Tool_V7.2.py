import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import sounddevice as sd
import scipy.signal as signal
import subprocess
from pathlib import Path

# ================= 配置区域 =================
DATA_ROOT = r"E:\5x5_LFV\2026-01-22_第二次拍摄户外场景"
MASTER_CAM = "CAM_C3"
SAMPLE_RATE = 48000
ANALYZE_SEC = 10
# ===========================================

class InteractiveSyncTool:
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.master_dir = self.root / MASTER_CAM
        self.video_list = sorted(glob.glob(os.path.join(self.master_dir, "*.MP4")))
        self.cam_folders = [f"CAM_{c}{n}" for c in "ABCDE" for n in range(1, 6)]
        self.sync_results = {}
        
        self.fig, self.axes = None, None
        self.all_audios = {}
        self.all_peaks = {}       # 存储绝对峰值位置 (Index)
        self.current_offsets = {} # 关键修复：存储当前的显示偏移量 (Seconds)
        
        self.master_template = None
        self.master_template_region = None
        self.span_selectors = []
        self.manually_corrected_cams = set()

    def extract_audio(self, video_path):
        cmd = ['ffmpeg', '-v', 'error', '-t', str(ANALYZE_SEC), '-i', str(video_path), '-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 'f32le', '-']
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            raw, _ = process.communicate()
            return np.frombuffer(raw, dtype=np.float32)
        except Exception: return None

    def on_select(self, min_pos, max_pos, cam_name):
        audio = self.all_audios.get(cam_name)
        if audio is None: return
        
        # ============================================================
        # 【核心修复】坐标逆变换
        # 屏幕看到的 X 轴是“对齐后的相对时间”，必须还原为“绝对时间”才能去查音频
        # ============================================================
        current_offset = self.current_offsets.get(cam_name, 0.0)
        
        # 还原绝对时间范围
        abs_min_time = min_pos + current_offset
        abs_max_time = max_pos + current_offset
        
        # 转换为采样点索引
        idx_min = max(0, int(abs_min_time * SAMPLE_RATE))
        idx_max = min(len(audio), int(abs_max_time * SAMPLE_RATE))
        
        if idx_max - idx_min < 50: return # 选区太小忽略
        
        # 播放这段（还原后的）绝对音频
        sd.play(audio[idx_min:idx_max], SAMPLE_RATE)
        
        if cam_name == MASTER_CAM:
            self.master_template_region = (idx_min, idx_max)
            self.master_template = audio[idx_min:idx_max]
            print(f"Master 模板更新，重新对齐未锁定机位...")
            self.auto_align_all()
        else:
            # 手动修正逻辑
            region = audio[idx_min:idx_max]
            # 在框选范围内找最大值（即使回声更高，只要你不框回声，它就找不到）
            local_peak = np.argmax(np.abs(region))
            
            self.all_peaks[cam_name] = idx_min + local_peak
            self.manually_corrected_cams.add(cam_name) # 锁定
            print(f"锁定 {cam_name}: 绝对峰值 {self.all_peaks[cam_name]/SAMPLE_RATE:.3f}s")
        
        self.update_plots()

    def on_click(self, event):
        """右键点击红点附近播放"""
        if event.button != 3 or event.inaxes is None: return
        
        ax = event.inaxes
        cam_name = ax.get_ylabel()
        audio = self.all_audios.get(cam_name)
        peak_idx = self.all_peaks.get(cam_name)
        
        if audio is None or peak_idx is None: return
        
        # 播放红点前后 0.2 秒
        start_idx = max(0, int(peak_idx - 0.2 * SAMPLE_RATE))
        end_idx = min(len(audio), int(peak_idx + 0.2 * SAMPLE_RATE))
        
        sd.play(audio[start_idx:end_idx], SAMPLE_RATE)

    def on_key(self, event):
        if event.key == 'enter':
            plt.close(self.fig)
        elif event.key == 'escape':
            self.all_peaks = None
            plt.close(self.fig)

    def auto_align_all(self):
        if self.master_template is None: return
        
        master_local_peak = np.argmax(np.abs(self.master_template))
        master_abs_peak = self.master_template_region[0] + master_local_peak
        self.all_peaks[MASTER_CAM] = master_abs_peak
        
        template_norm = (self.master_template - np.mean(self.master_template)) / np.std(self.master_template)
        
        for cam in self.cam_folders:
            # 跳过已手动锁定的机位
            if cam in self.manually_corrected_cams or cam == MASTER_CAM:
                continue
                
            target_audio = self.all_audios.get(cam)
            if target_audio is None: continue
            
            correlation = signal.correlate(target_audio, template_norm, mode='valid', method='fft')
            coarse_match_start = np.argmax(correlation)
            
            search_region = target_audio[coarse_match_start : coarse_match_start + len(self.master_template)]
            if len(search_region) == 0: continue
            
            local_peak_in_target = np.argmax(np.abs(search_region))
            self.all_peaks[cam] = coarse_match_start + local_peak_in_target

    def update_plots(self):
        master_peak_idx = self.all_peaks.get(MASTER_CAM)
        if master_peak_idx is None: return
        
        master_peak_time = master_peak_idx / SAMPLE_RATE
        
        for i, cam in enumerate(self.cam_folders):
            ax = self.axes[i]
            ax.clear()
            
            # 锁定状态显示为蓝色背景
            if cam in self.manually_corrected_cams:
                ax.set_facecolor('#e6f2ff')
            
            ax.set_ylabel(cam, rotation=0, labelpad=35, verticalalignment='center', horizontalalignment='right')
            ax.set_yticks([])
            
            audio = self.all_audios.get(cam)
            if audio is None: continue

            time_axis = np.arange(len(audio)) / SAMPLE_RATE
            
            peak_idx = self.all_peaks.get(cam)
            
            if peak_idx is not None:
                # 计算偏移量：偏移量 = 从机绝对时间 - 主机绝对时间
                # 绘图逻辑：X_visual = T_absolute - Offset
                # 这样所有机位的波峰都会画在 X_visual = Master_Time 的位置
                offset = (peak_idx / SAMPLE_RATE) - master_peak_time
                self.current_offsets[cam] = offset # 【关键】存储偏移量供 on_select 使用
                
                ax.plot(time_axis - offset, audio, color='black', linewidth=0.7)
                
                # 画红点 (应该对齐在 Master 时间线上)
                ax.plot(master_peak_time, audio[peak_idx], 'ro', markersize=4)
            else:
                ax.plot(time_axis, audio, color='gray', linewidth=0.7)
                self.current_offsets[cam] = 0.0

            ax.axvline(x=master_peak_time, color='red', linestyle='--', alpha=0.7)
            # 锁定视图范围在波峰前后 1.5 秒
            ax.set_xlim(master_peak_time - 1.5, master_peak_time + 1.5)
            
            if i < len(self.cam_folders) - 1:
                ax.set_xticks([])
        
        self.fig.canvas.draw_idle()

    def process_video(self, video_path):
        video_name = os.path.basename(video_path)
        print(f"\n>>> 正在加载视频组: {video_name}")
        
        self.all_audios.clear()
        self.all_peaks.clear()
        self.current_offsets.clear()
        self.manually_corrected_cams.clear()
        self.master_template = None
        
        for cam in self.cam_folders:
            v_path = self.root / cam / video_name
            if v_path.exists():
                self.all_audios[cam] = self.extract_audio(v_path)
        
        self.fig, self.axes = plt.subplots(len(self.cam_folders), 1, figsize=(16, 20), sharex=True)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
        self.fig.canvas.manager.set_window_title(f"Interactive Sync Tool V8: {video_name}")
        self.fig.suptitle(f"Left-Click: Mark/Correct | Right-Click: Play Peak | Scroll: Zoom | Enter: Accept", fontsize=12)
        
        self.span_selectors.clear()
        for i, cam in enumerate(self.cam_folders):
            # 绑定回调
            span = SpanSelector(self.axes[i], lambda mi, ma, c=cam: self.on_select(mi, ma, c), 'horizontal', useblit=True, props=dict(alpha=0.3, facecolor='cyan'))
            self.span_selectors.append(span)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # 初始空白绘图
        for i, cam in enumerate(self.cam_folders):
            ax = self.axes[i]
            ax.set_ylabel(cam, rotation=0, labelpad=35, verticalalignment='center', horizontalalignment='right')
            ax.set_yticks([])
            if i < len(self.cam_folders) - 1: ax.set_xticks([])
            
            audio = self.all_audios.get(cam)
            if audio is not None:
                time_axis = np.arange(len(audio)) / SAMPLE_RATE
                ax.plot(time_axis, audio, color='gray', linewidth=0.5)
        
        plt.xlabel("Time (seconds)")
        plt.show()
        
        if self.all_peaks:
            master_peak_idx = self.all_peaks.get(MASTER_CAM)
            if master_peak_idx is None: return
            
            master_clap_time = master_peak_idx / SAMPLE_RATE
            offsets = {}
            for cam, peak_idx in self.all_peaks.items():
                lag_samples = peak_idx - master_peak_idx
                lag_ms = (lag_samples / SAMPLE_RATE) * 1000.0
                frame_shift = int(round((lag_samples / SAMPLE_RATE) * 59.94))
                offsets[cam] = {"offset_ms": lag_ms, "frame_shift": frame_shift}
            
            self.sync_results[video_name] = {
                "master_clap_time": master_clap_time,
                "offsets": offsets
            }
            print(f"  [保存] {video_name} 同步数据已记录。")

    def run(self):
        for video_path in self.video_list:
            self.process_video(video_path)
            
        out_file = "sync_manifest.json"
        with open(out_file, 'w') as f:
            json.dump(self.sync_results, f, indent=4)
        print(f"\n=== 全部完成！同步清单已保存至 {out_file} ===")

if __name__ == "__main__":
    tool = InteractiveSyncTool(DATA_ROOT)
    tool.run()