import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import SpanSelector
import scipy.signal as signal
import sounddevice as sd

# ================= 配置区域 =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "sample_batch"
OUTPUT_MANIFEST = PROJECT_ROOT / "metadata" / "sample_batch" / "sync_manifest.json"
MASTER_CAM = "CAM_C3"
SAMPLE_RATE = 48000
ANALYZE_SEC = 10
FPS_NOMINAL = 59.94
OVERWRITE_EXISTING = True
ABS_WARN_MS = 300.0
ABS_FAIL_MS = 600.0
CAMERA_MAD_K = 6.0
# ===========================================


def get_chinese_font():
    """选择可用中文字体，避免 Matplotlib 图内中文显示为方块。"""
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Source Han Sans SC"]
    available = {font.name: font.fname for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return FontProperties(fname=available[name])
    plt.rcParams["axes.unicode_minus"] = False
    return None


CJK_FONT = get_chinese_font()

def median_abs_deviation(values):
    """计算 MAD，用于判断同一机位跨场景离群。"""
    if not values:
        return 0.0
    median = float(np.median(values))
    return float(np.median([abs(value - median) for value in values]))


def build_review_camera_map(sync_results, abs_warn_ms=ABS_WARN_MS, abs_fail_ms=ABS_FAIL_MS, camera_mad_k=CAMERA_MAD_K):
    """从已有 manifest 中找出建议复查的 scene/camera。"""
    review_map = {}
    rows = []
    for scene, payload in sync_results.items():
        for cam, info in payload.get("offsets", {}).items():
            try:
                offset_ms = float(info["offset_ms"])
            except Exception:
                review_map.setdefault(scene, set()).add(cam)
                continue
            row = {
                "scene": scene,
                "camera": cam,
                "offset_ms": offset_ms,
                "abs_offset_ms": abs(offset_ms),
                "flags": [],
            }
            rows.append(row)
            if abs(offset_ms) >= abs_fail_ms:
                row["flags"].append("abs_fail")
            elif abs(offset_ms) >= abs_warn_ms:
                row["flags"].append("abs_warn")

    by_camera = {f"CAM_{row}{col}": [] for row in "ABCDE" for col in range(1, 6)}
    for row in rows:
        by_camera.setdefault(row["camera"], []).append(row["offset_ms"])

    camera_stats = {}
    for cam, values in by_camera.items():
        if not values:
            continue
        median = float(np.median(values))
        mad = median_abs_deviation(values)
        robust_sigma = max(mad * 1.4826, 1e-6)
        camera_stats[cam] = (median, robust_sigma)

    for row in rows:
        median, robust_sigma = camera_stats.get(row["camera"], (0.0, 1.0))
        if abs(row["offset_ms"] - median) / robust_sigma >= camera_mad_k:
            row["flags"].append("camera_unstable_outlier")
        if row["flags"]:
            review_map.setdefault(row["scene"], set()).add(row["camera"])

    return review_map, rows


class InteractiveSyncTool:
    """交互式 5x5 视频同步工具，保留原同步算法，只优化交互和保存安全性。"""

    def __init__(self, root_dir, output_manifest, review_mode=False, review_cameras=None):
        self.root = Path(root_dir)
        self.output_manifest = Path(output_manifest)
        self.review_mode = review_mode
        self.review_cameras = review_cameras or {}
        self.master_dir = self.root / MASTER_CAM
        self.cam_folders = [f"CAM_{row}{col}" for row in "ABCDE" for col in range(1, 6)]
        self.video_list = []
        self.sync_results = {}
        self.completed_count_at_start = 0

        self.fig = None
        self.axes = None
        self.footer_text = None
        self.span_selectors = []

        self.current_video_name = None
        self.current_index = 0
        self.current_action = "quit"
        self.stop_requested = False

        self.all_audios = {}
        self.all_peaks = {}
        self.current_offsets = {}
        self.missing_cams = set()
        self.audio_failed_cams = set()
        self.manually_corrected_cams = set()
        self.current_review_cams = set()
        self.master_template = None
        self.master_template_region = None
        self.current_xlim = None

    def ensure_environment(self):
        """启动前检查路径、ffmpeg 和 master 目录。"""
        if not self.root.exists():
            raise FileNotFoundError(f"原始视频根目录不存在：{self.root}")
        if not self.master_dir.exists():
            raise FileNotFoundError(f"找不到参考机位目录：{self.master_dir}")
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as exc:
            raise RuntimeError("找不到可用的 ffmpeg，请先确认 ffmpeg 已加入 PATH。") from exc

        videos = list(self.master_dir.glob("*.MP4")) + list(self.master_dir.glob("*.mp4"))
        self.video_list = sorted(set(videos), key=lambda p: p.name.lower())
        if not self.video_list:
            raise FileNotFoundError(f"参考机位目录中没有 MP4 文件：{self.master_dir}")
        if self.review_mode:
            review_names = {name.lower() for name in self.review_cameras.keys()}
            self.video_list = [path for path in self.video_list if path.name.lower() in review_names]
            if not self.video_list:
                raise RuntimeError("复查模式未找到需要打开的视频组。请先确认 manifest 中存在可疑 offset。")

    def load_existing_manifest(self):
        """加载已存在的 manifest，用于中断后恢复。"""
        if not self.output_manifest.exists():
            return
        with self.output_manifest.open("r", encoding="utf-8") as handle:
            self.sync_results = json.load(handle)
        self.completed_count_at_start = len(self.sync_results)
        print(f"[恢复] 已加载 {self.completed_count_at_start} 个已完成视频组：{self.output_manifest}")

    def save_manifest(self):
        """安全写入 manifest；已有文件先备份为 .bak。"""
        self.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        if self.output_manifest.exists():
            backup_path = self.output_manifest.with_suffix(self.output_manifest.suffix + ".bak")
            shutil.copy2(self.output_manifest, backup_path)
        with self.output_manifest.open("w", encoding="utf-8") as handle:
            json.dump(self.sync_results, handle, indent=4, ensure_ascii=False)

    def find_camera_video(self, cam, video_name):
        """查找某机位对应视频，兼容 .MP4 和 .mp4。"""
        direct = self.root / cam / video_name
        if direct.exists():
            return direct
        lower = self.root / cam / Path(video_name).with_suffix(".mp4").name
        if lower.exists():
            return lower
        upper = self.root / cam / Path(video_name).with_suffix(".MP4").name
        if upper.exists():
            return upper
        return None

    def scan_missing_cameras(self, video_name):
        """扫描当前视频组缺失的机位文件。"""
        missing = []
        for cam in self.cam_folders:
            if self.find_camera_video(cam, video_name) is None:
                missing.append(cam)
        return missing

    def extract_audio(self, video_path):
        """用 ffmpeg 提取前 ANALYZE_SEC 秒单声道音频。"""
        cmd = [
            "ffmpeg", "-v", "error", "-t", str(ANALYZE_SEC), "-i", str(video_path),
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
        ]
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw, err = process.communicate()
            if process.returncode != 0 or not raw:
                message = err.decode("utf-8", errors="ignore").strip()
                print(f"[音频失败] {video_path}: {message}")
                return None
            return np.frombuffer(raw, dtype=np.float32)
        except Exception as exc:
            print(f"[音频失败] {video_path}: {exc}")
            return None

    def reset_current_state(self):
        """清空当前视频组状态。"""
        self.all_audios.clear()
        self.all_peaks.clear()
        self.current_offsets.clear()
        self.missing_cams.clear()
        self.audio_failed_cams.clear()
        self.manually_corrected_cams.clear()
        self.current_review_cams.clear()
        self.master_template = None
        self.master_template_region = None
        self.current_xlim = None
        self.current_action = "quit"

    def load_current_audios(self, video_name):
        """载入当前视频组的 25 路音频。"""
        self.missing_cams = set(self.scan_missing_cameras(video_name))
        self.audio_failed_cams.clear()
        for cam in self.cam_folders:
            if cam in self.missing_cams:
                continue
            video_path = self.find_camera_video(cam, video_name)
            audio = self.extract_audio(video_path)
            if audio is None:
                self.audio_failed_cams.add(cam)
            else:
                self.all_audios[cam] = audio

    def manifest_key_for_video(self, video_name):
        """兼容大小写扩展名，从 manifest 中找到当前视频组 key。"""
        if video_name in self.sync_results:
            return video_name
        lower = video_name.lower()
        for key in self.sync_results:
            if key.lower() == lower:
                return key
        return video_name

    def load_saved_peaks_for_current_group(self):
        """复查时预加载 manifest 中原先保存的 peak，直接显示旧红点位置。"""
        key = self.manifest_key_for_video(self.current_video_name)
        payload = self.sync_results.get(key)
        if not payload:
            return False
        offsets = payload.get("offsets", {})
        if MASTER_CAM not in offsets:
            return False
        master_time = float(payload["master_clap_time"])
        master_peak = int(round(master_time * SAMPLE_RATE))
        self.all_peaks[MASTER_CAM] = master_peak
        for cam, info in offsets.items():
            if cam not in self.cam_folders:
                continue
            peak_time = master_time + float(info.get("offset_ms", 0.0)) / 1000.0
            self.all_peaks[cam] = int(round(peak_time * SAMPLE_RATE))
        print(f"[复查] 已载入 {key} 的历史 peak；红点表示 manifest 中原保存位置。")
        return True

    def on_select(self, min_pos, max_pos, cam_name):
        """框选波形：master 更新模板，其他机位手动锁定峰值。"""
        audio = self.all_audios.get(cam_name)
        if audio is None:
            return

        current_offset = self.current_offsets.get(cam_name, 0.0)
        abs_min_time = min_pos + current_offset
        abs_max_time = max_pos + current_offset
        idx_min = max(0, int(abs_min_time * SAMPLE_RATE))
        idx_max = min(len(audio), int(abs_max_time * SAMPLE_RATE))
        if idx_max - idx_min < 50:
            print(f"[提示] {cam_name} 框选区域太小，已忽略。")
            return

        sd.play(audio[idx_min:idx_max], SAMPLE_RATE)

        if cam_name == MASTER_CAM:
            self.master_template_region = (idx_min, idx_max)
            self.master_template = audio[idx_min:idx_max]
            print("[Master] 已更新 CAM_C3 模板，正在自动对齐未锁定机位。")
            self.auto_align_all()
            master_peak_idx = self.all_peaks.get(MASTER_CAM)
            if master_peak_idx is not None:
                master_peak_time = master_peak_idx / SAMPLE_RATE
                self.current_xlim = (
                    max(0.0, master_peak_time - 1.5),
                    min(float(ANALYZE_SEC), master_peak_time + 1.5),
                )
        else:
            region = audio[idx_min:idx_max]
            local_peak = int(np.argmax(np.abs(region)))
            self.all_peaks[cam_name] = idx_min + local_peak
            self.manually_corrected_cams.add(cam_name)
            print(f"[手动锁定] {cam_name}: 峰值 {self.all_peaks[cam_name] / SAMPLE_RATE:.3f}s")

        self.update_plots()

    def on_click(self, event):
        """右键播放当前机位峰值附近音频。"""
        if event.button != 3 or event.inaxes is None:
            return
        cam_name = event.inaxes.get_gid()
        audio = self.all_audios.get(cam_name)
        peak_idx = self.all_peaks.get(cam_name)
        if audio is None or peak_idx is None:
            return
        start_idx = max(0, int(peak_idx - 0.2 * SAMPLE_RATE))
        end_idx = min(len(audio), int(peak_idx + 0.2 * SAMPLE_RATE))
        sd.play(audio[start_idx:end_idx], SAMPLE_RATE)

    def on_scroll(self, event):
        """滚轮缩放所有子图的时间范围。"""
        if event.inaxes is None or event.xdata is None or self.axes is None:
            return
        left, right = self.axes[0].get_xlim()
        width = right - left
        if width <= 0:
            return
        scale = 0.75 if event.button == "up" else 1.35
        new_width = max(0.05, min(ANALYZE_SEC, width * scale))
        center = event.xdata
        new_left = max(0.0, center - new_width / 2)
        new_right = min(ANALYZE_SEC, center + new_width / 2)
        if new_right - new_left < new_width:
            if new_left <= 0.0:
                new_right = min(ANALYZE_SEC, new_left + new_width)
            else:
                new_left = max(0.0, new_right - new_width)
        self.current_xlim = (new_left, new_right)
        for ax in self.axes:
            ax.set_xlim(self.current_xlim)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        """键盘快捷键入口。"""
        key = (event.key or "").lower()
        if key == "enter":
            if self.save_current_group():
                self.current_action = "next"
                plt.close(self.fig)
        elif key == "s":
            print(f"[跳过] {self.current_video_name} 未写入 manifest。")
            self.current_action = "skip"
            plt.close(self.fig)
        elif key == "r":
            print("[重置] 当前组峰值和手动锁定状态已清空。")
            self.all_peaks.clear()
            self.current_offsets.clear()
            self.manually_corrected_cams.clear()
            self.master_template = None
            self.master_template_region = None
            self.draw_initial_plots()
        elif key == "a":
            print("[自动对齐] 重新对齐未锁定机位。")
            self.auto_align_all()
            self.update_plots()
        elif key == "q":
            print("[结束] 已收到 Q，保存已完成结果并结束全部流程；当前未保存组不会写入。")
            self.stop_requested = True
            self.current_action = "quit"
            plt.close(self.fig)
        elif key == "escape":
            print("[退出] 已收到 Esc，安全退出；当前未保存组不会写入。")
            self.stop_requested = True
            self.current_action = "quit"
            plt.close(self.fig)

    def auto_align_all(self):
        """使用 master 模板对未锁定机位做相关匹配。"""
        if self.master_template is None or self.master_template_region is None:
            print("[提示] 需要先框选 CAM_C3 模板。")
            return
        template_std = float(np.std(self.master_template))
        if template_std < 1e-8:
            print("[警告] CAM_C3 模板能量太低，无法自动对齐。")
            return

        master_local_peak = int(np.argmax(np.abs(self.master_template)))
        master_abs_peak = self.master_template_region[0] + master_local_peak
        self.all_peaks[MASTER_CAM] = master_abs_peak

        template_norm = (self.master_template - np.mean(self.master_template)) / template_std
        for cam in self.cam_folders:
            if cam == MASTER_CAM or cam in self.manually_corrected_cams:
                continue
            target_audio = self.all_audios.get(cam)
            if target_audio is None or len(target_audio) < len(template_norm):
                continue
            correlation = signal.correlate(target_audio, template_norm, mode="valid", method="fft")
            coarse_match_start = int(np.argmax(correlation))
            search_region = target_audio[coarse_match_start: coarse_match_start + len(template_norm)]
            if len(search_region) == 0:
                continue
            local_peak = int(np.argmax(np.abs(search_region)))
            self.all_peaks[cam] = coarse_match_start + local_peak

    def camera_status(self, cam):
        """返回机位当前状态文本。"""
        if cam in self.missing_cams:
            return "缺失"
        if cam in self.audio_failed_cams:
            return "音频失败"
        if cam in self.manually_corrected_cams:
            return "手动锁定"
        if cam in self.current_review_cams:
            return "需复查"
        if cam in self.all_peaks:
            return "自动对齐"
        return "未对齐"

    def apply_axis_style(self, ax, cam):
        """根据机位状态设置子图样式。"""
        status = self.camera_status(cam)
        if status == "手动锁定":
            ax.set_facecolor("#e6f2ff")
        elif status == "需复查":
            ax.set_facecolor("#ffd6d6")
        elif status == "自动对齐":
            ax.set_facecolor("#e9f7ec")
        elif status in {"缺失", "音频失败"}:
            ax.set_facecolor("#fde8e8")
        elif status == "未对齐":
            ax.set_facecolor("#f7f7f7")

        offset = self.current_offsets.get(cam)
        if offset is None:
            label = f"{cam}\n{status}"
        else:
            label = f"{cam}\n{offset * 1000:+.2f} ms\n{status}"
        ax.set_ylabel(label, rotation=0, labelpad=50, va="center", ha="right", fontsize=8, fontproperties=CJK_FONT)
        ax.set_yticks([])
        ax.grid(axis="x", color="#dddddd", linewidth=0.4, alpha=0.8)

    def update_title(self):
        """刷新窗口标题和说明。"""
        missing_count = len(self.missing_cams) + len(self.audio_failed_cams)
        completed = len(self.sync_results)
        review_text = f" | 复查机位 {len(self.current_review_cams)}" if self.review_mode else ""
        title = f"同步 V7.3 | {self.current_video_name} | {self.current_index + 1}/{len(self.video_list)} | 已完成 {completed} | 异常机位 {missing_count}{review_text}"
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self.fig.suptitle(
            "框选 CAM_C3 更新模板并自动对齐；框选其他机位手动锁定；右键播放峰值附近音频；红色背景为建议复查机位",
            fontsize=12,
            fontproperties=CJK_FONT,
        )
        footer = "Enter 保存下一组 | S 跳过 | R 重置 | A 重新自动对齐 | Q 结束全部 | Esc 安全退出 | 滚轮缩放"
        if self.footer_text is None:
            self.footer_text = self.fig.text(
                0.5,
                0.012,
                footer,
                ha="center",
                fontsize=10,
                color="#333333",
                fontproperties=CJK_FONT,
            )
        else:
            self.footer_text.set_text(footer)

    def draw_initial_plots(self):
        """绘制当前组初始波形。"""
        for i, cam in enumerate(self.cam_folders):
            ax = self.axes[i]
            ax.clear()
            ax.set_gid(cam)
            self.apply_axis_style(ax, cam)
            audio = self.all_audios.get(cam)
            if audio is not None:
                time_axis = np.arange(len(audio)) / SAMPLE_RATE
                ax.plot(time_axis, audio, color="#555555", linewidth=0.5)
            if i < len(self.cam_folders) - 1:
                ax.set_xticks([])
        if self.current_xlim is None:
            self.current_xlim = (0.0, min(float(ANALYZE_SEC), 10.0))
        for ax in self.axes:
            ax.set_xlim(self.current_xlim)
        self.update_title()
        self.fig.canvas.draw_idle()

    def update_plots(self):
        """根据当前峰值重绘所有波形。"""
        master_peak_idx = self.all_peaks.get(MASTER_CAM)
        if master_peak_idx is None:
            self.draw_initial_plots()
            return
        master_peak_time = master_peak_idx / SAMPLE_RATE

        for i, cam in enumerate(self.cam_folders):
            ax = self.axes[i]
            ax.clear()
            ax.set_gid(cam)
            audio = self.all_audios.get(cam)
            peak_idx = self.all_peaks.get(cam)
            if audio is not None:
                time_axis = np.arange(len(audio)) / SAMPLE_RATE
                if peak_idx is not None:
                    offset = (peak_idx / SAMPLE_RATE) - master_peak_time
                    self.current_offsets[cam] = offset
                    ax.plot(time_axis - offset, audio, color="#222222", linewidth=0.6)
                    ax.plot(master_peak_time, audio[peak_idx], "o", color="#d62728", markersize=4)
                else:
                    ax.plot(time_axis, audio, color="#777777", linewidth=0.5)
                    self.current_offsets.pop(cam, None)
            ax.axvline(x=master_peak_time, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.8)
            self.apply_axis_style(ax, cam)
            if i < len(self.cam_folders) - 1:
                ax.set_xticks([])
        if self.current_xlim is None:
            self.current_xlim = (master_peak_time - 1.5, master_peak_time + 1.5)
        for ax in self.axes:
            ax.set_xlim(self.current_xlim)
        self.update_title()
        self.fig.canvas.draw_idle()

    def missing_offsets(self):
        """返回还没有有效 peak 的机位。"""
        return [cam for cam in self.cam_folders if cam not in self.all_peaks]

    def save_current_group(self):
        """校验完整性并保存当前视频组到 manifest。"""
        missing_offsets = self.missing_offsets()
        if missing_offsets:
            print(f"[不能保存] 当前组缺少 {len(missing_offsets)} 个机位 peak：{', '.join(missing_offsets)}")
            return False
        master_peak_idx = self.all_peaks.get(MASTER_CAM)
        if master_peak_idx is None:
            print("[不能保存] 缺少 CAM_C3 master peak。")
            return False

        master_clap_time = master_peak_idx / SAMPLE_RATE
        offsets = {}
        for cam in self.cam_folders:
            peak_idx = self.all_peaks[cam]
            lag_samples = peak_idx - master_peak_idx
            lag_ms = (lag_samples / SAMPLE_RATE) * 1000.0
            frame_shift = int(round((lag_samples / SAMPLE_RATE) * FPS_NOMINAL))
            offsets[cam] = {"offset_ms": lag_ms, "frame_shift": frame_shift}

        offsets[MASTER_CAM]["offset_ms"] = 0.0
        offsets[MASTER_CAM]["frame_shift"] = 0
        save_key = self.manifest_key_for_video(self.current_video_name)
        self.sync_results[save_key] = {
            "master_clap_time": master_clap_time,
            "offsets": offsets,
        }
        self.save_manifest()
        print(f"[保存] {save_key} 已写入：{self.output_manifest}")
        print("[offset 摘要] " + ", ".join(f"{cam}:{offsets[cam]['offset_ms']:+.2f}ms" for cam in self.cam_folders))
        return True

    def process_video(self, video_path, index):
        """处理单个视频组。"""
        self.reset_current_state()
        self.current_video_name = video_path.name
        self.current_index = index
        key = self.manifest_key_for_video(self.current_video_name)
        self.current_review_cams = set(self.review_cameras.get(key, self.review_cameras.get(self.current_video_name, set())))

        print(f"\n>>> 载入视频组 {index + 1}/{len(self.video_list)}：{self.current_video_name}")
        if self.current_review_cams:
            print(f"[复查机位] {', '.join(sorted(self.current_review_cams))}")
        self.load_current_audios(self.current_video_name)
        loaded_saved = self.load_saved_peaks_for_current_group()
        if self.missing_cams:
            print(f"[缺失视频] {', '.join(sorted(self.missing_cams))}")
        if self.audio_failed_cams:
            print(f"[音频失败] {', '.join(sorted(self.audio_failed_cams))}")

        self.fig, self.axes = plt.subplots(len(self.cam_folders), 1, figsize=(17, 21), sharex=True)
        plt.subplots_adjust(left=0.14, right=0.98, top=0.94, bottom=0.04, hspace=0.04)
        self.span_selectors.clear()
        for ax, cam in zip(self.axes, self.cam_folders):
            selector = SpanSelector(
                ax,
                lambda min_pos, max_pos, c=cam: self.on_select(min_pos, max_pos, c),
                "horizontal",
                useblit=True,
                props={"alpha": 0.25, "facecolor": "#00a6d6"},
            )
            self.span_selectors.append(selector)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        if loaded_saved:
            self.update_plots()
        else:
            self.draw_initial_plots()
        plt.show()

    def run(self):
        """主循环：逐视频组处理，支持恢复和一键结束。"""
        self.ensure_environment()
        self.load_existing_manifest()
        print(f"[启动] 原始视频根目录：{self.root}")
        print(f"[启动] manifest 输出：{self.output_manifest}")
        print(f"[启动] 共发现 {len(self.video_list)} 个 master 视频。")

        for index, video_path in enumerate(self.video_list):
            if self.stop_requested:
                break
            if not self.review_mode and not OVERWRITE_EXISTING and video_path.name in self.sync_results:
                print(f"[跳过已完成] {video_path.name}")
                continue
            self.process_video(video_path, index)
            if self.stop_requested:
                break

        self.save_manifest()
        print("\n=== 同步工具结束 ===")
        print(f"已完成视频组：{len(self.sync_results)}")
        print(f"manifest：{self.output_manifest}")


def parse_args():
    """解析复查模式参数；不传参数时保持原正常同步模式。"""
    parser = argparse.ArgumentParser(description="5x5 音频同步 GUI。")
    parser.add_argument("--review-warnings", action="store_true", help="只打开 manifest 中 offset_ms 可疑的场景，并标红可疑机位。")
    parser.add_argument("--manifest", default=str(OUTPUT_MANIFEST), help="sync manifest 路径。")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="原始视频根目录。")
    parser.add_argument("--abs-warn-ms", type=float, default=ABS_WARN_MS, help="复查模式绝对 offset 警告阈值。")
    parser.add_argument("--abs-fail-ms", type=float, default=ABS_FAIL_MS, help="复查模式绝对 offset 严重阈值。")
    parser.add_argument("--camera-mad-k", type=float, default=CAMERA_MAD_K, help="同一机位跨场景 robust z 离群阈值。")
    parser.add_argument("--review-list-only", action="store_true", help="只打印复查场景和机位，不打开 GUI。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest_path = Path(args.manifest)
    review_map = {}
    if args.review_warnings:
        if not manifest_path.exists():
            raise FileNotFoundError(f"复查模式需要已有 manifest：{manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        review_map, review_rows = build_review_camera_map(
            existing_manifest,
            abs_warn_ms=args.abs_warn_ms,
            abs_fail_ms=args.abs_fail_ms,
            camera_mad_k=args.camera_mad_k,
        )
        print(f"[复查模式] 可疑场景数：{len(review_map)}，可疑条目数：{sum(len(v) for v in review_map.values())}")
        for scene, cams in sorted(review_map.items()):
            print(f"  {scene}: {', '.join(sorted(cams))}")
        if args.review_list_only:
            raise SystemExit(0)

    tool = InteractiveSyncTool(
        Path(args.data_root),
        manifest_path,
        review_mode=args.review_warnings,
        review_cameras=review_map,
    )
    tool.run()




