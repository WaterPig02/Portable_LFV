"""RealDynLFV pipeline v0 orchestration and preflight entry point."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline_config import camera_ids, config_value, load_pipeline_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "sync-qa",
    "sync-clip",
    "calibrate",
    "calibration-qa",
    "rectify",
    "roi",
    "roi-qa",
    "export",
    "export-qa",
)


class Preflight:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []

    def require_file(self, label, value):
        if not value:
            self.errors.append(f"missing config value: {label}")
            return None
        path = Path(value)
        if not path.is_file():
            self.errors.append(f"missing file ({label}): {path}")
        return path

    def require_dir(self, label, value):
        if not value:
            self.errors.append(f"missing config value: {label}")
            return None
        path = Path(value)
        if not path.is_dir():
            self.errors.append(f"missing directory ({label}): {path}")
        return path

    def warn_output(self, label, value, scene=None, profile=None):
        if not value:
            self.errors.append(f"missing config value: {label}")
            return
        path = Path(value)
        candidate = path
        if profile:
            candidate = candidate / profile
        if scene:
            candidate = candidate / scene
        has_output = candidate.exists()
        if candidate.is_dir():
            has_output = any(candidate.iterdir())
        if has_output:
            self.warnings.append(f"output already exists and may be overwritten or skipped: {candidate}")

    def print(self):
        for message in self.info:
            print(f"[INFO] {message}")
        for message in self.warnings:
            print(f"[WARN] {message}")
        for message in self.errors:
            print(f"[ERROR] {message}")
        print(f"Preflight summary: errors={len(self.errors)}, warnings={len(self.warnings)}")


CONFIG = {}


def tool_path(config, name, fallback):
    return str(config_value(config, "tools", name, default=fallback))


def check_tool(preflight, executable, label):
    resolved = shutil.which(executable) if not Path(executable).is_file() else executable
    if not resolved:
        preflight.errors.append(f"{label} is unavailable: {executable}")
        return False
    preflight.info.append(f"{label}: {resolved}")
    return True


def check_nvenc(preflight, ffmpeg):
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
        if "hevc_nvenc" in result.stdout:
            preflight.info.append("FFmpeg reports hevc_nvenc support")
        else:
            preflight.warnings.append("FFmpeg does not report hevc_nvenc; sync-clip cannot use its current NVENC command")
    except Exception as exc:
        preflight.warnings.append(f"unable to inspect NVENC support: {exc}")


def check_camera_layout(preflight, config, root_key=None):
    cameras = camera_ids(config)
    if len(cameras) != 25 or len(set(cameras)) != 25:
        preflight.errors.append(f"camera layout must contain exactly 25 unique views, got {len(set(cameras))}")
    expected = {f"CAM_{row}{column}" for row in "ABCDE" for column in range(1, 6)}
    if set(cameras) != expected:
        preflight.errors.append("camera layout must be the RealDynLFV 5x5 layout CAM_A1..CAM_E5")
    reference = config_value(config, "camera", "reference")
    if reference != "CAM_C3":
        preflight.errors.append(f"reference camera must be CAM_C3, got {reference!r}")
    if root_key:
        root = config_value(config, "paths", root_key)
        root_path = preflight.require_dir(f"paths.{root_key}", root)
        if root_path and root_path.is_dir():
            missing = [camera for camera in cameras if not (root_path / camera).is_dir()]
            if missing:
                preflight.errors.append(f"{root_key} is missing camera directories: {', '.join(missing)}")


def load_json_checked(preflight, label, path_value):
    path = preflight.require_file(label, path_value)
    if not path or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        preflight.errors.append(f"invalid JSON ({label}): {exc}")
        return None


def preflight_stage(config, stage, scene, profile):
    check = Preflight()
    paths = config.get("paths", {})
    ffmpeg = tool_path(config, "ffmpeg", "ffmpeg")
    ffprobe = tool_path(config, "ffprobe", "ffprobe")

    check_camera_layout(check, config)
    check_tool(check, ffmpeg, "FFmpeg")
    check_tool(check, ffprobe, "ffprobe")

    if stage in {"sync-qa", "sync-clip", "roi-qa", "export"}:
        manifest = load_json_checked(check, "paths.sync_manifest", paths.get("sync_manifest"))
        if isinstance(manifest, dict):
            expected_cameras = set(camera_ids(config))
            for video_name, group in manifest.items():
                offsets = group.get("offsets", {}) if isinstance(group, dict) else {}
                if set(offsets) != expected_cameras:
                    check.errors.append(f"sync manifest camera set mismatch: {video_name}")
                    break
                if float(offsets.get("CAM_C3", {}).get("offset_ms", float("nan"))) != 0.0:
                    check.errors.append(f"CAM_C3 offset_ms must be 0.0: {video_name}")
                    break
        if manifest is not None and scene:
            stems = {Path(name).stem for name in manifest}
            if scene not in stems:
                check.errors.append(f"scene {scene} is not present in sync manifest")

    if stage == "sync-clip":
        check_camera_layout(check, config, "raw_root")
        check.warn_output("paths.synced_root", paths.get("synced_root"))
        check_nvenc(check, ffmpeg)

    if stage in {"calibrate", "calibration-qa"}:
        check_camera_layout(check, config, "synced_root")
        target_name = f"{scene}.mp4" if scene else str(config_value(config, "calibration", "target_video", default="0001.mp4"))
        synced_root = Path(paths.get("synced_root", ""))
        missing = [camera for camera in camera_ids(config) if not (synced_root / camera / target_name).is_file()]
        if missing:
            check.errors.append(f"calibration target {target_name} missing for: {', '.join(missing)}")
        if stage == "calibrate":
            check.warn_output("paths.calibration_json", paths.get("calibration_json"))
        else:
            load_json_checked(check, "paths.calibration_json", paths.get("calibration_json"))

    if stage in {"rectify", "roi", "roi-qa", "export"}:
        calibration = load_json_checked(check, "paths.calibration_json", paths.get("calibration_json"))
        if isinstance(calibration, dict) and set(calibration) != set(camera_ids(config)):
            check.errors.append("calibration JSON camera set does not match the configured 25-view layout")

    if stage == "rectify":
        check.warn_output("paths.rectification_dir", paths.get("rectification_dir"))

    if stage in {"roi", "roi-qa", "export"}:
        rectify_dir = check.require_dir("paths.rectification_dir", paths.get("rectification_dir"))
        if rectify_dir:
            check.require_file("rectify_meta.json", rectify_dir / "rectify_meta.json")
            missing_maps = [camera for camera in camera_ids(config) if not (rectify_dir / f"{camera}_rect_map.npz").is_file()]
            if missing_maps:
                check.errors.append(f"missing rectification maps: {', '.join(missing_maps)}")

    if stage in {"roi-qa", "export"}:
        load_json_checked(check, "paths.roi_metadata", paths.get("roi_metadata"))

    if stage in {"roi-qa", "export"}:
        time_data = load_json_checked(check, "paths.time_segments", paths.get("time_segments"))
        batch_name = config_value(config, "batch_name")
        if isinstance(time_data, dict) and batch_name not in time_data:
            check.errors.append(f"time segments do not contain batch key: {batch_name}")
        if scene and isinstance(time_data, dict) and scene not in time_data.get(batch_name, {}):
            check.errors.append(f"scene {scene} is not present in time segments for {batch_name}")

    if stage == "export":
        check_camera_layout(check, config, "synced_root")
        check.require_file("paths.lut", paths.get("lut"))
        check.warn_output("paths.output_root", paths.get("output_root"), scene=scene, profile=profile)

    if stage == "export-qa":
        output_root = check.require_dir("paths.output_root", paths.get("output_root"))
        if output_root:
            profile_root = output_root / profile
            if not profile_root.is_dir():
                check.errors.append(f"export profile directory is missing: {profile_root}")

    return check


def build_command(config, stage, scene, profile):
    python = tool_path(config, "python", sys.executable)
    config_path = str(config["_config_path"])
    paths = config["paths"]
    batch_name = str(config_value(config, "batch_name"))

    if stage == "sync-qa":
        return [python, str(PROJECT_ROOT / "sync" / "check_sync_manifest_offsets.py"), "--manifest", paths["sync_manifest"]]
    if stage == "sync-clip":
        command = [python, str(PROJECT_ROOT / "sync" / "Batch_Clip_Engine.py"), "--config", config_path]
        if scene:
            command.extend(["--scene", scene])
        return command
    if stage == "calibrate":
        command = [python, str(PROJECT_ROOT / "Code" / "calibration" / "Calib_Solver_V7_Locked.py"), "--config", config_path]
        if scene:
            command.extend(["--target-video", f"{scene}.mp4"])
        return command
    if stage == "calibration-qa":
        if batch_name not in {"firstsyn", "secondsyn"}:
            raise NotImplementedError("calibration-qa currently requires batch_name firstsyn or secondsyn")
        command = [
            python, str(PROJECT_ROOT / "Code" / "calibration" / "validate_calibration_quality.py"),
            "--batch-name", batch_name, "--input-root", paths["synced_root"],
            "--calibration-json", paths["calibration_json"], "--output-dir", paths["calibration_report_dir"],
        ]
        if scene:
            command.extend(["--target-video", f"{scene}.mp4"])
        return command
    if stage == "rectify":
        return [python, str(PROJECT_ROOT / "Code" / "prepare" / "Rectify_Generator.py"), "--config", config_path]
    if stage == "roi":
        return [
            python, str(PROJECT_ROOT / "Code" / "prepare" / "generate_release_roi.py"),
            "--rectify-dir", paths["rectification_dir"], "--output", paths["roi_metadata"],
            "--margin-px", str(config_value(config, "roi", "margin_px", default=4)),
        ]
    if stage == "roi-qa":
        if batch_name not in {"firstsyn", "secondsyn"}:
            raise NotImplementedError("roi-qa currently requires batch_name firstsyn or secondsyn")
        command = [
            python, str(PROJECT_ROOT / "Code" / "prepare" / "validate_release_roi.py"),
            "--batch-name", batch_name, "--rectify-dir", paths["rectification_dir"],
            "--roi-metadata", paths["roi_metadata"], "--input-root", paths["synced_root"],
            "--time-json", paths["time_segments"], "--time-key", batch_name,
        ]
        if scene:
            command.extend(["--scenes", scene])
        return command
    if stage == "export":
        if batch_name not in {"firstsyn", "secondsyn"}:
            raise NotImplementedError("export currently requires batch_name firstsyn or secondsyn")
        command = [
            python, str(PROJECT_ROOT / "Code" / "export" / "export_dataset.py"),
            "--batch-name", batch_name, "--profile", profile,
            "--input-root", paths["synced_root"], "--sync-manifest", paths["sync_manifest"],
            "--time-json", paths["time_segments"], "--rectify-dir", paths["rectification_dir"],
            "--roi-metadata", paths["roi_metadata"], "--output-root", paths["output_root"],
            "--lut-path", paths["lut"], "--max-workers", str(config_value(config, "export", "workers", default=4)),
        ]
        if scene:
            command.extend(["--scene", scene])
        return command
    if stage == "export-qa":
        if batch_name not in {"firstsyn", "secondsyn"}:
            raise NotImplementedError("export-qa currently requires batch_name firstsyn or secondsyn")
        command = [
            python, str(PROJECT_ROOT / "Code" / "export" / "validate_export.py"),
            "--batch-name", batch_name, "--profile", profile, "--output-root", paths["output_root"],
        ]
        if scene:
            command.extend(["--scene", scene])
        return command
    raise ValueError(f"unsupported stage: {stage}")


def main():
    parser = argparse.ArgumentParser(description="RealDynLFV pipeline v0 orchestration and preflight.")
    parser.add_argument("--config", required=True, help="Pipeline batch YAML config.")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--scene", default=None, help="Optional scene stem, for example 0027.")
    parser.add_argument("--profile", choices=["benchmark", "fidelity"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it.")
    parser.add_argument("--preflight-only", action="store_true", help="Run checks and stop before command construction/execution.")
    args = parser.parse_args()

    global CONFIG
    CONFIG = load_pipeline_config(args.config)
    profile = args.profile or str(config_value(CONFIG, "export", "default_profile", default="benchmark"))
    if profile == "fidelity":
        print("[WARN] fidelity is not a formal end-to-end 16-bit release in pipeline v0")

    check = preflight_stage(CONFIG, args.stage, args.scene, profile)
    check.print()
    if check.errors:
        raise SystemExit(2)
    if args.preflight_only:
        return

    try:
        command = build_command(CONFIG, args.stage, args.scene, profile)
    except NotImplementedError as exc:
        print(f"[TODO] {exc}")
        raise SystemExit(3)
    print("Command:")
    print(subprocess.list2cmdline(command))
    if args.dry_run:
        return
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
