import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from pipeline_config import config_value, load_pipeline_config

CAMERAS = [f"CAM_{row}{col}" for row in "ABCDE" for col in range(1, 6)]
REFERENCE_CAMERA = "CAM_C3"


def load_manifest(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def median_abs_deviation(values):
    if not values:
        return 0.0
    med = statistics.median(values)
    deviations = [abs(v - med) for v in values]
    return statistics.median(deviations)


def analyze_manifest(data, abs_warn_ms, abs_fail_ms, scene_mad_k, camera_mad_k):
    rows = []
    missing = []

    for scene, payload in sorted(data.items()):
        offsets = payload.get("offsets", {})
        scene_values = []
        for cam in CAMERAS:
            if cam not in offsets:
                missing.append({"scene": scene, "camera": cam, "reason": "missing_camera_offset"})
                continue
            try:
                value = float(offsets[cam]["offset_ms"])
            except Exception:
                missing.append({"scene": scene, "camera": cam, "reason": "invalid_offset_ms"})
                continue
            scene_values.append(value)
            rows.append({
                "scene": scene,
                "camera": cam,
                "offset_ms": value,
                "frame_shift": offsets[cam].get("frame_shift"),
                "abs_offset_ms": abs(value),
                "flags": [],
            })

        if REFERENCE_CAMERA in offsets:
            ref_value = float(offsets[REFERENCE_CAMERA].get("offset_ms", 999999))
            if abs(ref_value) > 1e-6:
                missing.append({"scene": scene, "camera": REFERENCE_CAMERA, "reason": f"reference_offset_not_zero:{ref_value}"})

        if scene_values:
            scene_median = statistics.median(scene_values)
            scene_mad = median_abs_deviation(scene_values)
            robust_sigma = max(scene_mad * 1.4826, 1e-6)
            for row in rows:
                if row["scene"] != scene:
                    continue
                row["scene_median_ms"] = scene_median
                row["scene_robust_z"] = abs(row["offset_ms"] - scene_median) / robust_sigma
                if row["scene_robust_z"] >= scene_mad_k:
                    row["flags"].append("scene_outlier")

    by_camera = {cam: [] for cam in CAMERAS}
    for row in rows:
        by_camera[row["camera"]].append(row["offset_ms"])

    camera_stats = {}
    for cam, values in by_camera.items():
        if not values:
            continue
        cam_median = statistics.median(values)
        cam_mad = median_abs_deviation(values)
        robust_sigma = max(cam_mad * 1.4826, 1e-6)
        camera_stats[cam] = {
            "count": len(values),
            "median_ms": cam_median,
            "min_ms": min(values),
            "max_ms": max(values),
            "range_ms": max(values) - min(values),
            "mad_ms": cam_mad,
        }
        for row in rows:
            if row["camera"] != cam:
                continue
            row["camera_median_ms"] = cam_median
            row["camera_robust_z"] = abs(row["offset_ms"] - cam_median) / robust_sigma
            if row["camera_robust_z"] >= camera_mad_k:
                row["flags"].append("camera_unstable_outlier")

    for row in rows:
        if row["abs_offset_ms"] >= abs_fail_ms:
            row["flags"].append("abs_fail")
        elif row["abs_offset_ms"] >= abs_warn_ms:
            row["flags"].append("abs_warn")

    flagged = [row for row in rows if row["flags"]]
    flagged.sort(key=lambda r: ("abs_fail" not in r["flags"], -r["abs_offset_ms"]))

    return rows, flagged, missing, camera_stats


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene", "camera", "offset_ms", "frame_shift", "abs_offset_ms",
        "scene_median_ms", "scene_robust_z", "camera_median_ms", "camera_robust_z", "flags",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["flags"] = ";".join(row.get("flags", []))
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(description="检查 sync manifest 中异常 offset_ms。")
    parser.add_argument("--config", default=None, help="Optional RealDynLFV batch YAML config.")
    parser.add_argument("--manifest", default=r"D:\Project\LF_dataset\Calibration\Calibration_Data\firstsyn\sync_manifest_firstsyn.json")
    parser.add_argument("--abs-warn-ms", type=float, default=300.0, help="绝对 offset 警告阈值。")
    parser.add_argument("--abs-fail-ms", type=float, default=600.0, help="绝对 offset 严重阈值。")
    parser.add_argument("--scene-mad-k", type=float, default=6.0, help="同一场景内 robust z 离群阈值。")
    parser.add_argument("--camera-mad-k", type=float, default=6.0, help="同一机位跨场景 robust z 离群阈值。")
    parser.add_argument("--csv", default=None, help="可选 CSV 明细输出路径。")
    parser.add_argument("--top", type=int, default=40, help="终端显示最严重的前 N 条。")
    args = parser.parse_args()
    if args.config:
        config = load_pipeline_config(args.config)
        args.manifest = config_value(config, "paths", "sync_manifest", default=args.manifest)

    data = load_manifest(args.manifest)
    rows, flagged, missing, camera_stats = analyze_manifest(
        data,
        abs_warn_ms=args.abs_warn_ms,
        abs_fail_ms=args.abs_fail_ms,
        scene_mad_k=args.scene_mad_k,
        camera_mad_k=args.camera_mad_k,
    )

    print("=== sync_manifest offset_ms 检查 ===")
    print(f"manifest: {args.manifest}")
    print(f"scene 数: {len(data)}")
    print(f"offset 条目数: {len(rows)}")
    print(f"缺失/非法条目数: {len(missing)}")
    print(f"异常/可疑条目数: {len(flagged)}")
    print(f"阈值: abs_warn={args.abs_warn_ms} ms, abs_fail={args.abs_fail_ms} ms, scene_mad_k={args.scene_mad_k}, camera_mad_k={args.camera_mad_k}")

    if missing:
        print("\n--- 缺失或非法条目 ---")
        for item in missing[:args.top]:
            print(f"{item['scene']} {item['camera']} {item['reason']}")

    print("\n--- 绝对值最大的 offset ---")
    for row in sorted(rows, key=lambda r: r["abs_offset_ms"], reverse=True)[:args.top]:
        flags = ",".join(row["flags"]) if row["flags"] else "-"
        print(f"{row['scene']:>10} {row['camera']:>6} offset={row['offset_ms']:+9.3f} ms frame={row['frame_shift']:>4} flags={flags}")

    if flagged:
        print("\n--- 建议复查条目 ---")
        for row in flagged[:args.top]:
            flags = ",".join(row["flags"])
            print(
                f"{row['scene']:>10} {row['camera']:>6} offset={row['offset_ms']:+9.3f} ms "
                f"scene_z={row.get('scene_robust_z', 0):6.2f} camera_z={row.get('camera_robust_z', 0):6.2f} flags={flags}"
            )

    print("\n--- 各机位跨场景范围最大排名 ---")
    ranked_cams = sorted(camera_stats.items(), key=lambda item: item[1]["range_ms"], reverse=True)
    for cam, stat in ranked_cams[:args.top]:
        print(
            f"{cam:>6} count={stat['count']:>3} median={stat['median_ms']:+9.3f} ms "
            f"min={stat['min_ms']:+9.3f} max={stat['max_ms']:+9.3f} range={stat['range_ms']:9.3f} mad={stat['mad_ms']:8.3f}"
        )

    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nCSV 明细已写出: {args.csv}")

    if flagged or missing:
        print("\n结论: 存在需要复查的 offset_ms。优先检查 abs_fail、scene_outlier、camera_unstable_outlier 同时出现的条目。")
    else:
        print("\n结论: 未发现超过当前阈值的明显异常 offset_ms。")


if __name__ == "__main__":
    main()
