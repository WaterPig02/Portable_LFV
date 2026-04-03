import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXPORT_DIR = Path(__file__).resolve().parents[1] / "export"
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

from asset_versioning import (
    CROP_POLICY_VERSION,
    ROI_METADATA_FILENAME,
    compute_export_signature,
    normalize_rectify_metadata,
)
from config import get_default_config


ASPECT_WIDTH = 16
ASPECT_HEIGHT = 9


def load_rectification_assets(rectify_dir):
    rectify_path = Path(rectify_dir)
    rectify_meta_path = rectify_path / "rectify_meta.json"
    if not rectify_meta_path.exists():
        raise FileNotFoundError(f"missing rectification metadata: {rectify_meta_path}")
    with rectify_meta_path.open("r", encoding="utf-8") as handle:
        raw_meta = json.load(handle)
    rectification_meta = normalize_rectify_metadata(raw_meta)
    if rectification_meta["image_size"] is None:
        first_camera_meta = next(iter(rectification_meta["cameras"].values()), None)
        if first_camera_meta is None:
            raise ValueError("rectification metadata contains no cameras")
        map_path = rectify_path / first_camera_meta["map_file"]
        if not map_path.exists():
            raise FileNotFoundError(f"missing rectification map for legacy metadata inference: {map_path}")
        data = np.load(map_path)
        rectification_meta["image_size"] = [int(data["map_x"].shape[1]), int(data["map_x"].shape[0])]
        rectification_meta["rectification_asset_version"] = compute_export_signature(
            {
                "schema_version": rectification_meta["schema_version"],
                "master_camera": rectification_meta["master_camera"],
                "image_size": rectification_meta["image_size"],
                "crop_alpha": rectification_meta["crop_alpha"],
                "cameras": rectification_meta["cameras"],
            }
        )
    return rectify_path, rectification_meta


def compute_common_valid_roi(rectify_dir, rectification_meta):
    width, height = rectification_meta["image_size"]
    x0_values = []
    y0_values = []
    x1_values = []
    y1_values = []

    for camera_id, camera_meta in sorted(rectification_meta["cameras"].items()):
        map_path = rectify_dir / camera_meta["map_file"]
        if not map_path.exists():
            raise FileNotFoundError(f"missing rectification map for {camera_id}: {map_path}")
        data = np.load(map_path)
        map_x = data["map_x"]
        map_y = data["map_y"]
        valid_mask = (map_x >= 0.0) & (map_x <= (width - 1)) & (map_y >= 0.0) & (map_y <= (height - 1))
        if not np.any(valid_mask):
            raise ValueError(f"no valid projection region for {camera_id}")
        valid_rows = np.where(valid_mask.any(axis=1))[0]
        valid_cols = np.where(valid_mask.any(axis=0))[0]
        x0_values.append(int(valid_cols[0]))
        y0_values.append(int(valid_rows[0]))
        x1_values.append(int(valid_cols[-1]) + 1)
        y1_values.append(int(valid_rows[-1]) + 1)

    x = max(x0_values)
    y = max(y0_values)
    right = min(x1_values)
    bottom = min(y1_values)
    roi_w = right - x
    roi_h = bottom - y
    if roi_w <= 0 or roi_h <= 0:
        raise ValueError("failed to compute common valid ROI")
    return {"x": int(x), "y": int(y), "width": int(roi_w), "height": int(roi_h)}


def compute_final_release_crop(common_roi):
    roi_w = int(common_roi["width"])
    roi_h = int(common_roi["height"])
    scale = min(roi_w // ASPECT_WIDTH, roi_h // ASPECT_HEIGHT)
    if scale <= 0:
        raise ValueError("invalid common ROI")

    crop_w = int(scale * ASPECT_WIDTH)
    crop_h = int(scale * ASPECT_HEIGHT)

    if crop_w > roi_w or crop_h > roi_h or crop_w <= 0 or crop_h <= 0:
        raise ValueError("common ROI cannot contain a legal 16:9 crop")

    offset_x = (roi_w - crop_w) // 2
    offset_y = (roi_h - crop_h) // 2
    crop = {
        "x": int(common_roi["x"] + offset_x),
        "y": int(common_roi["y"] + offset_y),
        "width": int(crop_w),
        "height": int(crop_h),
    }
    if crop["x"] < common_roi["x"] or crop["y"] < common_roi["y"]:
        raise ValueError("final crop is outside common ROI")
    if crop["x"] + crop["width"] > common_roi["x"] + common_roi["width"]:
        raise ValueError("final crop exceeds common ROI width")
    if crop["y"] + crop["height"] > common_roi["y"] + common_roi["height"]:
        raise ValueError("final crop exceeds common ROI height")
    if crop["width"] * ASPECT_HEIGHT != crop["height"] * ASPECT_WIDTH:
        raise ValueError("final crop is not an exact 16:9 rectangle")
    return crop


def build_roi_metadata(rectification_meta, common_valid_roi, final_release_crop):
    payload = {
        "schema_version": "release_roi_metadata_v1",
        "rectification_asset_version": rectification_meta["rectification_asset_version"],
        "image_size": rectification_meta["image_size"],
        "common_valid_roi": common_valid_roi,
        "final_release_crop_16_9": final_release_crop,
        "crop_policy_version": CROP_POLICY_VERSION,
        "roi_policy": "common_valid_roi_from_rectification_maps",
        "final_crop_policy": "centered_max_inscribed_16_9",
    }
    payload["roi_asset_version"] = compute_export_signature(payload)
    return payload


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--batch-name", default=None, choices=["secondsyn", "firstsyn"])
    pre_args, _ = pre_parser.parse_known_args()

    config = get_default_config(pre_args.batch_name)
    default_rectify_dir = config["paths"].get("rectify_dir")
    default_roi_output = config["paths"].get("roi_metadata")

    parser = argparse.ArgumentParser(description="Generate persistent ROI metadata for public export.")
    parser.add_argument("--batch-name", default=pre_args.batch_name, choices=["secondsyn", "firstsyn"], help="Named batch config to use.")
    parser.add_argument("--rectify-dir", default=default_rectify_dir, help="Directory containing rectify_meta.json and map files.")
    parser.add_argument(
        "--output",
        default=default_roi_output,
        help=f"Optional output path. Default: config path or <rectify-dir>/{ROI_METADATA_FILENAME}",
    )
    args = parser.parse_args()
    if args.rectify_dir is None:
        raise ValueError("missing rectify_dir; provide --rectify-dir or set paths.rectify_dir in config.py")

    rectify_dir, rectification_meta = load_rectification_assets(args.rectify_dir)
    common_valid_roi = compute_common_valid_roi(rectify_dir, rectification_meta)
    final_release_crop = compute_final_release_crop(common_valid_roi)
    roi_metadata = build_roi_metadata(rectification_meta, common_valid_roi, final_release_crop)

    output_path = Path(args.output) if args.output else rectify_dir / ROI_METADATA_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(roi_metadata, handle, indent=4, ensure_ascii=False)

    print(f"Saved ROI metadata to {output_path}")
    print(f"roi_asset_version={roi_metadata['roi_asset_version']}")


if __name__ == "__main__":
    main()
