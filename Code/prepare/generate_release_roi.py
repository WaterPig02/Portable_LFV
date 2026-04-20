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
    ROI_POLICY_NAME,
    ROI_POLICY_VERSION,
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


def build_valid_mask(map_x, map_y, width, height, margin_x, margin_y):
    return (
        (map_x >= float(margin_x))
        & (map_x <= float((width - 1) - margin_x))
        & (map_y >= float(margin_y))
        & (map_y <= float((height - 1) - margin_y))
    )


def bbox_from_mask(mask):
    if not np.any(mask):
        raise ValueError("common valid mask is empty")
    ys, xs = np.where(mask)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def largest_rectangle_in_mask(mask):
    height, width = mask.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_rect = None

    for row in range(height):
        heights = np.where(mask[row], heights + 1, 0)
        stack = []
        for idx in range(width + 1):
            current_height = int(heights[idx]) if idx < width else 0
            start = idx
            while stack and stack[-1][1] > current_height:
                start_idx, rect_height = stack.pop()
                rect_width = idx - start_idx
                area = rect_height * rect_width
                if area > best_area:
                    best_area = area
                    best_rect = {
                        "x": int(start_idx),
                        "y": int(row - rect_height + 1),
                        "width": int(rect_width),
                        "height": int(rect_height),
                    }
                start = start_idx
            if not stack or stack[-1][1] < current_height:
                stack.append((start, current_height))

    if best_rect is None or best_rect["width"] <= 0 or best_rect["height"] <= 0:
        raise ValueError("failed to compute safe rectangular ROI from common valid mask")
    return best_rect


def compute_common_valid_roi(rectify_dir, rectification_meta, margin_x, margin_y):
    width, height = rectification_meta["image_size"]
    common_mask = None
    camera_valid_masks = {}

    for camera_id, camera_meta in sorted(rectification_meta["cameras"].items()):
        map_path = rectify_dir / camera_meta["map_file"]
        if not map_path.exists():
            raise FileNotFoundError(f"missing rectification map for {camera_id}: {map_path}")
        data = np.load(map_path)
        map_x = data["map_x"]
        map_y = data["map_y"]
        valid_mask = build_valid_mask(map_x, map_y, width, height, margin_x, margin_y)
        if not np.any(valid_mask):
            raise ValueError(f"no valid projection region for {camera_id} after applying margin")
        camera_valid_masks[camera_id] = valid_mask
        common_mask = valid_mask.copy() if common_mask is None else (common_mask & valid_mask)

    common_valid_mask_bbox = bbox_from_mask(common_mask)
    safe_rect_roi = largest_rectangle_in_mask(common_mask)
    return common_mask, camera_valid_masks, common_valid_mask_bbox, safe_rect_roi


def compute_final_release_crop(safe_rect_roi):
    roi_w = int(safe_rect_roi["width"])
    roi_h = int(safe_rect_roi["height"])
    scale = min(roi_w // ASPECT_WIDTH, roi_h // ASPECT_HEIGHT)
    if scale <= 0:
        raise ValueError("safe rectangular ROI cannot contain a legal 16:9 crop")

    crop_w = int(scale * ASPECT_WIDTH)
    crop_h = int(scale * ASPECT_HEIGHT)
    if crop_w > roi_w or crop_h > roi_h or crop_w <= 0 or crop_h <= 0:
        raise ValueError("safe rectangular ROI cannot contain a legal 16:9 crop")

    offset_x = (roi_w - crop_w) // 2
    offset_y = (roi_h - crop_h) // 2
    crop = {
        "x": int(safe_rect_roi["x"] + offset_x),
        "y": int(safe_rect_roi["y"] + offset_y),
        "width": int(crop_w),
        "height": int(crop_h),
    }
    if crop["width"] * ASPECT_HEIGHT != crop["height"] * ASPECT_WIDTH:
        raise ValueError("final crop is not an exact 16:9 rectangle")
    return crop


def compute_crop_margin_summary(rectify_dir, rectification_meta, final_crop):
    width, height = rectification_meta["image_size"]
    x0 = int(final_crop["x"])
    y0 = int(final_crop["y"])
    x1 = x0 + int(final_crop["width"]) - 1
    y1 = y0 + int(final_crop["height"]) - 1

    summary = {}
    global_min = None
    for camera_id, camera_meta in sorted(rectification_meta["cameras"].items()):
        data = np.load(rectify_dir / camera_meta["map_file"])
        map_x = data["map_x"]
        map_y = data["map_y"]
        safety = np.minimum.reduce([map_x, map_y, (width - 1) - map_x, (height - 1) - map_y])
        top = safety[y0, x0 : x1 + 1]
        bottom = safety[y1, x0 : x1 + 1]
        left = safety[y0 : y1 + 1, x0]
        right = safety[y0 : y1 + 1, x1]
        edge_min = {
            "top": float(np.min(top)),
            "bottom": float(np.min(bottom)),
            "left": float(np.min(left)),
            "right": float(np.min(right)),
        }
        min_margin = min(edge_min.values())
        summary[camera_id] = {
            "edge_min_margin_px": edge_min,
            "min_margin_px": float(min_margin),
        }
        global_min = min(global_min, min_margin) if global_min is not None else min_margin
    return summary, float(global_min if global_min is not None else 0.0)


def validate_static_roi(common_mask, safe_rect_roi, final_crop):
    common_bbox = bbox_from_mask(common_mask)
    sx0 = int(safe_rect_roi["x"])
    sy0 = int(safe_rect_roi["y"])
    sx1 = sx0 + int(safe_rect_roi["width"])
    sy1 = sy0 + int(safe_rect_roi["height"])
    if not np.all(common_mask[sy0:sy1, sx0:sx1]):
        raise ValueError("safe_rect_roi is not fully contained inside the common valid mask")

    fx0 = int(final_crop["x"])
    fy0 = int(final_crop["y"])
    fx1 = fx0 + int(final_crop["width"])
    fy1 = fy0 + int(final_crop["height"])
    if fx0 < sx0 or fy0 < sy0 or fx1 > sx1 or fy1 > sy1:
        raise ValueError("final_release_crop_16_9 is not fully contained inside safe_rect_roi")
    if final_crop["width"] * ASPECT_HEIGHT != final_crop["height"] * ASPECT_WIDTH:
        raise ValueError("final_release_crop_16_9 is not an exact 16:9 rectangle")

    return {
        "status": "pass",
        "common_valid_mask_bbox": common_bbox,
        "safe_rect_contains_only_valid_pixels": True,
        "final_crop_inside_safe_rect": True,
        "final_crop_is_exact_16_9": True,
    }


def build_roi_metadata(
    rectification_meta,
    margin_x,
    margin_y,
    common_valid_mask_bbox,
    safe_rect_roi,
    final_release_crop,
    static_validation,
    crop_margin_summary,
    global_min_margin_px,
):
    validation_summary = {
        "static_validation": static_validation,
        "global_min_crop_margin_px": global_min_margin_px,
        "camera_crop_margin_summary": crop_margin_summary,
    }
    payload = {
        "schema_version": "release_roi_metadata_v2",
        "rectification_asset_version": rectification_meta["rectification_asset_version"],
        "image_size": rectification_meta["image_size"],
        "roi_policy_version": ROI_POLICY_VERSION,
        "roi_policy": ROI_POLICY_NAME,
        "roi_generation_method": "mask_intersection_with_margin_and_largest_safe_rectangle",
        "validity_margin_px": {"x": int(margin_x), "y": int(margin_y)},
        "common_valid_mask_bbox": common_valid_mask_bbox,
        "common_valid_roi": safe_rect_roi,
        "safe_rect_roi": safe_rect_roi,
        "final_release_crop_16_9": final_release_crop,
        "crop_policy_version": CROP_POLICY_VERSION,
        "roi_validation_status": "static_pass",
        "roi_validation_summary": validation_summary,
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
    default_margin = int(config["runtime"].get("roi_generation_margin_px", 4))

    parser = argparse.ArgumentParser(description="Generate persistent ROI metadata for public export.")
    parser.add_argument("--batch-name", default=pre_args.batch_name, choices=["secondsyn", "firstsyn"], help="Named batch config to use.")
    parser.add_argument("--rectify-dir", default=default_rectify_dir, help="Directory containing rectify_meta.json and map files.")
    parser.add_argument("--output", default=default_roi_output, help=f"Optional output path. Default: config path or <rectify-dir>/{ROI_METADATA_FILENAME}")
    parser.add_argument("--margin-px", type=int, default=default_margin, help="Safety margin in source-image pixels when computing valid masks.")
    args = parser.parse_args()
    if args.rectify_dir is None:
        raise ValueError("missing rectify_dir; provide --rectify-dir or set paths.rectify_dir in config.py")
    if args.margin_px < 0:
        raise ValueError("--margin-px must be >= 0")

    rectify_dir, rectification_meta = load_rectification_assets(args.rectify_dir)
    common_mask, _, common_valid_mask_bbox, safe_rect_roi = compute_common_valid_roi(
        rectify_dir, rectification_meta, args.margin_px, args.margin_px
    )
    final_release_crop = compute_final_release_crop(safe_rect_roi)
    static_validation = validate_static_roi(common_mask, safe_rect_roi, final_release_crop)
    crop_margin_summary, global_min_margin_px = compute_crop_margin_summary(
        rectify_dir, rectification_meta, final_release_crop
    )
    roi_metadata = build_roi_metadata(
        rectification_meta=rectification_meta,
        margin_x=args.margin_px,
        margin_y=args.margin_px,
        common_valid_mask_bbox=common_valid_mask_bbox,
        safe_rect_roi=safe_rect_roi,
        final_release_crop=final_release_crop,
        static_validation=static_validation,
        crop_margin_summary=crop_margin_summary,
        global_min_margin_px=global_min_margin_px,
    )

    output_path = Path(args.output) if args.output else rectify_dir / ROI_METADATA_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(roi_metadata, handle, indent=4, ensure_ascii=False)

    print(f"Saved ROI metadata to {output_path}")
    print(f"roi_asset_version={roi_metadata['roi_asset_version']}")
    print(f"safe_rect_roi={roi_metadata['safe_rect_roi']}")
    print(f"final_release_crop_16_9={roi_metadata['final_release_crop_16_9']}")


if __name__ == "__main__":
    main()
