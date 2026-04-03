import json
from pathlib import Path

from asset_versioning import (
    CROP_POLICY_VERSION,
    EXPORT_PIPELINE_VERSION,
    REFERENCE_CAMERA,
    ROI_POLICY_NAME,
)


def camera_to_view_id(camera_id):
    # 统一把 CAM_A1~CAM_E5 映射成稳定的 view_00~view_24。
    suffix = camera_id.split("_", 1)[1]
    row = suffix[0]
    col = int(suffix[1:])
    row_index = "ABCDE".index(row)
    return f"view_{row_index * 5 + (col - 1):02d}"


def build_color_standardization(profile, lut_info):
    if not profile["apply_color_standardization"]:
        return {
            "enabled": False,
            "policy_name": profile["color_policy_name"],
            "source_color_space": None,
            "target_color_space": None,
            "lut_file": None,
            "lut_path": None,
            "lut_checksum": None,
            "lut_asset_version": None,
        }

    return {
        "enabled": True,
        "policy_name": profile["color_policy_name"],
        "source_color_space": lut_info["source_color_space"],
        "target_color_space": lut_info["target_color_space"],
        "lut_file": lut_info["lut_file"],
        "lut_path": lut_info["lut_path"],
        "lut_checksum": lut_info["lut_checksum"],
        "lut_asset_version": lut_info["lut_asset_version"],
    }


def build_sequence_metadata(
    scene_id,
    source_video,
    batch_name,
    profile,
    roi_metadata,
    rectification_meta,
    phase_offsets,
    final_output_resolution,
    lut_info,
    source_sync_manifest,
    segment_info,
    runtime_info=None,
):
    # sequence metadata 只在一个 scene/segment 的 25 路视角都完成后写出。
    if REFERENCE_CAMERA not in phase_offsets:
        raise ValueError("reference camera missing from phase offsets")
    if float(phase_offsets[REFERENCE_CAMERA]) != 0.0:
        raise ValueError("reference camera phase_offset_ms must be 0.0")

    resize_target = profile["resize_target"] if profile["resize_target"] is not None else "no_resize"
    color_standardization = build_color_standardization(profile, lut_info)
    metadata = {
        "schema_version": "lfv_export_sequence_metadata_v1",
        "scene_id": scene_id,
        "source_video": source_video,
        "batch_name": batch_name,
        "profile": profile["profile"],
        "reference_camera": REFERENCE_CAMERA,
        "fps_nominal": 59.94,
        "output_format": profile["output_format"],
        "bit_depth": profile["bit_depth"],
        "resize_target": resize_target,
        "final_output_resolution": final_output_resolution,
        "roi_policy": ROI_POLICY_NAME,
        "common_valid_roi": roi_metadata["common_valid_roi"],
        "final_release_crop_16_9": roi_metadata["final_release_crop_16_9"],
        "crop_order": profile["crop_order"],
        "crop_policy_version": roi_metadata.get("crop_policy_version", CROP_POLICY_VERSION),
        "rectification_asset_version": rectification_meta["rectification_asset_version"],
        "roi_asset_version": roi_metadata["roi_asset_version"],
        "lut_asset_version": color_standardization["lut_asset_version"],
        "color_standardization": color_standardization,
        "phase_offset_ms": phase_offsets,
        "source_sync_manifest": source_sync_manifest,
        "segment": segment_info,
        "export_pipeline_version": EXPORT_PIPELINE_VERSION,
    }
    if runtime_info is not None:
        metadata["runtime"] = runtime_info
    if profile["output_format"] == "jpeg":
        metadata["jpeg_quality"] = profile["jpeg_quality"]
    else:
        metadata["png_bit_depth"] = profile["png_bit_depth"]
    return metadata


def build_view_metadata(
    camera_id,
    source_video,
    batch_name,
    profile,
    rectification_camera_meta,
    phase_offset_ms,
    frame_count,
    output_resolution,
    roi_asset_version,
    rectification_asset_version,
    crop_policy_version,
    segment_info,
    runtime_info=None,
):
    # 每个 view 的 metadata 只在该视角完整成功后写出。
    payload = {
        "schema_version": "lfv_export_view_metadata_v1",
        "view_id": camera_to_view_id(camera_id),
        "camera_id": camera_id,
        "source_clip": source_video,
        "batch_name": batch_name,
        "rectification_map_file": rectification_camera_meta["map_file"],
        "phase_offset_ms": float(phase_offset_ms),
        "frame_count": int(frame_count),
        "output_format": profile["output_format"],
        "bit_depth": profile["bit_depth"],
        "output_resolution": output_resolution,
        "reference_camera": REFERENCE_CAMERA,
        "rectification_asset_version": rectification_asset_version,
        "roi_asset_version": roi_asset_version,
        "crop_policy_version": crop_policy_version,
        "segment": segment_info,
        "export_pipeline_version": EXPORT_PIPELINE_VERSION,
    }
    if runtime_info is not None:
        payload["runtime"] = runtime_info
    if profile["output_format"] == "jpeg":
        payload["jpeg_quality"] = profile["jpeg_quality"]
    else:
        payload["png_bit_depth"] = profile["png_bit_depth"]
    return payload


def write_json(path, payload):
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)
