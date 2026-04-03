import hashlib
import json
from pathlib import Path


EXPORT_PIPELINE_VERSION = "export_pipeline_v1"
CROP_POLICY_VERSION = "crop_policy_v1"
ROI_POLICY_NAME = "common_valid_roi_from_rectification_maps"
CROP_ORDER = "rectify -> common ROI -> final 16:9 crop -> resize"
ROI_METADATA_FILENAME = "release_roi_metadata.json"
REFERENCE_CAMERA = "CAM_C3"


def stable_json_dumps(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_bytes(raw_bytes):
    return hashlib.sha256(raw_bytes).hexdigest()


def hash_file(path):
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(payload):
    return hash_bytes(stable_json_dumps(payload).encode("utf-8"))


def compute_export_signature(payload):
    return hash_json(payload)[:16]


def normalize_rectify_metadata(raw_meta):
    if "cameras" in raw_meta:
        cameras = raw_meta["cameras"]
        image_size = raw_meta.get("image_size")
        rectification_asset_version = raw_meta.get("rectification_asset_version")
        schema_version = raw_meta.get("schema_version", "rectification_meta_v2")
        master_camera = raw_meta.get("master_camera", REFERENCE_CAMERA)
        crop_alpha = raw_meta.get("crop_alpha")
    else:
        cameras = raw_meta
        first_camera = next(iter(cameras.values()), {})
        image_size = first_camera.get("image_size")
        rectification_asset_version = first_camera.get("rectification_asset_version")
        schema_version = "rectification_meta_v1_legacy"
        master_camera = REFERENCE_CAMERA
        crop_alpha = None

    if rectification_asset_version is None:
        payload = {
            "schema_version": schema_version,
            "master_camera": master_camera,
            "image_size": image_size,
            "crop_alpha": crop_alpha,
            "cameras": cameras,
        }
        rectification_asset_version = compute_export_signature(payload)

    return {
        "schema_version": schema_version,
        "master_camera": master_camera,
        "image_size": image_size,
        "crop_alpha": crop_alpha,
        "rectification_asset_version": rectification_asset_version,
        "cameras": cameras,
    }


def build_lut_asset_info(lut_path, source_color, target_color):
    lut_file = Path(lut_path)
    checksum = hash_file(lut_file)
    return {
        "lut_file": lut_file.name,
        "lut_path": str(lut_file),
        "lut_checksum": checksum,
        "lut_asset_version": compute_export_signature(
            {
                "lut_file": lut_file.name,
                "lut_checksum": checksum,
                "source_color": source_color,
                "target_color": target_color,
            }
        ),
        "source_color_space": source_color,
        "target_color_space": target_color,
    }
