from copy import deepcopy

from asset_versioning import CROP_ORDER


# 两个 profile 共享同一条几何主路径，只在输出格式、位深和是否 resize 上区分。
PROFILES = {
    "benchmark": {
        "profile": "benchmark",
        "output_format": "jpeg",
        "bit_depth": 8,
        "resize_target": [1920, 1080],
        "jpeg_quality": 97,
        "png_bit_depth": None,
        "apply_color_standardization": True,
        "color_policy_name": "dlogm_to_rec709_lut",
        "crop_order": CROP_ORDER,
        "no_resize": False,
    },
    "fidelity": {
        "profile": "fidelity",
        "output_format": "png",
        "bit_depth": 16,
        "resize_target": None,
        "jpeg_quality": None,
        "png_bit_depth": 16,
        "apply_color_standardization": True,
        "color_policy_name": "dlogm_to_rec709_lut",
        "crop_order": CROP_ORDER,
        "no_resize": True,
    },
}


def get_export_profile(name):
    if name not in PROFILES:
        raise ValueError(f"unsupported export profile: {name}")
    # 返回副本，避免运行时修改污染全局默认配置。
    return deepcopy(PROFILES[name])
