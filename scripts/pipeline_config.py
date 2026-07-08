"""Small YAML configuration helpers for the pipeline v0 wrappers."""

from pathlib import Path


def load_pipeline_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read pipeline config files: pip install PyYAML") from exc

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"pipeline config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"pipeline config must contain a YAML mapping: {config_path}")
    payload["_config_path"] = str(config_path)
    return payload


def config_value(config, *keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def camera_ids(config):
    rows = config_value(config, "camera", "rows", default=[])
    columns = config_value(config, "camera", "columns", default=[])
    return [f"CAM_{row}{column}" for row in rows for column in columns]

