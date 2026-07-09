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
    normalize_pipeline_config(payload)
    return payload


def normalize_pipeline_config(config):
    """Normalize legacy and generic batch config keys in-place."""
    if "batch_id" not in config and "batch_name" in config:
        config["batch_id"] = config["batch_name"]
    if "batch_name" not in config and "batch_id" in config:
        config["batch_name"] = config["batch_id"]

    runtime = config.setdefault("runtime", {})
    if "batch_id" not in runtime and config.get("batch_id") is not None:
        runtime["batch_id"] = config["batch_id"]
    if "batch_name" not in runtime and config.get("batch_id") is not None:
        runtime["batch_name"] = config["batch_id"]
    if "time_key" not in runtime and config.get("batch_id") is not None:
        runtime["time_key"] = config["batch_id"]
    return config


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


def batch_id(config):
    return str(config_value(config, "batch_id", default=config_value(config, "batch_name", default="batch")))


def time_key(config):
    return str(config_value(config, "runtime", "time_key", default=batch_id(config)))
