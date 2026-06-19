from pathlib import Path

import yaml


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _config_root(path):
    for parent in [path.parent, *path.parents]:
        if parent.name == "configs":
            return parent
    return path.parent


def _resolve_default_path(default_entry, current_path):
    if isinstance(default_entry, str):
        return (current_path.parent / default_entry).with_suffix(".yaml").resolve()

    if isinstance(default_entry, dict):
        if len(default_entry) != 1:
            raise ValueError(f"Default entries must have one key: {default_entry}")
        group, name = next(iter(default_entry.items()))
        return (_config_root(current_path) / group / f"{name}.yaml").resolve()

    raise TypeError(f"Unsupported default entry: {default_entry}")


def _set_nested_value(config, dotted_key, value):
    keys = dotted_key.split(".")
    if any(key == "" for key in keys):
        raise ValueError(f"Invalid override key: {dotted_key}")

    current = config
    for key in keys[:-1]:
        current = current.setdefault(key, {})
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set nested override under non-dict key: {key}")

    current[keys[-1]] = value


def parse_dotlist_overrides(overrides):
    parsed = {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(
                f"CLI override must use key=value format, got: {override}"
            )

        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        _set_nested_value(parsed, key, value)

    return parsed


def load_config(config_path, cli_overrides=None):
    path = Path(config_path).resolve()
    with path.open("r") as f:
        config = yaml.safe_load(f) or {}

    defaults = config.pop("defaults", [])
    overrides = config.pop("overrides", {})
    composed = {}
    for default_entry in defaults:
        default_path = _resolve_default_path(default_entry, path)
        composed = _deep_merge(composed, load_config(default_path))

    composed = _deep_merge(_deep_merge(composed, config), overrides)
    return _deep_merge(composed, parse_dotlist_overrides(cli_overrides))
