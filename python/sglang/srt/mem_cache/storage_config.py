# SPDX-License-Identifier: Apache-2.0
"""Storage configuration parsing shared by startup and runtime attachment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_storage_extra_config(value: str | dict | None) -> dict[str, Any]:
    # FLAT_MEMORY: Argument validation must not construct a cache or load CUDA code.
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError("Storage extra config must be an object, JSON, or @file")
    if not value.startswith("@"):
        result = json.loads(value)
    else:
        path = Path(value[1:])
        if path.suffix == ".json":
            result = json.loads(path.read_text())
        elif path.suffix == ".toml":
            import tomllib

            with path.open("rb") as stream:
                result = tomllib.load(stream)
        elif path.suffix in (".yaml", ".yml"):
            import yaml

            result = yaml.safe_load(path.read_text())
        else:
            raise ValueError(f"Unsupported storage config format: {path.suffix}")
    if not isinstance(result, dict):
        raise ValueError("Storage extra config must contain a JSON object")
    return result


def is_flat_memory_direct(backend: str | None, extra_config: str | dict | None) -> bool:
    return (
        backend == "flat_memory"
        and load_storage_extra_config(extra_config).get("gds_mode", "off") == "compat"
    )
