"""PyGFM runner adapter for the original GCOPE implementation.

The original GCOPE code is a ``fastargs`` project driven by ``src/exec.py`` and
flat command-line keys such as ``--general.func``.  PyGFM uses YAML files with
``baseline`` and ``stage``.  This module keeps the original algorithm code
intact and translates a PyGFM config into the argument vector expected by
GCOPE.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_SKIP_KEYS = {"baseline", "stage", "params"}

_STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "pretrain": {"general": {"func": "pretrain"}},
    "finetune": {"general": {"func": "adapt"}, "adapt": {"method": "finetune"}},
    "prog": {"general": {"func": "adapt"}, "adapt": {"method": "prog"}},
    "ete": {"general": {"func": "ete"}},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _normalise_config(cfg: dict[str, Any], default_stage: str | None = None) -> dict[str, Any]:
    params = cfg.get("params")
    merged = {k: v for k, v in cfg.items() if k != "params"}
    if isinstance(params, dict):
        merged = _deep_merge(params, merged)

    stage = str(merged.get("stage") or default_stage or "").strip().lower()
    if not stage:
        func = ((merged.get("general") or {}).get("func") if isinstance(merged.get("general"), dict) else None)
        method = ((merged.get("adapt") or {}).get("method") if isinstance(merged.get("adapt"), dict) else None)
        if func == "pretrain":
            stage = "pretrain"
        elif func == "ete":
            stage = "ete"
        elif func == "adapt" and method == "prog":
            stage = "prog"
        elif func == "adapt":
            stage = "finetune"
        else:
            stage = "pretrain"

    if stage not in _STAGE_DEFAULTS:
        raise ValueError(f"Unsupported GCOPE stage: {stage!r}. Expected one of {sorted(_STAGE_DEFAULTS)}")

    merged = _deep_merge(_STAGE_DEFAULTS[stage], merged)
    merged["baseline"] = "gcope"
    merged["stage"] = stage
    merged.setdefault("general", {})
    if isinstance(merged["general"], dict):
        merged["general"].setdefault("save_dir", f"storage/gcope/{stage}")
        os.makedirs(str(merged["general"]["save_dir"]), exist_ok=True)
    return merged


def _format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _flatten(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten(child_prefix, child))
        return items
    return [(prefix, value)]


def _config_to_argv(cfg: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in cfg.items():
        if key in _SKIP_KEYS:
            continue
        for flat_key, flat_value in _flatten(str(key), value):
            if flat_value is None:
                continue
            argv.extend([f"--{flat_key}", _format_cli_value(flat_value)])
    return argv


def run_from_config(cfg: dict[str, Any], default_stage: str | None = None) -> None:
    cfg = _normalise_config(cfg, default_stage=default_stage)
    original_src = Path(__file__).resolve().parent / "original_src"
    exec_path = original_src / "exec.py"
    if not exec_path.is_file():
        raise FileNotFoundError(f"GCOPE original entrypoint not found: {exec_path}")

    original_src_str = str(original_src)
    if original_src_str not in sys.path:
        sys.path.insert(0, original_src_str)

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(exec_path)] + _config_to_argv(cfg)
        module_name = "_pygfm_gcope_original_exec"
        spec = importlib.util.spec_from_file_location(module_name, exec_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import GCOPE exec.py from {exec_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.config.collect_argparse_args(module.parser)
        module.config.validate()
        module.config.get_all_config(dump_path=os.path.join(module.config["general.save_dir"], "config.json"))
        module.run()
    finally:
        sys.argv = old_argv


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("GCOPE YAML root must be a mapping.")
    return data


def main_from_cli(default_stage: str | None = None, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run GCOPE through the PyGFM adapter.")
    parser.add_argument("-c", "--config", "--yaml-config", dest="config", required=True)
    args = parser.parse_args(argv)
    run_from_config(load_yaml(args.config), default_stage=default_stage)
