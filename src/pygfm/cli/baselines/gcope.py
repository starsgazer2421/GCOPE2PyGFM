"""PyGFM registry hooks for GCOPE."""

from __future__ import annotations

from typing import Any


def _run(stage: str, cfg: dict[str, Any]) -> None:
    from pygfm.baseline_models.gcope.runner import run_from_config

    run_from_config(cfg, default_stage=stage)


RUNNERS = {
    "pretrain": lambda cfg: _run("pretrain", cfg),
    "finetune": lambda cfg: _run("finetune", cfg),
    "prog": lambda cfg: _run("prog", cfg),
    "ete": lambda cfg: _run("ete", cfg),
}
