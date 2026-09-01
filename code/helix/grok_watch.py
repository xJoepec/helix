"""One-shot keyless Unsloth Studio telemetry for Temporal Helix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError

from .integrations.checkpoints import CheckpointWatcher
from .integrations.unsloth import RunSummary, StudioClient


def build_snapshot(client, run_id: str | None = None) -> dict:
    detail = {}
    if run_id:
        detail = client.get_run(run_id)
        run = RunSummary(
            **{key: detail.get(key) for key in RunSummary.__dataclass_fields__}
        )
    else:
        runs = client.list_runs()
        if not runs:
            raise RuntimeError("Studio has no training runs")
        run = runs[0]
    status = client.get_status()
    try:
        metrics = client.get_metrics(run.id)
    except (HTTPError, OSError):
        detail = detail or client.get_run(run.id)
        metrics = detail.get("metrics", {})
    hardware = client.get_hardware()
    inference = client.get_inference_status()
    details = status.get("details") or {}
    training = bool(status.get("is_training_running"))
    active_model = inference.get("active_model")
    grad_history = metrics.get("grad_norm_history") or []
    current_step = metrics.get("current_step")
    current_loss = metrics.get("current_loss")
    current_lr = metrics.get("current_lr")
    step_history = metrics.get("step_history") or []
    loss_history = metrics.get("loss_history") or []
    lr_history = metrics.get("lr_history") or []
    current_step = (
        current_step
        if current_step is not None
        else (step_history[-1] if step_history else None)
    )
    current_loss = (
        current_loss
        if current_loss is not None
        else (loss_history[-1] if loss_history else None)
    )
    current_lr = current_lr if current_lr is not None else (lr_history[-1] if lr_history else None)
    metric_summary = {
        "current_step": current_step,
        "current_loss": current_loss,
        "current_lr": current_lr,
        "gradient_samples": len(grad_history),
        "last_grad_norm": grad_history[-1] if grad_history else None,
    }
    use_live_status = not run_id or status.get("job_id") == run.id
    selected_phase = (
        status.get("phase") if use_live_status else detail.get("status", run.status)
    )
    checkpoints = CheckpointWatcher().discover(Path(run.output_dir)) if run.output_dir else []
    return {
        "run": {
            "id": run.id,
            "status": run.status,
            "model": run.model_name,
            "dataset": run.dataset_name,
        },
        "status": {
            "phase": selected_phase,
            "step": details.get("step") if use_live_status else current_step,
            "total_steps": (
                details.get("total_steps")
                if use_live_status
                else detail.get("total_steps")
            ),
            "loss": details.get("loss") if use_live_status else current_loss,
        },
        "metrics": metric_summary,
        "checkpoints": [
            {"step": ref.step, "path": str(ref.path), "fingerprint": ref.fingerprint}
            for ref in checkpoints
        ],
        "hardware": hardware,
        "safety": {
            "training_active": training,
            "active_inference_model": active_model,
            "inference_allowed": not training and active_model is None,
        },
    }


def run_grok_watch(argv=None, *, client_factory: Callable[[str], object] = StudioClient) -> int:
    parser = argparse.ArgumentParser(description="Observe keyless Unsloth Studio telemetry")
    parser.add_argument("--studio-url", default="http://127.0.0.1:8888")
    parser.add_argument("--run-id")
    parser.add_argument("--once", action="store_true", help="one-shot mode (currently the default)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        snapshot = build_snapshot(client_factory(args.studio_url), args.run_id)
    except HTTPError as exc:
        message = f"Studio API HTTP {exc.code}: {exc.reason}"
        print(json.dumps({"error": message}) if args.json else f"error: {message}", file=sys.stderr)
        return 1
    except OSError as exc:
        message = f"Studio API unavailable: {exc}"
        print(json.dumps({"error": message}) if args.json else f"error: {message}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(snapshot, sort_keys=True))
    else:
        status = snapshot["status"]
        print(f"run: {snapshot['run']['id']} ({snapshot['run']['status']})")
        print(f"phase: {status['phase']} step={status['step']}/{status['total_steps']}")
        print(f"loss: {status['loss']}")
        print(f"inference allowed: {snapshot['safety']['inference_allowed']}")
    return 0


__all__ = ["build_snapshot", "run_grok_watch"]
