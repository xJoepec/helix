"""One-shot keyless Unsloth Studio telemetry for Temporal Helix."""

from __future__ import annotations

import argparse
import json
from typing import Callable

from .integrations.unsloth import StudioClient


def build_snapshot(client, run_id: str | None = None) -> dict:
    runs = client.list_runs()
    if not runs:
        raise RuntimeError("Studio has no training runs")
    run = next((item for item in runs if item.id == run_id), None) if run_id else runs[0]
    if run is None:
        raise ValueError(f"run not found: {run_id}")
    status = client.get_status()
    metrics = client.get_metrics(run.id)
    hardware = client.get_hardware()
    inference = client.get_inference_status()
    training = bool(status.get("is_training_running"))
    active_model = inference.get("active_model")
    grad_history = metrics.get("grad_norm_history") or []
    metric_summary = {
        "current_step": metrics.get("current_step"),
        "current_loss": metrics.get("current_loss"),
        "current_lr": metrics.get("current_lr"),
        "gradient_samples": len(grad_history),
        "last_grad_norm": grad_history[-1] if grad_history else None,
    }
    return {
        "run": {
            "id": run.id,
            "status": run.status,
            "model": run.model_name,
            "dataset": run.dataset_name,
        },
        "status": {
            "phase": status.get("phase"),
            "step": status.get("details", {}).get("step"),
            "total_steps": status.get("details", {}).get("total_steps"),
            "loss": status.get("details", {}).get("loss"),
        },
        "metrics": metric_summary,
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
    snapshot = build_snapshot(client_factory(args.studio_url), args.run_id)
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
