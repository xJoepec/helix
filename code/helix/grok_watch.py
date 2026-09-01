"""One-shot Unsloth Studio telemetry for Temporal Helix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError

from .integrations.checkpoints import CheckpointWatcher
from .integrations.unsloth import RunSummary, StudioClient


def _redact_tokens(message: str, tokens: tuple[str | None, ...]) -> str:
    for token in tokens:
        if token:
            message = message.replace(token, "[redacted]")
    return message


def _run_from_detail(detail: dict[str, Any]) -> RunSummary:
    run_payload = detail["run"]
    return RunSummary(
        **{key: run_payload.get(key) for key in RunSummary.__dataclass_fields__}
    )


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    grad_history = metrics.get("grad_norm_history") or []
    current_step = metrics.get("current_step")
    current_loss = metrics.get("current_loss")
    current_lr = metrics.get("current_lr")
    step_history = metrics.get("step_history") or []
    loss_history = metrics.get("loss_history") or []
    lr_history = metrics.get("lr_history") or []
    return {
        "current_step": (
            current_step
            if current_step is not None
            else (step_history[-1] if step_history else None)
        ),
        "current_loss": (
            current_loss
            if current_loss is not None
            else (loss_history[-1] if loss_history else None)
        ),
        "current_lr": (
            current_lr if current_lr is not None else (lr_history[-1] if lr_history else None)
        ),
        "gradient_samples": len(grad_history),
        "last_grad_norm": grad_history[-1] if grad_history else None,
    }


def build_snapshot(client, run_id: str | None = None) -> dict:
    status = client.get_status()
    status_job_id = status.get("job_id")
    if run_id:
        detail = client.get_run(run_id)
        run = _run_from_detail(detail)
    elif status_job_id:
        detail = client.get_run(status_job_id)
        run = _run_from_detail(detail)
    else:
        runs = client.list_runs()
        if not runs:
            raise RuntimeError("Studio has no training runs")
        detail = client.get_run(runs[0].id)
        run = _run_from_detail(detail)

    try:
        live_metrics = client.get_metrics(run.id)
    except (HTTPError, OSError):
        live_metrics = None
    hardware = client.get_hardware()
    inference = client.get_inference_status()
    details = status.get("details") or {}
    use_live_status = status_job_id == run.id
    metrics_job_id = live_metrics.get("job_id") if live_metrics else None
    use_live_metrics = metrics_job_id == run.id
    metrics = live_metrics if use_live_metrics else (detail.get("metrics") or {})
    metric_summary = _metric_summary(metrics)
    if status_job_id and not use_live_status:
        association_state = "mismatch"
    elif not use_live_metrics:
        association_state = "fallback"
    elif not status_job_id:
        association_state = "historical"
    else:
        association_state = "bound"
    training = bool(status.get("is_training_running")) if use_live_status else False
    active_model = inference.get("active_model")
    selected_phase = (
        status.get("phase") if use_live_status else run.status
    )
    checkpoint_refs = (
        CheckpointWatcher().discover(Path(run.output_dir)) if run.output_dir else []
    )
    checkpoints = [ref for ref in checkpoint_refs if ref.stable]
    pending_checkpoints = [ref for ref in checkpoint_refs if not ref.stable]

    def checkpoint_payload(ref):
        return {
            "step": ref.step,
            "path": str(ref.path),
            "fingerprint": ref.fingerprint,
            "stable": ref.stable,
            "stability_reason": ref.stability_reason,
        }

    return {
        "run": {
            "id": run.id,
            "status": run.status,
            "model": run.model_name,
            "dataset": run.dataset_name,
        },
        "status": {
            "phase": selected_phase,
            "step": details.get("step") if use_live_status else metric_summary["current_step"],
            "total_steps": (
                details.get("total_steps")
                if use_live_status
                else detail["run"].get("total_steps")
            ),
            "loss": details.get("loss") if use_live_status else metric_summary["current_loss"],
        },
        "metrics": metric_summary,
        "association": {
            "selected_run_id": run.id,
            "status_job_id": status_job_id,
            "metrics_job_id": metrics_job_id,
            "state": association_state,
        },
        "checkpoints": [checkpoint_payload(ref) for ref in checkpoints],
        "pending_checkpoints": [checkpoint_payload(ref) for ref in pending_checkpoints],
        "hardware": hardware,
        "safety": {
            "training_active": training,
            "active_inference_model": active_model,
            "inference_allowed": not training and active_model is None,
        },
    }


def run_grok_watch(argv=None, *, client_factory: Callable[..., object] = StudioClient) -> int:
    parser = argparse.ArgumentParser(description="Observe Unsloth Studio telemetry")
    parser.add_argument("--studio-url", default="http://127.0.0.1:8888")
    parser.add_argument("--studio-auth", choices=("auto", "keyless", "bearer"), default="auto")
    parser.add_argument("--studio-token")
    parser.add_argument("--run-id")
    parser.add_argument("--once", action="store_true", help="one-shot mode (currently the default)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    tokens = (args.studio_token, os.getenv("UNSLOTH_STUDIO_TOKEN"))
    try:
        client = client_factory(
            args.studio_url,
            token=args.studio_token,
            auth_mode=args.studio_auth,
        )
        snapshot = build_snapshot(client, args.run_id)
    except HTTPError as exc:
        message = f"Studio API HTTP {exc.code}: {_redact_tokens(str(exc.reason), tokens)}"
        print(json.dumps({"error": message}) if args.json else f"error: {message}", file=sys.stderr)
        return 1
    except OSError as exc:
        message = f"Studio API unavailable: {_redact_tokens(str(exc), tokens)}"
        print(json.dumps({"error": message}) if args.json else f"error: {message}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(snapshot, sort_keys=True))
    else:
        status = snapshot["status"]
        print(f"run: {snapshot['run']['id']} ({snapshot['run']['status']})")
        association = snapshot["association"]
        print(
            "association: "
            f"{association['state']} "
            f"selected={association['selected_run_id']} "
            f"status={association['status_job_id']} "
            f"metrics={association['metrics_job_id']}"
        )
        print(f"phase: {status['phase']} step={status['step']}/{status['total_steps']}")
        print(f"loss: {status['loss']}")
        print(f"inference allowed: {snapshot['safety']['inference_allowed']}")
    return 2 if snapshot["association"]["state"] == "mismatch" else 0


__all__ = ["build_snapshot", "run_grok_watch"]
