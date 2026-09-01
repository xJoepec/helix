import json
import os
import sys
import time
from urllib.error import HTTPError

import pytest
from helix.grok_watch import build_snapshot, run_grok_watch
from helix.integrations.unsloth import RunSummary


class FakeClient:
    def list_runs(self, limit=20, offset=0):
        return [RunSummary("job_1", "running", "model", "dataset", "now", None)]

    def get_run(self, run_id):
        assert run_id == "job_1"
        return {
            "run": {
                "id": "job_1",
                "status": "running",
                "model_name": "model",
                "dataset_name": "dataset",
                "started_at": "now",
                "output_dir": None,
            },
            "config": {},
            "metrics": {},
        }

    def get_status(self):
        return {
            "job_id": "job_1",
            "phase": "training",
            "is_training_running": True,
            "details": {"step": 7, "total_steps": 100, "loss": 1.2},
        }

    def get_metrics(self, job_id):
        return {
            "job_id": "job_1",
            "current_step": 7,
            "current_loss": 1.2,
            "grad_norm_history": [0.5],
        }

    def get_hardware(self):
        return {"devices": [{"vram_used_gb": 4.0, "vram_total_gb": 8.0}]}

    def get_inference_status(self):
        return {"active_model": None}


def test_snapshot_marks_inference_as_blocked_during_training() -> None:
    snapshot = build_snapshot(FakeClient())
    assert snapshot["run"]["id"] == "job_1"
    assert snapshot["safety"]["inference_allowed"] is False


def test_idle_status_with_null_details_has_empty_progress() -> None:
    class IdleClient(FakeClient):
        def get_status(self):
            return {
                "job_id": None,
                "phase": "idle",
                "is_training_running": False,
                "details": None,
            }

        def get_metrics(self, job_id):
            return {"job_id": None}

    snapshot = build_snapshot(IdleClient())
    assert snapshot["status"] == {
        "phase": "running",
        "step": None,
        "total_steps": None,
        "loss": None,
    }


def test_json_cli_output(capsys) -> None:
    assert run_grok_watch(["--json"], client_factory=lambda _, **__: FakeClient()) == 0
    assert json.loads(capsys.readouterr().out)["status"]["step"] == 7


def test_cli_forwards_auth_mode_and_token_only_as_client_keywords(capsys) -> None:
    seen = {}

    def client_factory(base_url, *, token=None, auth_mode="auto"):
        seen.update(base_url=base_url, token=token, auth_mode=auth_mode)
        return FakeClient()

    assert (
        run_grok_watch(
            [
                "--studio-url",
                "http://localhost:8888",
                "--studio-auth",
                "bearer",
                "--studio-token",
                "test-token",
                "--json",
            ],
            client_factory=client_factory,
        )
        == 0
    )

    assert seen == {
        "base_url": "http://localhost:8888",
        "token": "test-token",
        "auth_mode": "bearer",
    }
    assert "test-token" not in capsys.readouterr().out


def test_cli_redacts_bearer_token_from_connection_errors(capsys) -> None:
    class FailingClient:
        def get_status(self):
            raise OSError("Studio rejected test-token")

    assert (
        run_grok_watch(
            ["--studio-auth", "bearer", "--studio-token", "test-token", "--json"],
            client_factory=lambda _, **__: FailingClient(),
        )
        == 1
    )

    captured = capsys.readouterr()
    assert "test-token" not in captured.err
    assert json.loads(captured.err) == {
        "error": "Studio API unavailable: Studio rejected [redacted]"
    }


@pytest.mark.parametrize("json_mode", [False, True])
def test_cli_reports_missing_bearer_token_without_traceback(json_mode, capsys, monkeypatch) -> None:
    monkeypatch.delenv("UNSLOTH_STUDIO_TOKEN", raising=False)
    args = ["--studio-auth", "bearer"]
    if json_mode:
        args.append("--json")

    assert run_grok_watch(args) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    if json_mode:
        assert json.loads(captured.err) == {
            "error": "Studio configuration invalid: Studio bearer authentication requires a token"
        }
    else:
        assert captured.err == (
            "error: Studio configuration invalid: "
            "Studio bearer authentication requires a token\n"
        )


def test_console_main_uses_process_arguments(monkeypatch) -> None:
    from helix import cli, grok_watch

    seen = {}
    monkeypatch.setattr(sys, "argv", ["helix", "grok-watch", "--json"])

    def fake_run(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(grok_watch, "run_grok_watch", fake_run)
    assert cli.main() == 0
    assert seen["argv"] == ["--json"]


def test_historical_run_uses_persisted_metrics() -> None:
    class HistoricalClient(FakeClient):
        def list_runs(self, limit=20, offset=0):
            return [RunSummary("job_1", "completed", "model", "dataset", "now", None)]

        def get_status(self):
            return {
                "job_id": None,
                "phase": "idle",
                "is_training_running": False,
                "details": None,
            }

        def get_metrics(self, job_id):
            raise OSError("live metrics are unavailable for completed runs")

        def get_run(self, run_id):
            return {
                "run": {
                    "id": "job_1",
                    "status": "completed",
                    "model_name": "model",
                    "dataset_name": "dataset",
                    "started_at": "now",
                    "output_dir": None,
                    "total_steps": 42,
                },
                "config": {},
                "metrics": {
                    "step_history": [20, 42],
                    "loss_history": [0.5, 0.25],
                    "lr_history": [0.001, 0.0005],
                },
            }

    snapshot = build_snapshot(HistoricalClient(), "job_1")
    assert snapshot["metrics"]["current_step"] == 42
    assert snapshot["metrics"]["current_loss"] == 0.25
    assert snapshot["metrics"]["current_lr"] == 0.0005
    assert snapshot["status"]["phase"] == "completed"
    assert snapshot["status"]["step"] == 42
    assert snapshot["status"]["total_steps"] == 42
    assert snapshot["status"]["loss"] == 0.25


def test_explicit_run_uses_direct_lookup_beyond_first_page() -> None:
    class SelectedClient(FakeClient):
        def list_runs(self, limit=20, offset=0):
            raise AssertionError("explicit lookup must not depend on the first page")

        def get_run(self, run_id):
            assert run_id == "job_older"
            return {
                "run": {
                    "id": "job_older",
                    "status": "completed",
                    "model_name": "older-model",
                    "dataset_name": "older-dataset",
                    "started_at": "yesterday",
                    "output_dir": None,
                },
                "config": {},
                "metrics": {
                    "step_history": [100],
                    "loss_history": [0.125],
                    "lr_history": [0.0001],
                },
            }

        def get_status(self):
            return {
                "job_id": None,
                "phase": "idle",
                "is_training_running": False,
                "details": None,
            }

        def get_metrics(self, job_id):
            raise OSError("historical run has no live metrics")

    snapshot = build_snapshot(SelectedClient(), "job_older")
    assert snapshot["run"] == {
        "id": "job_older",
        "status": "completed",
        "model": "older-model",
        "dataset": "older-dataset",
    }


def test_implicit_selection_follows_status_job_id() -> None:
    class StatusBoundClient(FakeClient):
        def list_runs(self, limit=20, offset=0):
            return [RunSummary("job_a", "completed", "old", "old-data", "earlier", None)]

        def get_status(self):
            return {
                "job_id": "job_b",
                "phase": "training",
                "is_training_running": True,
                "details": {"step": 8, "total_steps": 100, "loss": 0.8},
            }

        def get_run(self, run_id):
            assert run_id == "job_b"
            return {
                "run": {
                    "id": "job_b",
                    "status": "running",
                    "model_name": "new",
                    "dataset_name": "new-data",
                    "started_at": "now",
                    "output_dir": None,
                },
                "config": {},
                "metrics": {},
            }

    snapshot = build_snapshot(StatusBoundClient())

    assert snapshot["run"]["id"] == "job_b"


def test_explicit_run_preserves_its_persisted_state_when_status_names_another_run() -> None:
    class ExplicitClient(FakeClient):
        def get_status(self):
            return {
                "job_id": "job_b",
                "phase": "training",
                "is_training_running": True,
                "details": {"step": 99, "total_steps": 100, "loss": 9.9},
            }

        def get_run(self, run_id):
            assert run_id == "job_a"
            return {
                "run": {
                    "id": "job_a",
                    "status": "completed",
                    "model_name": "archived",
                    "dataset_name": "archive",
                    "started_at": "yesterday",
                    "output_dir": None,
                },
                "config": {},
                "metrics": {
                    "step_history": [12],
                    "loss_history": [0.12],
                    "lr_history": [0.00012],
                },
            }

        def get_metrics(self, job_id):
            return {"job_id": "job_b", "current_step": 99, "current_loss": 9.9}

    snapshot = build_snapshot(ExplicitClient(), "job_a")

    assert snapshot["run"]["id"] == "job_a"
    assert snapshot["status"] == {
        "phase": "completed",
        "step": 12,
        "total_steps": None,
        "loss": 0.12,
    }
    assert snapshot["association"] == {
        "selected_run_id": "job_a",
        "status_job_id": "job_b",
        "metrics_job_id": "job_b",
        "state": "mismatch",
    }


@pytest.mark.parametrize("metrics_job_id", [None, "job_b"])
def test_unidentified_or_mismatched_live_metrics_fall_back_to_selected_run_history(
    metrics_job_id,
) -> None:
    class MetricsClient(FakeClient):
        def get_status(self):
            return {
                "job_id": "job_a",
                "phase": "training",
                "is_training_running": True,
                "details": {"step": 5, "total_steps": 50, "loss": 0.5},
            }

        def get_run(self, run_id):
            assert run_id == "job_a"
            return {
                "run": {
                    "id": "job_a",
                    "status": "running",
                    "model_name": "model",
                    "dataset_name": "dataset",
                    "started_at": "now",
                    "output_dir": None,
                },
                "config": {},
                "metrics": {
                    "step_history": [5],
                    "loss_history": [0.5],
                    "lr_history": [0.0005],
                },
            }

        def get_metrics(self, job_id):
            return {
                "job_id": metrics_job_id,
                "current_step": 99,
                "current_loss": 9.9,
                "current_lr": 0.0099,
            }

    snapshot = build_snapshot(MetricsClient(), "job_a")

    assert snapshot["metrics"]["current_step"] == 5
    assert snapshot["metrics"]["current_loss"] == 0.5
    assert snapshot["metrics"]["current_lr"] == 0.0005
    assert snapshot["association"]["state"] == "fallback"


def test_checkpoints_are_discovered_only_from_the_selected_run_output_dir(tmp_path) -> None:
    selected_output = tmp_path / "selected"
    selected_output.mkdir()
    selected_checkpoint = selected_output / "checkpoint-10"
    selected_checkpoint.mkdir()
    selected_weights = selected_checkpoint / "model.safetensors"
    selected_weights.write_bytes(b"selected")
    (selected_checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 10}), encoding="utf-8"
    )
    unrelated_output = tmp_path / "unrelated"
    unrelated_output.mkdir()
    unrelated_checkpoint = unrelated_output / "checkpoint-99"
    unrelated_checkpoint.mkdir()
    unrelated_weights = unrelated_checkpoint / "model.safetensors"
    unrelated_weights.write_bytes(b"unrelated")
    (unrelated_checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 99}), encoding="utf-8"
    )
    old = time.time() - 60
    for path in (
        selected_weights,
        selected_checkpoint / "trainer_state.json",
        unrelated_weights,
        unrelated_checkpoint / "trainer_state.json",
    ):
        os.utime(path, (old, old))

    class CheckpointClient(FakeClient):
        def get_status(self):
            return {
                "job_id": "job_b",
                "phase": "training",
                "is_training_running": True,
                "details": {"step": 99, "total_steps": 100, "loss": 9.9},
            }

        def get_run(self, run_id):
            assert run_id == "job_a"
            return {
                "run": {
                    "id": "job_a",
                    "status": "completed",
                    "model_name": "model",
                    "dataset_name": "dataset",
                    "started_at": "yesterday",
                    "output_dir": str(selected_output),
                },
                "config": {"output_dir": str(unrelated_output)},
                "metrics": {},
            }

        def get_metrics(self, job_id):
            return {"job_id": "job_b"}

    snapshot = build_snapshot(CheckpointClient(), "job_a")

    assert [checkpoint["path"] for checkpoint in snapshot["checkpoints"]] == [
        str(selected_checkpoint)
    ]


def test_snapshot_discovers_complete_checkpoints_from_output_dir(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    shard = checkpoint / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": shard.name}}), encoding="utf-8"
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 12}), encoding="utf-8"
    )
    old = time.time() - 60
    for path in (
        shard,
        checkpoint / "model.safetensors.index.json",
        checkpoint / "trainer_state.json",
    ):
        os.utime(path, (old, old))

    class CheckpointClient(FakeClient):
        def list_runs(self, limit=20, offset=0):
            return [RunSummary("job_1", "running", "model", "dataset", "now", str(tmp_path))]

        def get_run(self, run_id):
            detail = super().get_run(run_id)
            detail["run"]["output_dir"] = str(tmp_path)
            return detail

    snapshot = build_snapshot(CheckpointClient())
    assert len(snapshot["checkpoints"]) == 1
    assert snapshot["checkpoints"][0]["step"] == 12
    assert snapshot["checkpoints"][0]["path"] == str(checkpoint)


def test_snapshot_separates_stable_and_pending_checkpoints(tmp_path) -> None:
    stable = tmp_path / "checkpoint-10"
    stable.mkdir()
    stable_model = stable / "model.safetensors"
    stable_model.write_bytes(b"stable")
    (stable / "trainer_state.json").write_text(
        json.dumps({"global_step": 10}), encoding="utf-8"
    )
    pending = tmp_path / "checkpoint-11"
    pending.mkdir()
    pending_model = pending / "model.safetensors"
    pending_model.write_bytes(b"pending")
    (pending / "trainer_state.json").write_text(
        json.dumps({"global_step": 11}), encoding="utf-8"
    )
    old = time.time() - 60
    for path in (stable_model, stable / "trainer_state.json"):
        os.utime(path, (old, old))

    class CheckpointClient(FakeClient):
        def list_runs(self, limit=20, offset=0):
            return [RunSummary("job_1", "running", "model", "dataset", "now", str(tmp_path))]

        def get_run(self, run_id):
            detail = super().get_run(run_id)
            detail["run"]["output_dir"] = str(tmp_path)
            return detail

    snapshot = build_snapshot(CheckpointClient())

    assert [item["step"] for item in snapshot["checkpoints"]] == [10]
    assert [item["step"] for item in snapshot["pending_checkpoints"]] == [11]
    assert snapshot["checkpoints"][0]["stable"] is True
    assert snapshot["pending_checkpoints"][0]["stable"] is False


def test_human_cli_reports_concise_http_error(capsys) -> None:
    class FailingClient:
        def get_status(self):
            raise HTTPError(
                "http://127.0.0.1:8888/api/train/status?private=query",
                503,
                "Service Unavailable",
                None,
                None,
            )

    assert run_grok_watch([], client_factory=lambda _, **__: FailingClient()) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: Studio API HTTP 503: Service Unavailable\n"


def test_json_cli_reports_concise_connection_error(capsys) -> None:
    class FailingClient:
        def get_status(self):
            raise OSError("connection refused")

    assert run_grok_watch(["--json"], client_factory=lambda _, **__: FailingClient()) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"error": "Studio API unavailable: connection refused"}
