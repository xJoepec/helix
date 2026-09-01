import json
import sys
from urllib.error import HTTPError

from helix.grok_watch import build_snapshot, run_grok_watch
from helix.integrations.unsloth import RunSummary


class FakeClient:
    def list_runs(self, limit=20, offset=0):
        return [RunSummary("job_1", "running", "model", "dataset", "now", None)]

    def get_status(self):
        return {
            "job_id": "job_1",
            "phase": "training",
            "is_training_running": True,
            "details": {"step": 7, "total_steps": 100, "loss": 1.2},
        }

    def get_metrics(self, job_id):
        return {"current_step": 7, "current_loss": 1.2, "grad_norm_history": [0.5]}

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

    snapshot = build_snapshot(IdleClient())
    assert snapshot["status"] == {
        "phase": "idle",
        "step": None,
        "total_steps": None,
        "loss": None,
    }


def test_json_cli_output(capsys) -> None:
    assert run_grok_watch(["--json"], client_factory=lambda _: FakeClient()) == 0
    assert json.loads(capsys.readouterr().out)["status"]["step"] == 7


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
                "id": "job_1",
                "status": "completed",
                "model_name": "model",
                "dataset_name": "dataset",
                "started_at": "now",
                "output_dir": None,
                "total_steps": 42,
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
                "id": "job_older",
                "status": "completed",
                "model_name": "older-model",
                "dataset_name": "older-dataset",
                "started_at": "yesterday",
                "output_dir": None,
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


def test_snapshot_discovers_complete_checkpoints_from_output_dir(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    shard = checkpoint / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": shard.name}}), encoding="utf-8"
    )

    class CheckpointClient(FakeClient):
        def list_runs(self, limit=20, offset=0):
            return [RunSummary("job_1", "running", "model", "dataset", "now", str(tmp_path))]

    snapshot = build_snapshot(CheckpointClient())
    assert len(snapshot["checkpoints"]) == 1
    assert snapshot["checkpoints"][0]["step"] == 12
    assert snapshot["checkpoints"][0]["path"] == str(checkpoint)


def test_human_cli_reports_concise_http_error(capsys) -> None:
    class FailingClient:
        def list_runs(self, limit=20, offset=0):
            raise HTTPError(
                "http://127.0.0.1:8888/api/train/runs?private=query",
                503,
                "Service Unavailable",
                None,
                None,
            )

    assert run_grok_watch([], client_factory=lambda _: FailingClient()) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: Studio API HTTP 503: Service Unavailable\n"


def test_json_cli_reports_concise_connection_error(capsys) -> None:
    class FailingClient:
        def list_runs(self, limit=20, offset=0):
            raise OSError("connection refused")

    assert run_grok_watch(["--json"], client_factory=lambda _: FailingClient()) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"error": "Studio API unavailable: connection refused"}
