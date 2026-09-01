import json
import sys

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

        def get_metrics(self, job_id):
            raise OSError("live metrics are unavailable for completed runs")

        def get_run(self, run_id):
            return {"metrics": {"current_step": 42, "current_loss": 0.25}}

    snapshot = build_snapshot(HistoricalClient())
    assert snapshot["metrics"]["current_step"] == 42
