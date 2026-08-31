import pytest
from helix.integrations.unsloth import StudioClient


def test_rejects_non_loopback_url() -> None:
    with pytest.raises(ValueError, match="loopback"):
        StudioClient("http://example.com:8888")


def test_lists_runs_without_authorization_header() -> None:
    seen = {}

    def transport(url, headers):
        seen.update(url=url, headers=headers)
        return {
            "runs": [
                {
                    "id": "job_1",
                    "status": "running",
                    "model_name": "m",
                    "dataset_name": "d",
                    "started_at": "now",
                }
            ],
            "total": 1,
        }

    runs = StudioClient("http://127.0.0.1:8888", transport=transport).list_runs(limit=1)
    assert runs[0].id == "job_1"
    assert "Authorization" not in seen["headers"]
    assert "limit=1" in seen["url"]
