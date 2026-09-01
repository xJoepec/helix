import pytest
from helix.integrations.unsloth import StudioClient


def test_rejects_non_loopback_url() -> None:
    with pytest.raises(ValueError, match="loopback"):
        StudioClient("http://example.com:8888")


@pytest.mark.parametrize(
    ("unsafe_url", "message"),
    [
        ("http://user:password@127.0.0.1:8888", "username or password"),
        ("http://127.0.0.1:8888?private=value", "query"),
        ("http://127.0.0.1:8888#private", "fragment"),
    ],
)
def test_rejects_loopback_url_credentials_query_and_fragment(unsafe_url, message) -> None:
    with pytest.raises(ValueError, match=message):
        StudioClient(unsafe_url)


def test_canonicalizes_loopback_url_before_building_request_urls() -> None:
    seen = {}

    def transport(url, headers):
        seen.update(url=url, headers=headers)
        return {"job_id": None}

    client = StudioClient("HTTPS://LOCALHOST:8888/studio", transport=transport)
    client.get_status()

    assert client.base_url == "https://localhost:8888"
    assert seen["url"] == "https://localhost:8888/api/train/status"


def test_keyless_mode_omits_an_explicit_bearer_token_from_requests() -> None:
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

    runs = StudioClient(
        "http://127.0.0.1:8888",
        token="test-token",
        auth_mode="keyless",
        transport=transport,
    ).list_runs(limit=1)
    assert runs[0].id == "job_1"
    assert "Authorization" not in seen["headers"]
    assert "limit=1" in seen["url"]
    assert "test-token" not in seen["url"]


def test_auto_mode_without_a_token_omits_authorization_header(monkeypatch) -> None:
    seen = {}
    monkeypatch.delenv("UNSLOTH_STUDIO_TOKEN", raising=False)

    def transport(url, headers):
        seen.update(url=url, headers=headers)
        return {"runs": [], "total": 0}

    StudioClient("http://127.0.0.1:8888", transport=transport).list_runs()

    assert "Authorization" not in seen["headers"]


def test_auto_mode_uses_an_explicit_token_for_bearer_requests() -> None:
    seen = {}

    def transport(url, headers):
        seen.update(url=url, headers=headers)
        return {"runs": [], "total": 0}

    StudioClient(
        "http://127.0.0.1:8888",
        token="explicit-token",
        transport=transport,
    ).list_runs()

    assert seen["headers"]["Authorization"] == "Bearer explicit-token"
    assert "explicit-token" not in seen["url"]


def test_keyless_mode_ignores_a_token_from_the_environment(monkeypatch) -> None:
    seen = {}
    monkeypatch.setenv("UNSLOTH_STUDIO_TOKEN", "environment-token")

    def transport(url, headers):
        seen.update(url=url, headers=headers)
        return {"runs": [], "total": 0}

    StudioClient(
        "http://127.0.0.1:8888",
        auth_mode="keyless",
        transport=transport,
    ).list_runs()

    assert "Authorization" not in seen["headers"]


def test_explicit_bearer_mode_sends_normalized_authorization_header() -> None:
    seen = {}

    def transport(url, headers):
        seen.update(url=url, headers=headers)
        return {"runs": [], "total": 0}

    StudioClient(
        "http://127.0.0.1:8888",
        token="test-token",
        auth_mode=" BEARER ",
        transport=transport,
    ).list_runs()

    assert seen["headers"]["Authorization"] == "Bearer test-token"
    assert "test-token" not in seen["url"]


def test_auto_mode_uses_token_from_the_configured_environment(monkeypatch) -> None:
    seen = {}
    monkeypatch.setenv("UNSLOTH_STUDIO_TOKEN", "environment-token")

    def transport(url, headers):
        seen.update(url=url, headers=headers)
        return {"runs": [], "total": 0}

    StudioClient("http://127.0.0.1:8888", transport=transport).list_runs()

    assert seen["headers"]["Authorization"] == "Bearer environment-token"
    assert "environment-token" not in seen["url"]


def test_bearer_mode_requires_a_nonempty_token() -> None:
    with pytest.raises(ValueError, match="requires a token"):
        StudioClient("http://127.0.0.1:8888", auth_mode="bearer")


def test_bearer_mode_rejects_a_whitespace_only_token() -> None:
    with pytest.raises(ValueError, match="requires a token"):
        StudioClient("http://127.0.0.1:8888", token="   ", auth_mode="bearer")
