"""Read-only client for a loopback Unsloth Studio instance."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

JsonTransport = Callable[[str, dict[str, str]], dict[str, Any]]


@dataclass(frozen=True)
class RunSummary:
    id: str
    status: str
    model_name: str
    dataset_name: str
    started_at: str
    output_dir: str | None = None


def _default_transport(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=15) as response:  # noqa: S310 - loopback validated
        return json.loads(response.read().decode("utf-8"))


class StudioClient:
    """Small read-only API surface; mutation endpoints are intentionally absent."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        auth_mode: str = "auto",
        token_env: str = "UNSLOTH_STUDIO_TOKEN",
        transport: JsonTransport | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("Studio URL must use a loopback host")
        normalized_auth_mode = auth_mode.strip().lower()
        if normalized_auth_mode not in {"auto", "keyless", "bearer"}:
            raise ValueError("Studio auth mode must be auto, keyless, or bearer")
        resolved_token = token if token is not None else os.getenv(token_env)
        if normalized_auth_mode == "bearer" and not resolved_token:
            raise ValueError("Studio bearer authentication requires a token")
        self.base_url = base_url.rstrip("/")
        self._auth_mode = normalized_auth_mode
        self._token = resolved_token
        self._transport = transport or _default_transport

    def _get(self, path: str, query: dict[str, object] | None = None) -> dict[str, Any]:
        suffix = f"?{urlencode(query)}" if query else ""
        headers = {"Accept": "application/json"}
        if self._auth_mode == "bearer" or (self._auth_mode == "auto" and self._token):
            headers["Authorization"] = f"Bearer {self._token}"
        return self._transport(f"{self.base_url}{path}{suffix}", headers)

    def list_runs(self, limit: int = 20, offset: int = 0) -> list[RunSummary]:
        payload = self._get("/api/train/runs", {"limit": limit, "offset": offset})
        return [
            RunSummary(**{key: row.get(key) for key in RunSummary.__dataclass_fields__})
            for row in payload["runs"]
        ]

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/api/train/runs/{run_id}")

    def get_status(self) -> dict[str, Any]:
        return self._get("/api/train/status")

    def get_metrics(self, job_id: str) -> dict[str, Any]:
        return self._get("/api/train/metrics", {"expected_job_id": job_id})

    def get_hardware(self) -> dict[str, Any]:
        return self._get("/api/train/hardware")

    def get_inference_status(self) -> dict[str, Any]:
        return self._get("/api/inference/status")
