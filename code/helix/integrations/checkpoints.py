"""Discover complete Hugging Face checkpoints without mutating them."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointRef:
    step: int
    path: Path
    shard_paths: tuple[Path, ...]
    fingerprint: str
    stable: bool = False
    stability_reason: str = "recent"


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _is_save_in_progress_artifact(path: Path) -> bool:
    name = path.name.casefold()
    return name.startswith(("tmp", ".tmp", "partial-", ".partial-")) or name.endswith(
        (".tmp", ".temp", ".part", ".partial", ".incomplete")
    )


class CheckpointWatcher:
    def __init__(self, min_age_seconds: float = 2.0) -> None:
        if not math.isfinite(min_age_seconds) or min_age_seconds < 0:
            raise ValueError("min_age_seconds must be finite and nonnegative")
        self.min_age_seconds = min_age_seconds

    def discover(
        self, output_dir: Path, *, stable_only: bool = False
    ) -> list[CheckpointRef]:
        refs = []
        output_root = output_dir.resolve()
        for path in output_dir.glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            root = path.resolve()
            if not match or not path.is_dir() or root.parent != output_root:
                continue

            try:
                if any(_is_save_in_progress_artifact(item) for item in path.iterdir()):
                    continue
            except OSError:
                continue

            index_path = path / "model.safetensors.index.json"
            adapter_path = path / "adapter_model.safetensors"
            trainer_state_path = path / "trainer_state.json"
            model_path = path / "model.safetensors"

            if (
                not trainer_state_path.is_file()
                or trainer_state_path.is_symlink()
            ):
                continue
            try:
                trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
            except (TypeError, ValueError, OSError):
                continue
            if not isinstance(trainer_state, dict):
                continue
            trainer_step = trainer_state.get("global_step")
            if trainer_step is not None and (
                isinstance(trainer_step, bool)
                or not isinstance(trainer_step, int)
                or trainer_step != int(match.group(1))
            ):
                continue

            required_paths = [trainer_state_path]
            if index_path.exists():
                if not index_path.is_file() or index_path.is_symlink():
                    continue
                try:
                    data = json.loads(index_path.read_text(encoding="utf-8"))
                    weight_map = data["weight_map"]
                    if not isinstance(weight_map, dict):
                        continue
                    names = sorted(set(weight_map.values()))
                except (KeyError, TypeError, ValueError, OSError):
                    continue
                if not names or any(
                    not isinstance(name, str)
                    or Path(name).parent != Path()
                    or Path(name).suffix != ".safetensors"
                    for name in names
                ):
                    continue
                shards = tuple((path / name).resolve() for name in names)
                if any(shard.parent != root or not shard.is_file() for shard in shards):
                    continue
                if any((path / name).is_symlink() for name in names):
                    continue
                required_paths.extend([index_path, *shards])
            elif adapter_path.exists():
                if not adapter_path.is_file() or adapter_path.is_symlink():
                    continue
                shards = (adapter_path.resolve(),)
                required_paths.append(adapter_path)
            elif model_path.is_file():
                if model_path.is_symlink():
                    continue
                shards = (model_path.resolve(),)
                required_paths.append(model_path)
            else:
                continue

            try:
                manifest = []
                for required in required_paths:
                    resolved = required.resolve()
                    if (
                        required.is_symlink()
                        or resolved.parent != root
                        or not required.is_file()
                    ):
                        raise OSError
                    stat = required.stat()
                    manifest.append(
                        f"{required.relative_to(path).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}"
                    )
            except OSError:
                continue
            digest = hashlib.sha256("\n".join(sorted(manifest)).encode("utf-8"))
            try:
                now = time.time()
                stable = all(
                    now - _mtime(required) >= self.min_age_seconds
                    for required in required_paths
                )
            except OSError:
                continue
            if stable_only and not stable:
                continue
            refs.append(
                CheckpointRef(
                    int(match.group(1)),
                    path,
                    shards,
                    digest.hexdigest(),
                    stable,
                    "stable" if stable else "recent",
                )
            )
        return sorted(refs, key=lambda ref: ref.step)
