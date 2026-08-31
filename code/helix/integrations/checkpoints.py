"""Discover complete Hugging Face checkpoints without mutating them."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointRef:
    step: int
    path: Path
    shard_paths: tuple[Path, ...]
    fingerprint: str


class CheckpointWatcher:
    def discover(self, output_dir: Path) -> list[CheckpointRef]:
        refs = []
        for path in output_dir.glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            index_path = path / "model.safetensors.index.json"
            if not match or not index_path.is_file():
                continue
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                names = sorted(set(data["weight_map"].values()))
            except (KeyError, TypeError, ValueError, OSError):
                continue
            shards = tuple((path / name).resolve() for name in names)
            root = path.resolve()
            if not shards or any(
                root not in shard.parents or not shard.is_file() for shard in shards
            ):
                continue
            digest = hashlib.sha256(index_path.read_bytes())
            for shard in shards:
                stat = shard.stat()
                digest.update(f"{shard.name}:{stat.st_size}".encode())
            refs.append(CheckpointRef(int(match.group(1)), path, shards, digest.hexdigest()))
        return sorted(refs, key=lambda ref: ref.step)
