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
        output_root = output_dir.resolve()
        for path in output_dir.glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            root = path.resolve()
            if not match or not path.is_dir() or root.parent != output_root:
                continue

            index_path = path / "model.safetensors.index.json"
            adapter_path = path / "adapter_model.safetensors"
            trainer_state_path = path / "trainer_state.json"
            model_path = path / "model.safetensors"

            if index_path.exists():
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
                try:
                    digest = hashlib.sha256(index_path.read_bytes())
                except OSError:
                    continue
            elif adapter_path.exists():
                if not adapter_path.is_file() or not trainer_state_path.is_file():
                    continue
                shards = (adapter_path.resolve(),)
                try:
                    digest = hashlib.sha256(trainer_state_path.read_bytes())
                except OSError:
                    continue
            elif model_path.is_file():
                shards = (model_path.resolve(),)
                digest = hashlib.sha256()
            else:
                continue

            try:
                for shard in shards:
                    stat = shard.stat()
                    digest.update(f"{shard.name}:{stat.st_size}".encode())
            except OSError:
                continue
            refs.append(CheckpointRef(int(match.group(1)), path, shards, digest.hexdigest()))
        return sorted(refs, key=lambda ref: ref.step)
