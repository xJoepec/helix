import json
import os
import time

from helix.integrations import checkpoints as checkpoints_module
from helix.integrations.checkpoints import CheckpointWatcher


def _trainer_state(checkpoint, *, global_step=None):
    state = {} if global_step is None else {"global_step": global_step}
    (checkpoint / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")


def test_discovers_only_complete_indexed_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-500"
    checkpoint.mkdir()
    first_shard = checkpoint / "model-00001-of-00002.safetensors"
    second_shard = checkpoint / "model-00002-of-00002.safetensors"
    first_shard.write_bytes(b"first weights")
    second_shard.write_bytes(b"second weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": first_shard.name,
                    "layer.1": second_shard.name,
                }
            }
        ),
        encoding="utf-8",
    )
    _trainer_state(checkpoint, global_step=500)
    refs = CheckpointWatcher().discover(tmp_path)
    assert [ref.step for ref in refs] == [500]
    assert refs[0].shard_paths == (first_shard, second_shard)


def test_discovers_unsloth_qlora_adapter_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-42"
    checkpoint.mkdir()
    adapter = checkpoint / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter weights")
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

    refs = CheckpointWatcher().discover(tmp_path)

    assert [ref.step for ref in refs] == [42]
    assert refs[0].shard_paths == (adapter,)


def test_discovers_single_file_safetensors_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-9"
    checkpoint.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"model weights")
    _trainer_state(checkpoint, global_step=9)

    refs = CheckpointWatcher().discover(tmp_path)

    assert [ref.step for ref in refs] == [9]
    assert refs[0].shard_paths == (model,)


def test_rejects_missing_shard(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "missing.safetensors"}}), encoding="utf-8"
    )
    _trainer_state(checkpoint, global_step=7)
    assert CheckpointWatcher().discover(tmp_path) == []


def test_rejects_partial_adapter_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-8"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter weights")

    assert CheckpointWatcher().discover(tmp_path) == []


def test_rejects_indexed_checkpoint_with_traversal_shard(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-11"
    checkpoint.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": f"../{outside.name}"}}), encoding="utf-8"
    )
    _trainer_state(checkpoint, global_step=11)

    assert CheckpointWatcher().discover(tmp_path) == []


def test_does_not_fallback_from_partial_index_to_single_file(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "missing.safetensors"}}), encoding="utf-8"
    )
    _trainer_state(checkpoint, global_step=12)

    assert CheckpointWatcher().discover(tmp_path) == []


def test_rejects_checkpoint_when_trainer_step_does_not_match_directory(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-13"
    checkpoint.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"weights")
    _trainer_state(checkpoint, global_step=12)

    assert CheckpointWatcher().discover(tmp_path) == []


def test_marks_recently_modified_complete_checkpoint_pending(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-14"
    checkpoint.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"weights")
    _trainer_state(checkpoint, global_step=14)

    refs = CheckpointWatcher(min_age_seconds=60).discover(tmp_path)

    assert len(refs) == 1
    assert refs[0].stable is False
    assert refs[0].stability_reason == "recent"


def test_marks_complete_checkpoint_with_old_mtimes_stable(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-15"
    checkpoint.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"weights")
    _trainer_state(checkpoint, global_step=15)
    old = time.time() - 120
    for path in (model, checkpoint / "trainer_state.json"):
        os.utime(path, (old, old))

    refs = CheckpointWatcher(min_age_seconds=60).discover(tmp_path, stable_only=True)

    assert [ref.step for ref in refs] == [15]
    assert refs[0].stable is True
    assert refs[0].stability_reason == "stable"


def test_skips_checkpoint_when_age_stat_races_with_save(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint-16"
    checkpoint.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"weights")
    _trainer_state(checkpoint, global_step=16)
    trainer_state = checkpoint / "trainer_state.json"
    original_mtime = checkpoints_module._mtime
    original_time = checkpoints_module.time.time
    age_phase = False

    def mark_age_phase():
        nonlocal age_phase
        age_phase = True
        return original_time()

    def race_on_age_stat(path):
        if checkpoints_module.Path(path) == trainer_state and age_phase:
            raise OSError("checkpoint changed during discovery")
        return original_mtime(path)

    monkeypatch.setattr(checkpoints_module.time, "time", mark_age_phase)
    monkeypatch.setattr(checkpoints_module, "_mtime", race_on_age_stat)

    assert CheckpointWatcher().discover(tmp_path) == []
