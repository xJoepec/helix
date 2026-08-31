import json

from helix.integrations.checkpoints import CheckpointWatcher


def test_discovers_only_complete_indexed_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-500"
    checkpoint.mkdir()
    shard = checkpoint / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": shard.name}}), encoding="utf-8"
    )
    refs = CheckpointWatcher().discover(tmp_path)
    assert [ref.step for ref in refs] == [500]
    assert refs[0].shard_paths == (shard,)


def test_rejects_missing_shard(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "missing.safetensors"}}), encoding="utf-8"
    )
    assert CheckpointWatcher().discover(tmp_path) == []
