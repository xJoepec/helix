import json

from helix.integrations.checkpoints import CheckpointWatcher


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

    refs = CheckpointWatcher().discover(tmp_path)

    assert [ref.step for ref in refs] == [9]
    assert refs[0].shard_paths == (model,)


def test_rejects_missing_shard(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "missing.safetensors"}}), encoding="utf-8"
    )
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

    assert CheckpointWatcher().discover(tmp_path) == []


def test_does_not_fallback_from_partial_index_to_single_file(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "missing.safetensors"}}), encoding="utf-8"
    )

    assert CheckpointWatcher().discover(tmp_path) == []
