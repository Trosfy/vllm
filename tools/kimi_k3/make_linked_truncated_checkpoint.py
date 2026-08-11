#!/usr/bin/env python3
"""Create a metadata-only first-N-layer Kimi K3 checkpoint.

Selected safetensor shards are symlinked to the original Hugging Face cache;
no weight payload is copied. The filtered index prevents the loader from
visiting tensors outside the retained layers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

LAYER_RE = re.compile(r"language_model\.model\.layers\.(\d+)\.")


def _wanted(name: str, layers: int) -> bool:
    match = LAYER_RE.search(name)
    return match is None or int(match.group(1)) < layers


def _truncate_config(config: dict, layers: int) -> dict:
    text_config = config.get("text_config", config)
    source_layers = int(text_config["num_hidden_layers"])
    if not 1 <= layers <= source_layers:
        raise ValueError(f"--layers must be in [1, {source_layers}]")
    text_config["num_hidden_layers"] = layers
    linear = text_config["linear_attn_config"]
    for key in ("kda_layers", "full_attn_layers"):
        linear[key] = [value for value in linear[key] if int(value) <= layers]
    return config


def build(source: Path, destination: Path, layers: int) -> dict:
    source = source.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    index_path = source / "model.safetensors.index.json"
    config_path = source / "config.json"
    index = json.loads(index_path.read_text())
    weight_map = {
        name: filename
        for name, filename in index["weight_map"].items()
        if _wanted(name, layers)
    }
    shards = sorted(set(weight_map.values()))
    if not weight_map or not shards:
        raise RuntimeError("filtered checkpoint contains no tensors")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-{os.getpid()}-",
            dir=destination.parent,
        )
    )
    try:
        for item in source.iterdir():
            if item.name in {"config.json", "model.safetensors.index.json"}:
                continue
            if item.name.endswith(".safetensors"):
                continue
            (staging / item.name).symlink_to(item.resolve())
        for filename in shards:
            shard = (source / filename).resolve()
            if not shard.is_file():
                raise FileNotFoundError(shard)
            (staging / filename).symlink_to(shard)

        filtered_index = {
            "metadata": {
                **index.get("metadata", {}),
                "linked_truncated_layers": layers,
                "linked_truncated_tensor_count": len(weight_map),
            },
            "weight_map": dict(sorted(weight_map.items())),
        }
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(filtered_index, sort_keys=True) + "\n"
        )
        config = _truncate_config(json.loads(config_path.read_text()), layers)
        (staging / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "destination": str(destination),
        "layers": layers,
        "tensor_count": len(weight_map),
        "linked_shard_count": len(shards),
    }


def build_dspark_overlay(
    source: Path,
    destination: Path,
    target_layer_ids: tuple[int, ...],
    target_num_hidden_layers: int,
) -> dict:
    """Create a linked DSpark checkpoint with remapped target taps."""
    source = source.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    num_target_layers = int(config["num_target_layers"])
    if len(target_layer_ids) != num_target_layers:
        raise ValueError(
            "--draft-target-layer-ids must contain "
            f"{num_target_layers} entries"
        )
    if any(
        layer_id < 0 or layer_id >= target_num_hidden_layers
        for layer_id in target_layer_ids
    ):
        raise ValueError(
            "draft target layer ids must address the truncated target"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-{os.getpid()}-",
            dir=destination.parent,
        )
    )
    try:
        for item in source.iterdir():
            if item.name == "config.json":
                continue
            (staging / item.name).symlink_to(item.resolve())
        config["target_layer_ids"] = list(target_layer_ids)
        config["target_num_hidden_layers"] = target_num_hidden_layers
        (staging / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "destination": str(destination),
        "target_layer_ids": list(target_layer_ids),
        "target_num_hidden_layers": target_num_hidden_layers,
    }


def _parse_layer_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one layer id is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--draft-source", type=Path)
    parser.add_argument("--draft-destination", type=Path)
    parser.add_argument("--draft-target-layer-ids", type=_parse_layer_ids)
    args = parser.parse_args()
    draft_options = (
        args.draft_source,
        args.draft_destination,
        args.draft_target_layer_ids,
    )
    if any(option is not None for option in draft_options) and not all(
        option is not None for option in draft_options
    ):
        parser.error(
            "--draft-source, --draft-destination, and "
            "--draft-target-layer-ids must be used together"
        )
    result = build(args.source, args.destination, args.layers)
    if args.draft_source is not None:
        assert args.draft_destination is not None
        assert args.draft_target_layer_ids is not None
        result["draft"] = build_dspark_overlay(
            args.draft_source,
            args.draft_destination,
            args.draft_target_layer_ids,
            args.layers,
        )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
