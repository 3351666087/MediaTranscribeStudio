"""Stream safetensors checkpoints into smaller Windows-friendly shards.

The safetensors runtime memory-maps one complete shard at a time.  Large
single shards can therefore fail with Windows error 1455 even when the model
itself would fit across GPU, RAM, and the paging file.  This tool changes only
the physical checkpoint layout: tensor bytes, names, dtypes, shapes, and model
configuration remain unchanged.

The implementation intentionally does not import torch, transformers, or
safetensors.  It parses the documented safetensors container header and copies
tensor byte ranges through a bounded buffer, so preparing a low-commit layout
does not first require mapping the oversized source shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping, Sequence

_HEADER_LENGTH_BYTES = 8
_COPY_BUFFER_BYTES = 8 * 1024 * 1024
_INDEX_NAME = "model.safetensors.index.json"
_MANIFEST_NAME = ".mts-model-manifest.json"
_RESHARD_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class TensorSource:
    """One immutable tensor byte range in a source safetensors file."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    source_path: Path
    source_offset: int
    size_bytes: int

    def header_entry(self, start: int) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_offsets": [start, start + self.size_bytes],
        }


@dataclass(frozen=True)
class ReshardResult:
    source: Path
    output: Path
    tensor_count: int
    total_tensor_bytes: int
    shard_count: int
    largest_source_shard_bytes: int
    largest_output_shard_bytes: int
    shard_sha256: Mapping[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "output": str(self.output),
            "tensorCount": self.tensor_count,
            "totalTensorBytes": self.total_tensor_bytes,
            "shardCount": self.shard_count,
            "largestSourceShardBytes": self.largest_source_shard_bytes,
            "largestOutputShardBytes": self.largest_output_shard_bytes,
            "shardSha256": dict(self.shard_sha256),
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("safetensors file ended before the declared byte range")
    return data


def _read_safetensors_header(
    path: Path,
) -> tuple[dict[str, object], int]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = _read_exact(handle, _HEADER_LENGTH_BYTES)
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length < 2 or header_length > file_size - _HEADER_LENGTH_BYTES:
            raise ValueError(f"invalid safetensors header length in {path.name}")
        raw_header = _read_exact(handle, header_length)
    try:
        decoded = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header in {path.name}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"safetensors header must be an object in {path.name}")
    return decoded, _HEADER_LENGTH_BYTES + header_length


def _checkpoint_files(source: Path) -> tuple[Path, ...]:
    index_path = source / _INDEX_NAME
    if index_path.is_file():
        raw_index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = raw_index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json has no weight_map")
        names = tuple(dict.fromkeys(str(value) for value in weight_map.values()))
        files = tuple(source / name for name in names)
    else:
        files = tuple(sorted(source.glob("*.safetensors")))
    if not files:
        raise ValueError("source directory contains no safetensors checkpoint")
    for path in files:
        if path.parent != source or not path.is_file():
            raise ValueError("checkpoint index references a missing local shard")
    return files


def inspect_checkpoint(
    source: str | Path,
) -> tuple[tuple[TensorSource, ...], dict[str, str], tuple[Path, ...]]:
    """Return tensors, shared metadata, and source shards without mmap."""

    source_path = Path(source).expanduser().resolve(strict=True)
    if not source_path.is_dir():
        raise ValueError("source must be a model directory")
    files = _checkpoint_files(source_path)
    tensors: list[TensorSource] = []
    metadata: dict[str, str] | None = None
    seen_names: set[str] = set()

    for shard in files:
        header, data_start = _read_safetensors_header(shard)
        raw_metadata = header.pop("__metadata__", None)
        if raw_metadata is not None:
            if not isinstance(raw_metadata, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in raw_metadata.items()
            ):
                raise ValueError(f"invalid __metadata__ in {shard.name}")
            current_metadata = dict(raw_metadata)
            if metadata is None:
                metadata = current_metadata
            elif current_metadata != metadata:
                raise ValueError("source shards have inconsistent metadata")

        ordered: list[tuple[int, TensorSource]] = []
        for name, raw_entry in header.items():
            if not isinstance(name, str) or name in seen_names:
                raise ValueError("tensor names must be unique strings")
            if not isinstance(raw_entry, dict):
                raise ValueError(f"tensor {name!r} has an invalid header entry")
            dtype = raw_entry.get("dtype")
            shape = raw_entry.get("shape")
            offsets = raw_entry.get("data_offsets")
            if (
                not isinstance(dtype, str)
                or not isinstance(shape, list)
                or not all(isinstance(item, int) and item >= 0 for item in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(item, int) for item in offsets)
            ):
                raise ValueError(f"tensor {name!r} has invalid metadata")
            start, end = offsets
            if start < 0 or end < start:
                raise ValueError(f"tensor {name!r} has invalid data offsets")
            absolute_start = data_start + start
            absolute_end = data_start + end
            if absolute_end > shard.stat().st_size:
                raise ValueError(f"tensor {name!r} exceeds {shard.name}")
            tensor = TensorSource(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                source_path=shard,
                source_offset=absolute_start,
                size_bytes=end - start,
            )
            ordered.append((start, tensor))
            seen_names.add(name)

        ordered.sort(key=lambda item: (item[0], item[1].name))
        previous_end = 0
        for relative_start, tensor in ordered:
            if relative_start < previous_end:
                raise ValueError(f"overlapping tensor data in {shard.name}")
            previous_end = relative_start + tensor.size_bytes
            tensors.append(tensor)

    return tuple(tensors), metadata or {}, files


def _group_tensors(
    tensors: Sequence[TensorSource],
    max_shard_bytes: int,
) -> tuple[tuple[TensorSource, ...], ...]:
    if max_shard_bytes < 1:
        raise ValueError("max_shard_bytes must be positive")
    groups: list[tuple[TensorSource, ...]] = []
    current: list[TensorSource] = []
    current_bytes = 0
    for tensor in tensors:
        if current and current_bytes + tensor.size_bytes > max_shard_bytes:
            groups.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(tensor)
        current_bytes += tensor.size_bytes
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _padded_header(
    tensors: Sequence[TensorSource],
    metadata: Mapping[str, str],
) -> bytes:
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = dict(metadata)
    offset = 0
    for tensor in tensors:
        header[tensor.name] = tensor.header_entry(offset)
        offset += tensor.size_bytes
    raw = _canonical_json(header)
    padding = (-len(raw)) % 8
    return raw + (b" " * padding)


def _copy_tensor(
    source_handle: BinaryIO,
    target_handle: BinaryIO,
    tensor: TensorSource,
    digest: "hashlib._Hash",
) -> None:
    source_handle.seek(tensor.source_offset)
    remaining = tensor.size_bytes
    while remaining:
        chunk = _read_exact(
            source_handle,
            min(_COPY_BUFFER_BYTES, remaining),
        )
        target_handle.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def _write_shard(
    destination: Path,
    tensors: Sequence[TensorSource],
    metadata: Mapping[str, str],
) -> str:
    header = _padded_header(tensors, metadata)
    digest = hashlib.sha256()
    handles: dict[Path, BinaryIO] = {}
    try:
        with destination.open("wb") as target:
            prefix = struct.pack("<Q", len(header))
            target.write(prefix)
            target.write(header)
            digest.update(prefix)
            digest.update(header)
            for tensor in tensors:
                handle = handles.get(tensor.source_path)
                if handle is None:
                    handle = tensor.source_path.open("rb")
                    handles[tensor.source_path] = handle
                _copy_tensor(handle, target, tensor, digest)
            target.flush()
            os.fsync(target.fileno())
    finally:
        for handle in handles.values():
            handle.close()
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_COPY_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_support_files(source: Path, destination: Path) -> None:
    for entry in source.iterdir():
        if not entry.is_file():
            continue
        if (
            entry.suffix == ".safetensors"
            or entry.name in {_INDEX_NAME, _MANIFEST_NAME}
        ):
            continue
        shutil.copy2(entry, destination / entry.name)


def _source_manifest(source: Path) -> dict[str, object]:
    path = source / _MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source model manifest is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("source model manifest must be an object")
    return value


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_derived_manifest(
    *,
    source: Path,
    destination: Path,
    source_shards: Sequence[Path],
    tensors: Sequence[TensorSource],
    max_shard_bytes: int,
) -> None:
    source_manifest = _source_manifest(source)
    source_identity = {
        key: source_manifest[key]
        for key in (
            "modelKey",
            "provider",
            "repoId",
            "revision",
            "lockSha256",
        )
        if key in source_manifest
    }
    source_records = [_file_record(path, source) for path in source_shards]
    output_files = sorted(
        (
            path
            for path in destination.iterdir()
            if path.is_file() and path.name != _MANIFEST_NAME
        ),
        key=lambda path: path.name,
    )
    output_records = [_file_record(path, destination) for path in output_files]
    payload = {
        "schemaVersion": "1.1.0",
        "kind": "derived-safetensors-reshard",
        **source_identity,
        "totalBytes": sum(
            int(record["size"]) for record in output_records
        ),
        "files": output_records,
        "reshard": {
            "schemaVersion": _RESHARD_SCHEMA_VERSION,
            "tool": "tools/reshard_safetensors.py",
            "maxShardBytes": max_shard_bytes,
            "tensorCount": len(tensors),
            "totalTensorBytes": sum(
                tensor.size_bytes for tensor in tensors
            ),
            "exactTensorBytesPreserved": True,
            "modelQualityChanged": False,
            "source": {
                **source_identity,
                "manifestSchemaVersion": source_manifest.get(
                    "schemaVersion"
                ),
                "totalBytes": source_manifest.get("totalBytes"),
                "checkpointFiles": source_records,
            },
        },
    }
    (destination / _MANIFEST_NAME).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def reshard_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    max_shard_bytes: int,
    verify: bool = True,
) -> ReshardResult:
    """Create a byte-identical tensor layout with bounded shard sizes."""

    source_path = Path(source).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    if output_path == source_path or source_path in output_path.parents:
        raise ValueError("output must not be the source or a child of the source")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tensors, metadata, source_shards = inspect_checkpoint(source_path)
    groups = _group_tensors(tensors, max_shard_bytes)
    if not groups:
        raise ValueError("checkpoint contains no tensors")

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.reshard-",
            dir=output_path.parent,
        )
    )
    shard_hashes: dict[str, str] = {}
    weight_map: dict[str, str] = {}
    try:
        _copy_support_files(source_path, temporary)
        shard_count = len(groups)
        width = max(5, len(str(shard_count)))
        for index, group in enumerate(groups, start=1):
            shard_name = (
                f"model-{index:0{width}d}-of-"
                f"{shard_count:0{width}d}.safetensors"
            )
            shard_path = temporary / shard_name
            expected_hash = _write_shard(shard_path, group, metadata)
            if verify:
                actual_hash = _sha256(shard_path)
                if actual_hash != expected_hash:
                    raise OSError(f"verification failed for {shard_name}")
            shard_hashes[shard_name] = expected_hash
            for tensor in group:
                weight_map[tensor.name] = shard_name

        index_payload = {
            "metadata": {
                "total_size": sum(item.size_bytes for item in tensors),
            },
            "weight_map": weight_map,
        }
        (temporary / _INDEX_NAME).write_text(
            json.dumps(
                index_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_derived_manifest(
            source=source_path,
            destination=temporary,
            source_shards=source_shards,
            tensors=tensors,
            max_shard_bytes=max_shard_bytes,
        )
        os.replace(temporary, output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    largest_output = max(
        (output_path / name).stat().st_size for name in shard_hashes
    )
    return ReshardResult(
        source=source_path,
        output=output_path,
        tensor_count=len(tensors),
        total_tensor_bytes=sum(item.size_bytes for item in tensors),
        shard_count=len(groups),
        largest_source_shard_bytes=max(path.stat().st_size for path in source_shards),
        largest_output_shard_bytes=largest_output,
        shard_sha256=shard_hashes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a safetensors checkpoint into smaller shards without "
            "changing tensor values."
        )
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-shard-mib",
        type=int,
        default=768,
        help="maximum target tensor payload per shard (default: 768 MiB)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the output reread hash verification",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = reshard_checkpoint(
        args.source,
        args.output,
        max_shard_bytes=args.max_shard_mib * 1024 * 1024,
        verify=not args.no_verify,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
