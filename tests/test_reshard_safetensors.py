from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.reshard_safetensors import (
    inspect_checkpoint,
    reshard_checkpoint,
)


def _write_fake_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for _name, _dtype, _shape, data in tensors:
            handle.write(data)


class ReshardSafetensorsTests(unittest.TestCase):
    def test_reshard_preserves_tensor_bytes_and_support_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            tensors = [
                ("alpha", "U8", [6], b"abcdef"),
                ("beta", "U8", [7], b"ghijklm"),
                ("gamma", "U8", [4], b"nopq"),
            ]
            _write_fake_safetensors(source / "model.safetensors", tensors)
            (source / "config.json").write_text(
                '{"model_type":"fixture"}\n',
                encoding="utf-8",
            )
            (source / ".mts-model-manifest.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": "1.0.0",
                        "modelKey": "fixture",
                        "provider": "local-test",
                        "repoId": "test/fixture",
                        "revision": "abc123",
                        "lockSha256": "1" * 64,
                        "totalBytes": 17,
                        "files": [
                            {
                                "path": "model.safetensors",
                                "size": (
                                    source / "model.safetensors"
                                ).stat().st_size,
                                "sha256": "stale-source-layout",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = reshard_checkpoint(
                source,
                output,
                max_shard_bytes=13,
            )

            self.assertEqual(result.tensor_count, 3)
            self.assertEqual(result.shard_count, 2)
            self.assertTrue((output / "config.json").is_file())
            output_tensors, metadata, shards = inspect_checkpoint(output)
            self.assertEqual(metadata, {"format": "pt"})
            self.assertEqual(
                [(item.name, item.dtype, item.shape) for item in output_tensors],
                [
                    ("alpha", "U8", (6,)),
                    ("beta", "U8", (7,)),
                    ("gamma", "U8", (4,)),
                ],
            )
            expected = {name: data for name, _dtype, _shape, data in tensors}
            for tensor in output_tensors:
                with tensor.source_path.open("rb") as handle:
                    handle.seek(tensor.source_offset)
                    actual = handle.read(tensor.size_bytes)
                self.assertEqual(actual, expected[tensor.name])
            index = json.loads(
                (output / "model.safetensors.index.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(index["metadata"]["total_size"], 17)
            self.assertEqual(set(index["weight_map"]), set(expected))
            for shard in shards:
                self.assertEqual(
                    result.shard_sha256[shard.name],
                    hashlib.sha256(shard.read_bytes()).hexdigest(),
                )
            manifest = json.loads(
                (output / ".mts-model-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["schemaVersion"], "1.1.0")
            self.assertEqual(manifest["modelKey"], "fixture")
            self.assertEqual(
                manifest["reshard"]["source"]["revision"],
                "abc123",
            )
            self.assertTrue(
                manifest["reshard"]["exactTensorBytesPreserved"]
            )
            self.assertFalse(manifest["reshard"]["modelQualityChanged"])
            output_checkpoint_files = {
                item["path"]
                for item in manifest["files"]
                if item["path"].endswith(".safetensors")
            }
            self.assertNotIn("model.safetensors", output_checkpoint_files)
            self.assertEqual(
                output_checkpoint_files,
                {path.name for path in shards},
            )

    def test_refuses_existing_or_nested_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            _write_fake_safetensors(
                source / "model.safetensors",
                [("alpha", "U8", [1], b"a")],
            )
            with self.assertRaises(ValueError):
                reshard_checkpoint(
                    source,
                    source / "nested",
                    max_shard_bytes=1,
                )
            existing = Path(temporary) / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                reshard_checkpoint(
                    source,
                    existing,
                    max_shard_bytes=1,
                )


if __name__ == "__main__":
    unittest.main()
