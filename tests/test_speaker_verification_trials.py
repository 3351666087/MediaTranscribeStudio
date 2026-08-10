import json
import tempfile
import unittest
import wave
from pathlib import Path

from backend.persistence import canonical_json_sha256
from tools.benchmark_redimnet2_verification import (
    ReDimNet2BenchmarkError,
    load_trial_manifest,
    verification_metrics,
)
from tools.build_speaker_verification_trials import (
    SpeakerTrialError,
    build_manifest,
    solo_speaker_spans,
)


class SpeakerVerificationTrialTests(unittest.TestCase):
    def test_solo_spans_exclude_every_overlap(self) -> None:
        turns = [
            (0, 10_000, "speaker-a"),
            (4_000, 6_000, "speaker-b"),
            (10_000, 20_000, "speaker-b"),
        ]

        self.assertEqual(
            solo_speaker_spans(turns),
            [
                (0, 4_000, "speaker-a"),
                (6_000, 10_000, "speaker-a"),
                (10_000, 20_000, "speaker-b"),
            ],
        )

    def test_manifest_is_balanced_hashed_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "source.wav"
            annotations = root / "source.json"
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 16_000 * 50)
            annotations.write_text(
                json.dumps(
                    {
                        "dataset": "example/diarization",
                        "revision": "a" * 40,
                        "config": "mono",
                        "split": "test",
                        "rowIndex": 0,
                        "timestamps_start": [0.0, 12.0, 25.0, 37.0],
                        "timestamps_end": [10.0, 22.0, 35.0, 47.0],
                        "speakers": ["a", "b", "a", "b"],
                    }
                ),
                encoding="utf-8",
            )

            options = {
                "audio_path": audio,
                "annotation_path": annotations,
                "source_recording_id": "recording-1",
                "evaluation_split": "development",
                "clip_duration_ms": 2_000,
                "maximum_clips_per_speaker": 4,
                "minimum_spacing_ms": 4_000,
                "minimum_pair_separation_ms": 10_000,
                "maximum_pairs_per_class": 8,
                "random_seed": 7,
            }
            first = build_manifest(**options)
            second = build_manifest(**options)

            self.assertEqual(first, second)
            self.assertEqual(first["counts"]["speakers"], 2)
            self.assertEqual(first["counts"]["clips"], 8)
            self.assertEqual(first["counts"]["sameSpeakerTrials"], 8)
            self.assertEqual(first["counts"]["differentSpeakerTrials"], 8)
            self.assertEqual(
                {trial["sameSpeaker"] for trial in first["trials"]},
                {True, False},
            )
            expected = dict(first)
            digest = expected.pop("canonicalSha256")
            self.assertEqual(digest, canonical_json_sha256(expected))

    def test_manifest_rejects_annotations_beyond_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "source.wav"
            annotations = root / "source.json"
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 16_000)
            annotations.write_text(
                json.dumps(
                    {
                        "timestamps_start": [0.0, 2.0],
                        "timestamps_end": [1.0, 3.0],
                        "speakers": ["a", "b"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SpeakerTrialError, "beyond"):
                build_manifest(
                    audio_path=audio,
                    annotation_path=annotations,
                    source_recording_id="recording-1",
                    evaluation_split="development",
                )

    def test_verification_metrics_are_perfect_for_separated_scores(self) -> None:
        metrics = verification_metrics(
            [(0.95, True), (0.8, True), (0.2, False), (-0.1, False)]
        )

        self.assertEqual(metrics["eer"], 0.0)
        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["balancedAccuracyAtEer"], 1.0)
        self.assertGreater(metrics["meanCosineSeparation"], 0.0)

    def test_trial_loader_rejects_tampered_canonical_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trials.json"
            path.write_text(
                json.dumps(
                    {
                        "canonicalSha256": "0" * 64,
                        "source": {},
                        "clips": [],
                        "trials": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ReDimNet2BenchmarkError, "canonical digest"
            ):
                load_trial_manifest(path)


if __name__ == "__main__":
    unittest.main()
