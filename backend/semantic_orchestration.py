"""Persistent bounded loop for semantic candidate arbitration and composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import JobCancelled, WorkerError
from .persistence import (
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
)
from .semantic_candidate_generation import (
    SemanticCandidateGenerationRegistry,
    validate_semantic_candidate_generation,
)
from .semantic_candidate_lattice import (
    SemanticCandidateLatticeError,
    build_semantic_candidate_lattice_from_document,
    validate_semantic_candidate_lattice,
)
from .semantic_composition import (
    SemanticCompositionError,
    SemanticJobArbitrationRunner,
    build_semantic_composition,
    validate_semantic_composition,
    validate_semantic_job_arbitration,
)


@dataclass(frozen=True)
class SemanticCompositionRunResult:
    """Validated artifacts produced by one complete semantic loop."""

    initial_lattice: dict[str, Any]
    final_lattice: dict[str, Any]
    arbitration: dict[str, Any]
    composition: dict[str, Any]
    initial_lattice_path: Path
    final_lattice_path: Path
    arbitration_path: Path
    composition_path: Path
    generation_paths: tuple[Path, ...]
    artifact_paths: tuple[Path, ...]
    round_count: int
    resumed_artifact_count: int


class SemanticCompositionOrchestrator:
    """Run and persist a fail-closed arbitration/generation loop."""

    def __init__(
        self,
        *,
        arbitrator: SemanticJobArbitrationRunner,
        generators: SemanticCandidateGenerationRegistry,
        max_rounds: int = 3,
    ) -> None:
        if (
            isinstance(max_rounds, bool)
            or not isinstance(max_rounds, int)
            or max_rounds < 1
            or max_rounds > 8
        ):
            raise ValueError("semantic composition max_rounds must be between 1 and 8")
        self.arbitrator = arbitrator
        self.generators = generators
        self.max_rounds = max_rounds

    def release_resources(self) -> None:
        first_error: Exception | None = None
        for owner in (self.arbitrator, self.generators):
            release = getattr(owner, "release_resources", None)
            if not callable(release):
                continue
            try:
                release()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _release_generation_resources_between_rounds(self) -> None:
        """Unload acoustic challengers before the next local-LLM pass."""

        release = getattr(self.generators, "release_resources", None)
        if not callable(release):
            return
        try:
            release()
        except Exception as exc:
            raise WorkerError(
                "SEMANTIC_CANDIDATE_RESOURCE_RELEASE_FAILED",
                "semantic candidate generators could not release resources "
                "before the next arbitration round",
                details={"exceptionType": type(exc).__name__},
                retryable=True,
            ) from exc

    def _release_arbitrator_resources_between_rounds(self) -> None:
        """Unload the local LLM before acoustic challenger generation."""

        try:
            self.arbitrator.release_resources()
        except Exception as exc:
            raise WorkerError(
                "SEMANTIC_ARBITRATOR_RESOURCE_RELEASE_FAILED",
                "semantic arbitrator could not release resources before "
                "candidate generation",
                details={"exceptionType": type(exc).__name__},
                retryable=True,
            ) from exc

    @staticmethod
    def _persist_or_resume(
        path: Path,
        built: Mapping[str, Any],
        *,
        validator: Any,
    ) -> tuple[dict[str, Any], bool]:
        if path.exists():
            return dict(validator(read_json_strict(path))), True
        value = dict(validator(built))
        atomic_write_json_no_replace(path, value)
        return value, False

    def run(
        self,
        document: Mapping[str, Any],
        *,
        artifact_root: Path,
    ) -> SemanticCompositionRunResult:
        """Complete one transcript-bound semantic loop or fail at a hard bound."""

        transcript_sha = canonical_json_sha256(document)
        root = artifact_root.resolve()
        initial_path = root / "semantic-candidate-lattice.initial.v1.json"
        initial_built = build_semantic_candidate_lattice_from_document(document)
        initial, resumed = self._persist_or_resume(
            initial_path,
            initial_built,
            validator=lambda value: validate_semantic_candidate_lattice(
                value,
                expected_source_media_sha256=document["source"]["sha256"],
                expected_transcript_sha256=transcript_sha,
            ),
        )
        resumed_count = int(resumed)
        artifact_paths: list[Path] = [initial_path]
        generation_paths: list[Path] = []
        current_lattice = initial
        current_lattice_path = initial_path
        carried_lattice: dict[str, Any] | None = None
        carried_arbitration: dict[str, Any] | None = None
        exhausted_request_group_ids: set[str] = set()

        # ``max_rounds`` bounds candidate-generation passes.  A generation pass
        # always needs a subsequent arbitration pass to consume the new lattice;
        # otherwise a valid final generation on the bound's last pass would be
        # discarded before composition.  The extra pass is arbitration-only and
        # cannot request another generation.
        for round_number in range(1, self.max_rounds + 2):
            round_root = root / f"round-{round_number:02d}"
            arbitration_path = round_root / "semantic-job-arbitration.v1.json"
            if arbitration_path.exists():
                arbitration = validate_semantic_job_arbitration(
                    read_json_strict(arbitration_path),
                    expected_job_id=str(document.get("jobId") or ""),
                    expected_lattice=current_lattice,
                )
                resumed_count += 1
            else:
                try:
                    arbitration_exhausted_group_ids = set(
                        exhausted_request_group_ids
                    )
                    if round_number > self.max_rounds:
                        # The extra pass consumes the final generated lattice;
                        # it is deliberately arbitration-only.  Do not let a
                        # model reopen a new challenger after the bounded
                        # generation budget has been spent.  Passing every
                        # available group as exhausted removes ``-1`` from the
                        # positional response enum while keeping candidate
                        # selection and semantic evidence fully model-driven.
                        arbitration_exhausted_group_ids.update(
                            str(group["groupId"])
                            for domain in current_lattice["domains"]
                            for group in domain["groups"]
                            if group.get("status") == "available"
                        )
                    arbitration = self.arbitrator.run(
                        document,
                        candidate_lattice=current_lattice,
                        carried_lattice=carried_lattice,
                        carried_arbitration=carried_arbitration,
                        exhausted_request_group_ids=(
                            frozenset(arbitration_exhausted_group_ids)
                        ),
                    )
                except WorkerError as exc:
                    if exc.code == "SEMANTIC_JOB_PROVIDER_FAILED":
                        failure_root = round_root / "arbitration-failures"
                        failure_index = 1
                        while True:
                            failure_path = (
                                failure_root
                                / (
                                    "semantic-arbitration-failure-"
                                    f"{failure_index:04d}.v1.json"
                                )
                            )
                            failure = {
                                "schemaVersion": "1.0.0",
                                "artifactType": (
                                    "semantic-arbitration-failure"
                                ),
                                "jobId": str(document.get("jobId") or ""),
                                "round": round_number,
                                "failureIndex": failure_index,
                                "model": str(
                                    getattr(
                                        self.arbitrator,
                                        "model",
                                        "unknown",
                                    )
                                ),
                                "input": {
                                    "transcriptSha256": transcript_sha,
                                    "sourceMediaSha256": current_lattice[
                                        "binding"
                                    ]["sourceMediaSha256"],
                                    "latticeSha256": current_lattice[
                                        "latticeSha256"
                                    ],
                                },
                                "error": exc.as_payload(),
                                "responseContentPersisted": False,
                            }
                            try:
                                atomic_write_json_no_replace(
                                    failure_path,
                                    failure,
                                )
                            except FileExistsError:
                                failure_index += 1
                                continue
                            failure_sha256 = canonical_json_sha256(failure)
                            exc.details["diagnosticArtifactPath"] = str(
                                failure_path
                            )
                            exc.details[
                                "diagnosticArtifactSha256"
                            ] = failure_sha256
                            break
                    raise
                arbitration = validate_semantic_job_arbitration(
                    arbitration,
                    expected_job_id=str(document.get("jobId") or ""),
                    expected_lattice=current_lattice,
                )
                atomic_write_json_no_replace(arbitration_path, arbitration)
            artifact_paths.append(arbitration_path)

            if arbitration["status"] == "ready-to-compose":
                composition_path = root / "semantic-composition.v1.json"
                if composition_path.exists():
                    composition = validate_semantic_composition(
                        read_json_strict(composition_path),
                        expected_document=document,
                        expected_lattice=current_lattice,
                        expected_arbitration=arbitration,
                    )
                    resumed_count += 1
                else:
                    composition = build_semantic_composition(
                        document,
                        current_lattice,
                        arbitration,
                    )
                    atomic_write_json_no_replace(composition_path, composition)
                artifact_paths.append(composition_path)
                return SemanticCompositionRunResult(
                    initial_lattice=initial,
                    final_lattice=current_lattice,
                    arbitration=arbitration,
                    composition=composition,
                    initial_lattice_path=initial_path,
                    final_lattice_path=current_lattice_path,
                    arbitration_path=arbitration_path,
                    composition_path=composition_path,
                    generation_paths=tuple(generation_paths),
                    artifact_paths=tuple(artifact_paths),
                    round_count=round_number,
                    resumed_artifact_count=resumed_count,
                )

            if round_number > self.max_rounds:
                raise WorkerError(
                    "SEMANTIC_COMPOSITION_ROUND_LIMIT",
                    "semantic arbitration requested another candidate after the "
                    "final bounded generation pass",
                    details={
                        "maxRounds": self.max_rounds,
                        "finalArbitrationRound": round_number,
                        "finalLatticeSha256": current_lattice["latticeSha256"],
                        "candidateGenerationRequestCount": len(
                            arbitration["candidateGenerationRequests"]
                        ),
                    },
                )

            self._release_arbitrator_resources_between_rounds()
            generation_path = (
                round_root / "semantic-candidate-generation.v1.json"
            )
            try:
                if generation_path.exists():
                    generation = validate_semantic_candidate_generation(
                        read_json_strict(generation_path),
                        expected_document=document,
                        expected_lattice=current_lattice,
                        expected_arbitration=arbitration,
                    )
                    resumed_count += 1
                else:
                    generation = self.generators.fulfill(
                        document,
                        current_lattice,
                        arbitration,
                    )
                    generation = validate_semantic_candidate_generation(
                        generation,
                        expected_document=document,
                        expected_lattice=current_lattice,
                        expected_arbitration=arbitration,
                    )
                    atomic_write_json_no_replace(generation_path, generation)
            except JobCancelled:
                raise
            except (
                OSError,
                ValueError,
                WorkerError,
                SemanticCandidateLatticeError,
                SemanticCompositionError,
            ) as exc:
                failure_root = round_root / "candidate-generation-failures"
                failure_index = 1
                while True:
                    failure_path = failure_root / (
                        "semantic-candidate-generation-failure-"
                        f"{failure_index:04d}.v1.json"
                    )
                    failure = {
                        "schemaVersion": "1.0.0",
                        "artifactType": "semantic-candidate-generation-failure",
                        "jobId": str(document.get("jobId") or ""),
                        "round": round_number,
                        "failureIndex": failure_index,
                        "input": {
                            "transcriptSha256": transcript_sha,
                            "sourceMediaSha256": current_lattice["binding"][
                                "sourceMediaSha256"
                            ],
                            "latticeSha256": current_lattice["latticeSha256"],
                            "arbitrationDecisionSha256": arbitration[
                                "decisionSha256"
                            ],
                            "candidateGenerationRequests": [
                                {
                                    "domain": request["domain"],
                                    "groupId": request["groupId"],
                                    "scopeId": request["scopeId"],
                                    "requestKind": request["requestKind"],
                                    "minimumAlternativeCount": request[
                                        "minimumAlternativeCount"
                                    ],
                                }
                                for request in arbitration[
                                    "candidateGenerationRequests"
                                ]
                            ],
                        },
                        "error": {
                            "code": (
                                exc.code
                                if isinstance(exc, WorkerError)
                                else "SEMANTIC_CANDIDATE_GENERATION_FAILED"
                            ),
                            "message": str(exc),
                            "exceptionType": type(exc).__name__,
                        },
                        "sourceContentPersisted": False,
                    }
                    try:
                        atomic_write_json_no_replace(failure_path, failure)
                    except FileExistsError:
                        failure_index += 1
                        continue
                    break
                diagnostic_sha = canonical_json_sha256(failure)
                if isinstance(exc, WorkerError):
                    exc.details["diagnosticArtifactPath"] = str(failure_path)
                    exc.details["diagnosticArtifactSha256"] = diagnostic_sha
                    raise
                raise WorkerError(
                    "SEMANTIC_CANDIDATE_GENERATION_FAILED",
                    "semantic candidate generation failed closed",
                    details={
                        "round": round_number,
                        "reason": str(exc),
                        "exceptionType": type(exc).__name__,
                        "diagnosticArtifactPath": str(failure_path),
                        "diagnosticArtifactSha256": diagnostic_sha,
                    },
                ) from exc
            generation_paths.append(generation_path)
            artifact_paths.append(generation_path)
            # Candidate generation may load ASR and diarization weights into
            # the same constrained host used by the 27B arbitrator.  Persist
            # and validate the generated evidence first, then unload those
            # weights before Ollama is loaded for the next round.  Registry
            # evidence caches remain available and adapter releases are
            # idempotent/reloadable.
            self._release_generation_resources_between_rounds()
            exhausted_request_group_ids.update(
                str(request["groupId"])
                for request in arbitration["candidateGenerationRequests"]
                if isinstance(request.get("groupId"), str)
            )
            next_lattice = generation["outputLattice"]
            if (
                next_lattice["latticeSha256"]
                == current_lattice["latticeSha256"]
            ):
                if (
                    generation["metrics"]["fulfilledRequestCount"] != 0
                    or generation["metrics"]["unfulfilledRequestCount"] == 0
                ):
                    raise WorkerError(
                        "SEMANTIC_CANDIDATE_GENERATION_STALLED",
                        "semantic candidate generation made no lattice progress",
                        details={"round": round_number},
                    )
            current_lattice_path = (
                round_root / "semantic-candidate-lattice.output.v1.json"
            )
            persisted_lattice, lattice_resumed = self._persist_or_resume(
                current_lattice_path,
                next_lattice,
                validator=lambda value: validate_semantic_candidate_lattice(
                    value,
                    expected_source_media_sha256=document["source"]["sha256"],
                    expected_transcript_sha256=transcript_sha,
                ),
            )
            resumed_count += int(lattice_resumed)
            artifact_paths.append(current_lattice_path)
            carried_lattice = current_lattice
            carried_arbitration = arbitration
            current_lattice = persisted_lattice

        raise WorkerError(
            "SEMANTIC_COMPOSITION_ROUND_LIMIT",
            "semantic arbitration did not reach a composable state within its bound",
            details={
                "maxRounds": self.max_rounds,
                "finalLatticeSha256": current_lattice["latticeSha256"],
            },
        )


__all__ = [
    "SemanticCompositionOrchestrator",
    "SemanticCompositionRunResult",
]
