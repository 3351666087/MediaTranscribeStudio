"""Fail-closed launcher and verifier for the Java PDF renderer sidecar."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from contracts.validate_contracts import (
    ContractError,
    EXPECTED_FACETS,
    validate_render_request,
    validate_report_document,
)


SCHEMA_VERSION = "1.0.0"
_REQUIRED_ARTIFACT_KEYS = (
    "reportDocumentPath",
    "htmlPath",
    "pdfPath",
    "manifestPath",
    "qualityReportPath",
    "repairQueuePath",
    "screenshotsDirectory",
    "contactSheetPath",
)
_REQUIRED_MANIFEST_TYPES = {
    "report-document",
    "canonical-xhtml",
    "pdf",
    "contact-sheet",
    "quality-report",
    "repair-queue",
}
_REQUIRED_HARD_GATES = (
    "PDF-OPENABLE",
    "PDF-PAGE-COUNT",
    "PDF-PAGE-SIZE",
    "PDF-TRANSCRIPT-TEXT-INTEGRITY",
    "PDF-SEGMENT-COUNT",
    "PDF-TIMESTAMP-INTEGRITY",
    "PDF-SPEAKER-SET-INTEGRITY",
    "PDF-FONT-EMBEDDED",
    "PDF-NO-BLANK-PAGES",
    "PDF-NO-CONTENT-OVERFLOW",
    "PDF-OFFLINE-ASSETS",
    "PDF-PAGE-EVIDENCE",
    "PDF-IMMUTABLE-CONTENT-HASH",
)


class PdfRenderError(RuntimeError):
    """Raised when the renderer or any of its evidence fails validation."""


@dataclass(frozen=True)
class PdfRenderOutcome:
    """Verified result plus canonical absolute artifact paths."""

    result: dict[str, Any]
    request_path: Path
    report_document_path: Path
    artifact_paths: dict[str, Path]
    report_document_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PdfRenderError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PdfRenderError(f"{label} root must be an object: {path}")
    return value


class JavaPdfClient:
    """Invoke a local renderer and verify every artifact before returning."""

    def __init__(
        self,
        renderer_command: Sequence[str | os.PathLike[str]],
        *,
        allowed_output_root: str | Path,
        renderer_cwd: Optional[str | Path] = None,
        timeout_seconds: float = 300.0,
        max_output_bytes: int = 2 * 1024 * 1024,
        minimum_score: float = 85.0,
        max_rounds: int = 5,
        capture_dpi: int = 144,
        margin_mm: float = 14.0,
        preferred_font: Optional[str] = None,
        template_id: str = "mts-cute-transcript-v1",
        environment: Optional[Mapping[str, str]] = None,
    ) -> None:
        command = [os.fspath(item) for item in renderer_command]
        if not command or any(not str(item).strip() for item in command):
            raise PdfRenderError("renderer_command must contain non-empty arguments")
        self.renderer_command = tuple(command)
        self.allowed_output_root = Path(allowed_output_root).expanduser().resolve()
        self.renderer_cwd = (
            Path(renderer_cwd).expanduser().resolve()
            if renderer_cwd is not None
            else None
        )
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise PdfRenderError("timeout_seconds must be positive")
        self.max_output_bytes = int(max_output_bytes)
        if self.max_output_bytes < 1024:
            raise PdfRenderError("max_output_bytes must be at least 1024")
        self.minimum_score = float(minimum_score)
        if self.minimum_score < 85.0 or self.minimum_score > 100.0:
            raise PdfRenderError("minimum_score must be between 85 and 100")
        self.max_rounds = int(max_rounds)
        if self.max_rounds < 1 or self.max_rounds > 5:
            raise PdfRenderError("max_rounds must be between 1 and 5")
        self.capture_dpi = int(capture_dpi)
        if self.capture_dpi < 96 or self.capture_dpi > 300:
            raise PdfRenderError("capture_dpi must be between 96 and 300")
        self.margin_mm = float(margin_mm)
        if self.margin_mm < 8 or self.margin_mm > 30:
            raise PdfRenderError("margin_mm must be between 8 and 30")
        self.preferred_font = str(preferred_font or "").strip() or None
        self.template_id = str(template_id or "").strip()
        if not self.template_id:
            raise PdfRenderError("template_id must not be empty")
        self.environment = {
            str(key): str(value) for key, value in (environment or {}).items()
        }

    @classmethod
    def from_jar(
        cls,
        jar_path: str | Path,
        *,
        allowed_output_root: str | Path,
        java_executable: str | Path = "java",
        **kwargs: Any,
    ) -> "JavaPdfClient":
        jar = Path(jar_path).expanduser()
        if not jar.is_file():
            raise PdfRenderError(f"Java PDF renderer JAR is missing: {jar}")
        jar = jar.resolve(strict=True)
        if jar.suffix.casefold() != ".jar":
            raise PdfRenderError("Java PDF renderer path must end in .jar")

        java_text = os.fspath(java_executable).strip()
        if not java_text:
            raise PdfRenderError("java_executable must not be empty")
        contains_separator = any(
            separator and separator in java_text
            for separator in (os.sep, os.altsep)
        )
        if contains_separator or Path(java_text).is_absolute():
            java_path = Path(java_text).expanduser()
            if not java_path.is_file():
                raise PdfRenderError(
                    f"Java executable is missing: {java_path}"
                )
            resolved_java = str(java_path.resolve(strict=True))
        else:
            resolved_java = shutil.which(java_text) or ""
            if not resolved_java:
                raise PdfRenderError(
                    f"Java executable is not available on PATH: {java_text}"
                )
        return cls(
            [
                resolved_java,
                "-jar",
                str(jar),
                "--request",
                "{request}",
            ],
            allowed_output_root=allowed_output_root,
            **kwargs,
        )

    def create_request(
        self,
        *,
        report_document_path: Path,
        output_directory: Path,
        job_id: str,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        normalized_job_id = str(job_id or "").strip()
        if not normalized_job_id or len(normalized_job_id) > 160:
            raise PdfRenderError("job_id must contain 1-160 characters")
        normalized_request_id = str(
            request_id or f"{normalized_job_id}:pdf"
        ).strip()
        if not normalized_request_id or len(normalized_request_id) > 160:
            raise PdfRenderError("request_id must contain 1-160 characters")
        font_policy: dict[str, Any] = {
            "requireEmbeddedCjk": True,
            "allowSystemFallback": False,
        }
        if self.preferred_font:
            font_policy["preferredFont"] = self.preferred_font
        request: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "requestId": normalized_request_id,
            "jobId": normalized_job_id,
            "reportDocumentPath": str(report_document_path.resolve()),
            "outputDirectory": str(output_directory.resolve()),
            "renderer": {
                "provider": "java-openhtmltopdf",
                "offline": True,
                "page": {"size": "A4", "marginMm": self.margin_mm},
                "fontPolicy": font_policy,
                "templateId": self.template_id,
            },
            "qualityPolicy": {
                "requireHardGates": True,
                "minimumScore": self.minimum_score,
                "maxRounds": self.max_rounds,
                "captureDpi": self.capture_dpi,
                "facetIds": list(EXPECTED_FACETS),
            },
        }
        try:
            validate_render_request(request)
        except (ContractError, TypeError, ValueError) as exc:
            raise PdfRenderError(f"invalid render request: {exc}") from exc
        return request

    def render(
        self,
        report_document: Mapping[str, Any],
        *,
        job_id: str,
        output_directory: str | Path,
        request_id: Optional[str] = None,
    ) -> PdfRenderOutcome:
        try:
            validate_report_document(dict(report_document))
        except (ContractError, TypeError, ValueError) as exc:
            raise PdfRenderError(f"refusing to render invalid ReportDocument: {exc}") from exc

        output_root = self._safe_output_directory(output_directory)
        output_root.mkdir(parents=True, exist_ok=True)
        control_directory = output_root / ".render"
        input_directory = output_root / "input"
        control_directory.mkdir(parents=True, exist_ok=True)
        input_directory.mkdir(parents=True, exist_ok=True)
        report_path = input_directory / "report-document.json"
        request_path = control_directory / "request.json"
        _atomic_write_json(report_path, dict(report_document))
        report_sha256 = _sha256_file(report_path)

        request = self.create_request(
            report_document_path=report_path,
            output_directory=output_root,
            job_id=job_id,
            request_id=request_id,
        )
        _atomic_write_json(request_path, request)
        stdout = self._run_renderer(request_path, control_directory)
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise PdfRenderError("renderer stdout is not a single JSON document") from exc
        if not isinstance(result, dict):
            raise PdfRenderError("renderer result root must be an object")

        artifact_paths = self._verify_result(
            result,
            request=request,
            output_root=output_root,
            report_path=report_path,
            report_sha256=report_sha256,
            document_id=str(report_document["documentId"]),
        )
        return PdfRenderOutcome(
            result=result,
            request_path=request_path,
            report_document_path=report_path,
            artifact_paths=artifact_paths,
            report_document_sha256=report_sha256,
        )

    def _safe_output_directory(self, value: str | Path) -> Path:
        requested = Path(value).expanduser()
        if requested.is_absolute():
            candidate = requested.resolve()
        else:
            candidate = (self.allowed_output_root / requested).resolve()
        if not self._is_within(candidate, self.allowed_output_root):
            raise PdfRenderError(
                f"output directory escapes allowed root: {candidate}"
            )
        self._reject_linked_path(candidate, stop_at=self.allowed_output_root)
        return candidate

    @staticmethod
    def _is_within(candidate: Path, root: Path) -> bool:
        try:
            return os.path.commonpath([str(candidate), str(root)]) == str(root)
        except ValueError:
            return False

    @staticmethod
    def _reject_linked_path(candidate: Path, *, stop_at: Path) -> None:
        current = candidate
        while True:
            if current.exists():
                is_junction = bool(
                    getattr(os.path, "isjunction", lambda _path: False)(current)
                )
                if current.is_symlink() or is_junction:
                    raise PdfRenderError(
                        f"linked output paths are not allowed: {current}"
                    )
            if current == stop_at or current.parent == current:
                break
            current = current.parent

    def _command_for(self, request_path: Path) -> list[str]:
        request_text = str(request_path)
        if any("{request}" in argument for argument in self.renderer_command):
            return [
                argument.replace("{request}", request_text)
                for argument in self.renderer_command
            ]
        return [*self.renderer_command, request_text]

    def _run_renderer(self, request_path: Path, control_directory: Path) -> str:
        command = self._command_for(request_path)
        stdout_path = control_directory / "renderer.stdout"
        stderr_path = control_directory / "renderer.stderr"
        environment = os.environ.copy()
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            environment.pop(key, None)
        environment.update(
            {
                "MTS_OFFLINE": "1",
                "NO_PROXY": "*",
                "no_proxy": "*",
                **self.environment,
            }
        )

        creationflags = 0
        startupinfo = None
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            creationflags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
        else:
            popen_kwargs["start_new_session"] = True

        started = time.monotonic()
        with stdout_path.open("wb") as stdout_handle, stderr_path.open(
            "wb"
        ) as stderr_handle:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.renderer_cwd) if self.renderer_cwd else None,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    shell=False,
                    creationflags=creationflags,
                    startupinfo=startupinfo,
                    **popen_kwargs,
                )
            except OSError as exc:
                raise PdfRenderError(
                    f"failed to start Java PDF renderer: {command[0]}"
                ) from exc

            failure: Optional[str] = None
            while process.poll() is None:
                if time.monotonic() - started > self.timeout_seconds:
                    failure = (
                        f"renderer exceeded timeout of {self.timeout_seconds:.2f}s"
                    )
                    break
                stdout_handle.flush()
                stderr_handle.flush()
                if (
                    stdout_path.stat().st_size > self.max_output_bytes
                    or stderr_path.stat().st_size > self.max_output_bytes
                ):
                    failure = (
                        f"renderer output exceeded {self.max_output_bytes} bytes"
                    )
                    break
                time.sleep(0.05)
            if failure is not None:
                self._terminate_process_tree(process)
                raise PdfRenderError(failure)
            return_code = process.wait()

        stdout_bytes = stdout_path.read_bytes()
        stderr_bytes = stderr_path.read_bytes()
        if len(stdout_bytes) > self.max_output_bytes or len(stderr_bytes) > self.max_output_bytes:
            raise PdfRenderError(
                f"renderer output exceeded {self.max_output_bytes} bytes"
            )
        stdout_text = stdout_bytes.decode("utf-8", errors="strict").strip()
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        if return_code != 0:
            detail = stderr_text[-2000:] or stdout_text[-2000:] or "no diagnostic output"
            raise PdfRenderError(
                f"renderer exited with code {return_code}: {detail}"
            )
        if not stdout_text:
            raise PdfRenderError("renderer returned empty stdout")
        return stdout_text

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                startupinfo=startupinfo,
                check=False,
            )
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _verify_result(
        self,
        result: Mapping[str, Any],
        *,
        request: Mapping[str, Any],
        output_root: Path,
        report_path: Path,
        report_sha256: str,
        document_id: str,
    ) -> dict[str, Path]:
        required_top = {
            "schemaVersion",
            "requestId",
            "jobId",
            "status",
            "rendererVersion",
            "roundsCompleted",
            "artifacts",
            "quality",
        }
        allowed_top = {*required_top, "error"}
        missing_top = required_top - set(result)
        if missing_top:
            raise PdfRenderError(
                f"renderer result is missing fields: {sorted(missing_top)}"
            )
        unknown_top = set(result) - allowed_top
        if unknown_top:
            raise PdfRenderError(
                f"renderer result contains unsupported fields: {sorted(unknown_top)}"
            )
        if result.get("schemaVersion") != SCHEMA_VERSION:
            raise PdfRenderError("renderer returned an unsupported schemaVersion")
        if result.get("requestId") != request["requestId"]:
            raise PdfRenderError("renderer requestId does not match the request")
        if result.get("jobId") != request["jobId"]:
            raise PdfRenderError("renderer jobId does not match the request")
        if result.get("status") != "passed":
            error = result.get("error")
            detail = error.get("message") if isinstance(error, Mapping) else ""
            raise PdfRenderError(
                f"renderer did not pass: {result.get('status')}"
                + (f" ({detail})" if detail else "")
            )
        renderer_version = str(result.get("rendererVersion") or "").strip()
        if not renderer_version or len(renderer_version) > 80:
            raise PdfRenderError("rendererVersion is missing or invalid")
        rounds_value = result.get("roundsCompleted")
        if isinstance(rounds_value, bool) or not isinstance(rounds_value, int):
            raise PdfRenderError("roundsCompleted must be an integer")
        rounds = rounds_value
        if rounds < 1 or rounds > self.max_rounds:
            raise PdfRenderError("roundsCompleted is outside the permitted range")

        quality_summary = result.get("quality")
        if not isinstance(quality_summary, Mapping):
            raise PdfRenderError("renderer quality summary must be an object")
        if quality_summary.get("status") != "passed":
            raise PdfRenderError("renderer quality status is not passed")
        if quality_summary.get("hardGatesPassed") is not True:
            raise PdfRenderError("renderer hard gates did not pass")
        try:
            result_score = float(quality_summary.get("score"))
        except (TypeError, ValueError) as exc:
            raise PdfRenderError("renderer quality score must be numeric") from exc
        if not math.isfinite(result_score):
            raise PdfRenderError("renderer quality score must be finite")
        if result_score < self.minimum_score:
            raise PdfRenderError(
                f"renderer score {result_score:.2f} is below {self.minimum_score:.2f}"
            )

        artifacts = result.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise PdfRenderError("renderer artifacts must be an object")
        unknown_artifacts = set(artifacts) - set(_REQUIRED_ARTIFACT_KEYS)
        if unknown_artifacts:
            raise PdfRenderError(
                "renderer artifacts contain unsupported fields: "
                f"{sorted(unknown_artifacts)}"
            )
        missing_artifacts = set(_REQUIRED_ARTIFACT_KEYS) - set(artifacts)
        if missing_artifacts:
            raise PdfRenderError(
                f"renderer result is missing artifact paths: {sorted(missing_artifacts)}"
            )
        resolved: dict[str, Path] = {}
        for key in _REQUIRED_ARTIFACT_KEYS:
            path = self._resolve_artifact_path(
                artifacts[key],
                output_root=output_root,
                label=key,
            )
            if key == "screenshotsDirectory":
                if not path.is_dir():
                    raise PdfRenderError(f"{key} does not exist as a directory: {path}")
            elif not path.is_file():
                raise PdfRenderError(f"{key} does not exist as a file: {path}")
            resolved[key] = path
        if resolved["reportDocumentPath"] != report_path.resolve():
            raise PdfRenderError(
                "renderer result points at a different ReportDocument"
            )
        if _sha256_file(resolved["reportDocumentPath"]) != report_sha256:
            raise PdfRenderError("ReportDocument changed during rendering")
        with resolved["pdfPath"].open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise PdfRenderError("pdfPath does not contain a PDF header")

        quality_report = _load_json_object(
            resolved["qualityReportPath"], label="PDF quality report"
        )
        self._verify_quality_report(
            quality_report,
            output_root=output_root,
            document_id=document_id,
            rounds=rounds,
            result_score=result_score,
        )
        manifest = _load_json_object(
            resolved["manifestPath"], label="artifact manifest"
        )
        self._verify_manifest(
            manifest,
            output_root=output_root,
            request=request,
            document_id=document_id,
            renderer_version=renderer_version,
            report_path=report_path,
            report_sha256=report_sha256,
        )
        return resolved

    def _resolve_artifact_path(
        self,
        raw_value: Any,
        *,
        output_root: Path,
        label: str,
    ) -> Path:
        text = str(raw_value or "").strip()
        if not text:
            raise PdfRenderError(f"{label} is empty")
        relative = Path(text)
        if relative.is_absolute() or ".." in relative.parts:
            raise PdfRenderError(f"{label} must be a safe relative path")
        candidate = (output_root / relative).resolve()
        if not self._is_within(candidate, output_root):
            raise PdfRenderError(f"{label} escapes the render output directory")
        self._reject_linked_path(candidate, stop_at=output_root)
        return candidate

    def _verify_quality_report(
        self,
        report: Mapping[str, Any],
        *,
        output_root: Path,
        document_id: str,
        rounds: int,
        result_score: float,
    ) -> None:
        if report.get("schemaVersion") != SCHEMA_VERSION:
            raise PdfRenderError("quality report has an unsupported schemaVersion")
        if report.get("documentId") != document_id:
            raise PdfRenderError("quality report documentId does not match")
        if report.get("status") != "passed":
            raise PdfRenderError("quality report status is not passed")
        if report.get("hardGatesPassed") is not True:
            raise PdfRenderError("quality report hard gates did not pass")
        try:
            report_round = int(report.get("round"))
            minimum_score = float(report.get("minimumScore"))
            score = float(report.get("score"))
        except (TypeError, ValueError) as exc:
            raise PdfRenderError("quality report numeric fields are invalid") from exc
        if not math.isfinite(minimum_score) or not math.isfinite(score):
            raise PdfRenderError("quality report numeric fields must be finite")
        if report_round != rounds:
            raise PdfRenderError("quality report round does not match renderer result")
        if minimum_score < self.minimum_score or score < self.minimum_score:
            raise PdfRenderError("quality report is below the configured threshold")
        if abs(score - result_score) > 1e-9:
            raise PdfRenderError("quality report score does not match renderer result")

        hard_gates = report.get("hardGates")
        if (
            not isinstance(hard_gates, list)
            or not hard_gates
            or any(
                not isinstance(gate, Mapping) or gate.get("status") != "passed"
                for gate in hard_gates
            )
        ):
            raise PdfRenderError(
                "every PDF hard gate must be present and passed"
            )
        hard_gate_ids = [
            str(gate.get("id") or "")
            for gate in hard_gates
            if isinstance(gate, Mapping)
        ]
        if hard_gate_ids != list(_REQUIRED_HARD_GATES):
            raise PdfRenderError(
                "quality report must preserve all 13 PDF hard gates in order"
            )
        facets = report.get("facets")
        if not isinstance(facets, list):
            raise PdfRenderError("quality report facets must be an array")
        if [facet.get("id") for facet in facets if isinstance(facet, Mapping)] != list(
            EXPECTED_FACETS
        ):
            raise PdfRenderError(
                "quality report must preserve all 14 Design Pack facets in order"
            )
        if any(
            not isinstance(facet, Mapping) or facet.get("status") != "passed"
            for facet in facets
        ):
            raise PdfRenderError("all Design Pack facets must pass")
        try:
            facet_weight = sum(float(facet["weight"]) for facet in facets)
        except (KeyError, TypeError, ValueError) as exc:
            raise PdfRenderError("Design Pack facet weights must be numeric") from exc
        if abs(facet_weight - 1.0) > 1e-9:
            raise PdfRenderError("Design Pack facet weights must sum exactly to 1")

        evidence = report.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise PdfRenderError("quality report must include verified evidence")
        for index, item in enumerate(evidence):
            if not isinstance(item, Mapping) or item.get("verified") is not True:
                raise PdfRenderError(
                    f"quality report evidence[{index}] is not verified"
                )
            path = self._resolve_artifact_path(
                item.get("relativePath"),
                output_root=output_root,
                label=f"quality evidence[{index}]",
            )
            if not path.is_file():
                raise PdfRenderError(
                    f"quality evidence[{index}] does not exist: {path}"
                )
            if _sha256_file(path) != item.get("sha256"):
                raise PdfRenderError(
                    f"quality evidence[{index}] SHA-256 mismatch"
                )
        repairs = report.get("repairQueue")
        if not isinstance(repairs, list):
            raise PdfRenderError("quality report repairQueue must be an array")
        if any(
            isinstance(repair, Mapping)
            and repair.get("status") in {"pending", "blocked"}
            for repair in repairs
        ):
            raise PdfRenderError(
                "a passed quality report cannot contain unresolved repairs"
            )

    def _verify_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        output_root: Path,
        request: Mapping[str, Any],
        document_id: str,
        renderer_version: str,
        report_path: Path,
        report_sha256: str,
    ) -> None:
        if manifest.get("schemaVersion") != SCHEMA_VERSION:
            raise PdfRenderError("artifact manifest has an unsupported schemaVersion")
        if manifest.get("jobId") != request["jobId"]:
            raise PdfRenderError("artifact manifest jobId does not match")
        if manifest.get("documentId") != document_id:
            raise PdfRenderError("artifact manifest documentId does not match")
        if manifest.get("rendererVersion") != renderer_version:
            raise PdfRenderError("artifact manifest rendererVersion does not match")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise PdfRenderError("artifact manifest must contain artifacts")
        seen_ids: set[str] = set()
        seen_types: set[str] = set()
        report_manifest_entry: Optional[Mapping[str, Any]] = None
        for index, item in enumerate(artifacts):
            if not isinstance(item, Mapping):
                raise PdfRenderError(f"manifest artifact[{index}] must be an object")
            artifact_id = str(item.get("artifactId") or "").strip()
            if not artifact_id or artifact_id in seen_ids:
                raise PdfRenderError(
                    f"manifest artifact[{index}] has a missing or duplicate id"
                )
            seen_ids.add(artifact_id)
            artifact_type = str(item.get("type") or "").strip()
            seen_types.add(artifact_type)
            if item.get("verified") is not True:
                raise PdfRenderError(
                    f"manifest artifact[{index}] is not marked verified"
                )
            path = self._resolve_artifact_path(
                item.get("relativePath"),
                output_root=output_root,
                label=f"manifest artifact[{index}]",
            )
            if not path.is_file():
                raise PdfRenderError(
                    f"manifest artifact[{index}] does not exist: {path}"
                )
            try:
                expected_bytes = int(item.get("bytes"))
            except (TypeError, ValueError) as exc:
                raise PdfRenderError(
                    f"manifest artifact[{index}] bytes is invalid"
                ) from exc
            if path.stat().st_size != expected_bytes:
                raise PdfRenderError(
                    f"manifest artifact[{index}] byte count mismatch"
                )
            if _sha256_file(path) != item.get("sha256"):
                raise PdfRenderError(
                    f"manifest artifact[{index}] SHA-256 mismatch"
                )
            if artifact_type == "report-document":
                report_manifest_entry = item
                if path != report_path.resolve():
                    raise PdfRenderError(
                        "manifest report-document path does not match the input"
                    )
        missing_types = _REQUIRED_MANIFEST_TYPES - seen_types
        if missing_types:
            raise PdfRenderError(
                f"artifact manifest is missing types: {sorted(missing_types)}"
            )
        if report_manifest_entry is None:
            raise PdfRenderError("artifact manifest lacks the ReportDocument")
        if report_manifest_entry.get("sha256") != report_sha256:
            raise PdfRenderError(
                "artifact manifest ReportDocument SHA-256 does not match"
            )
