"""Filesystem trust boundary for local media and generated artifacts."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Iterable

from .errors import WorkerError, invalid_request


def _contains_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _absolute_lexical(value: str | os.PathLike[str]) -> Path:
    """Return an absolute path without resolving links or junctions."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(candidate), str(root)]) == str(root)
    except ValueError:
        return False


def _is_link_or_junction(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        metadata = path.lstat()
    except (FileNotFoundError, OSError):
        return False
    is_junction = getattr(os.path, "isjunction", None)
    if is_junction is not None:
        try:
            if is_junction(path):
                return True
        except OSError:
            return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _reject_linked_components(candidate: Path, root: Path) -> None:
    """Reject links/reparse points before resolving their target."""

    current = candidate
    inspected: list[Path] = []
    while True:
        inspected.append(current)
        if current == root or current.parent == current:
            break
        current = current.parent
    if root not in inspected:
        raise WorkerError(
            "PATH_OUTSIDE_ALLOWED_ROOT",
            "path does not descend from its configured root",
        )
    for component in reversed(inspected):
        if component.exists() and _is_link_or_junction(component):
            raise WorkerError(
                "LINKED_PATH_FORBIDDEN",
                "symbolic links and Windows junctions are not allowed",
                details={"path": str(component)},
            )


def _reject_configured_root_links(path: Path) -> None:
    """Reject a configured root when it or an existing parent is a link."""

    for component in (*reversed(path.parents), path):
        if _is_link_or_junction(component):
            raise ValueError(
                f"configured roots must not contain linked path components: {component}"
            )


class PathPolicy:
    """Resolve paths without allowing traversal, symlink, or junction escape."""

    def __init__(
        self,
        *,
        allowed_input_roots: Iterable[str | os.PathLike[str]],
        allowed_output_root: str | os.PathLike[str],
    ) -> None:
        lexical_roots = tuple(_absolute_lexical(value) for value in allowed_input_roots)
        if not lexical_roots:
            raise ValueError("at least one allowed input root is required")
        for root in lexical_roots:
            _reject_configured_root_links(root)
        if any(not root.is_dir() for root in lexical_roots):
            raise ValueError("every allowed input root must be an existing directory")
        roots = tuple(root.resolve(strict=True) for root in lexical_roots)

        lexical_output_root = _absolute_lexical(allowed_output_root)
        _reject_configured_root_links(lexical_output_root)
        lexical_output_root.mkdir(parents=True, exist_ok=True)
        _reject_configured_root_links(lexical_output_root)
        if not lexical_output_root.is_dir():
            raise ValueError("allowed output root must be a directory")
        output_root = lexical_output_root.resolve(strict=True)
        self.allowed_input_roots = roots
        self.allowed_output_root = output_root

    @staticmethod
    def _validate_raw(value: object, field_name: str) -> str:
        if not isinstance(value, (str, os.PathLike)):
            raise invalid_request(f"{field_name} must be a path string")
        text = os.fspath(value)
        if not text or not text.strip():
            raise invalid_request(f"{field_name} must not be empty")
        if _contains_control_characters(text):
            raise invalid_request(
                f"{field_name} contains forbidden control characters"
            )
        path = Path(text)
        if ".." in path.parts:
            raise WorkerError(
                "PATH_TRAVERSAL_FORBIDDEN",
                f"{field_name} must not contain parent traversal",
            )
        return text

    def resolve_source(self, value: object) -> Path:
        text = self._validate_raw(value, "sourcePath")
        requested = Path(text).expanduser()
        candidates: list[tuple[Path, Path]] = []
        if requested.is_absolute():
            for root in self.allowed_input_roots:
                if _is_within(requested.absolute(), root):
                    candidates.append((requested.absolute(), root))
        else:
            for root in self.allowed_input_roots:
                candidates.append(((root / requested).absolute(), root))

        valid: list[Path] = []
        for lexical, root in candidates:
            if not _is_within(lexical, root):
                continue
            _reject_linked_components(lexical, root)
            try:
                resolved = lexical.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            if _is_within(resolved, root) and resolved.is_file():
                valid.append(resolved)
        unique = tuple(dict.fromkeys(valid))
        if len(unique) != 1:
            code = "SOURCE_NOT_FOUND" if not unique else "SOURCE_PATH_AMBIGUOUS"
            raise WorkerError(
                code,
                "sourcePath must resolve to exactly one regular file under an allowed input root",
            )
        return unique[0]

    def resolve_output(self, value: object) -> Path:
        text = self._validate_raw(value, "outputDirectory")
        requested = Path(text).expanduser()
        lexical = (
            requested.absolute()
            if requested.is_absolute()
            else (self.allowed_output_root / requested).absolute()
        )
        if not _is_within(lexical, self.allowed_output_root):
            raise WorkerError(
                "OUTPUT_PATH_OUTSIDE_ROOT",
                "outputDirectory escapes the configured output root",
            )
        _reject_linked_components(lexical, self.allowed_output_root)
        resolved = lexical.resolve(strict=False)
        if (
            resolved == self.allowed_output_root
            or not _is_within(resolved, self.allowed_output_root)
        ):
            raise WorkerError(
                "OUTPUT_PATH_OUTSIDE_ROOT",
                "outputDirectory must be a child of the configured output root",
            )
        if resolved.exists():
            if not resolved.is_dir():
                raise WorkerError(
                    "OUTPUT_PATH_INVALID", "outputDirectory exists but is not a directory"
                )
            raise WorkerError(
                "OUTPUT_DIRECTORY_ALREADY_EXISTS",
                "outputDirectory must not already exist for a new job",
            )
        return resolved

    def create_output_directory(self, output_directory: Path) -> Path:
        """Exclusively claim a previously resolved per-job output directory."""

        lexical = _absolute_lexical(output_directory)
        if (
            lexical == self.allowed_output_root
            or not _is_within(lexical, self.allowed_output_root)
        ):
            raise WorkerError(
                "OUTPUT_PATH_OUTSIDE_ROOT",
                "outputDirectory must be a child of the configured output root",
            )
        _reject_linked_components(lexical, self.allowed_output_root)
        try:
            lexical.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise WorkerError(
                "OUTPUT_DIRECTORY_ALREADY_EXISTS",
                "outputDirectory was claimed before this job could start",
            ) from exc
        except OSError as exc:
            raise WorkerError(
                "OUTPUT_DIRECTORY_CREATE_FAILED",
                "outputDirectory could not be created safely",
                details={"exceptionType": type(exc).__name__},
            ) from exc

        try:
            _reject_linked_components(lexical, self.allowed_output_root)
            resolved = lexical.resolve(strict=True)
            if (
                resolved == self.allowed_output_root
                or not _is_within(resolved, self.allowed_output_root)
                or not resolved.is_dir()
            ):
                raise WorkerError(
                    "OUTPUT_PATH_OUTSIDE_ROOT",
                    "created outputDirectory failed boundary verification",
                )
            return resolved
        except Exception:
            try:
                lexical.rmdir()
            except OSError:
                pass
            raise

    def verify_artifact(self, path: Path, output_directory: Path) -> Path:
        lexical = path.expanduser()
        if not lexical.is_absolute():
            lexical = output_directory / lexical
        lexical = lexical.absolute()
        if not _is_within(lexical, output_directory):
            raise WorkerError(
                "RENDER_ARTIFACT_OUTSIDE_OUTPUT",
                "renderer returned an artifact outside the job output directory",
                details={"path": str(path)},
            )
        _reject_linked_components(lexical, output_directory)
        try:
            resolved = lexical.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise WorkerError(
                "RENDER_ARTIFACT_MISSING",
                "renderer returned a missing artifact",
                details={"path": str(path)},
            ) from exc
        if not _is_within(resolved, output_directory):
            raise WorkerError(
                "RENDER_ARTIFACT_OUTSIDE_OUTPUT",
                "renderer artifact resolved outside the job output directory",
            )
        if not resolved.is_file():
            raise WorkerError(
                "RENDER_ARTIFACT_INVALID",
                "renderer artifacts must be regular files",
                details={"path": str(path)},
            )
        return resolved
