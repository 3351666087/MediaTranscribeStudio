"""Offline, auditable Windows font evidence for subtitle visual QA.

The provider in this module intentionally proves less rather than guessing:

* a font must be present in a local Windows font registry key;
* the registry value must resolve to a regular, non-reparse font file below
  an explicitly trusted Windows font directory;
* the registry label must agree with family/style metadata parsed from the
  font file;
* the requested family must exactly match parsed OpenType family metadata;
* every non-whitespace cue code point must map to a non-``.notdef`` glyph in
  the selected face;
* the evidence JSON is bound to stable SHA-256 snapshots of the font and both
  representative frames.

No network service, font-name-only assertion, extension allowlist, or fallback
success is used.  Unsupported platforms and unavailable local evidence return
``None`` (or a negative observation where useful), which leaves the existing
``fontGlyph`` QA gate closed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import stat
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from backend.subtitle_visual_evidence import (
    ComponentDescriptor,
    FontEvidenceObservation,
    VerifiedFontClaim,
)


WINDOWS_FONT_EVIDENCE_SCHEMA_VERSION = "1.0.0"

_PROVIDER_NAME = "windows-local-font-evidence"
_PROVIDER_VERSION = "1.0.0"
_FONT_EXTENSIONS = frozenset({".ttf", ".otf", ".ttc", ".otc"})
_REGISTRY_KEYS = (
    r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Fonts",
)
_MAX_TEXT_CHARACTERS = 100_000
_MAX_FAMILY_CHARACTERS = 320
_MAX_REGISTRY_TEXT = 32_768
_MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_FONT_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_FRAME_BYTES = 128 * 1024 * 1024
_DEFAULT_MAX_REGISTRY_ENTRIES = 32_768
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_WINDOWS_ENVIRONMENT_VARIABLE = re.compile(r"%([^%]+)%")
_REGISTRY_FONT_SUFFIX = re.compile(
    r"\s*\((?:true\s*type|opentype|all\s+res|truetype|type\s*1)\)\s*$",
    re.IGNORECASE,
)
_STYLE_WORDS = frozenset(
    {
        "black",
        "bold",
        "book",
        "condensed",
        "demi",
        "demibold",
        "expanded",
        "extra",
        "extrabold",
        "extralight",
        "heavy",
        "italic",
        "light",
        "medium",
        "narrow",
        "oblique",
        "regular",
        "roman",
        "semibold",
        "semicondensed",
        "thin",
        "ultra",
        "ultrabold",
        "ultralight",
    }
)
_REGULAR_STYLE_WORDS = frozenset({"book", "normal", "regular", "roman"})


class WindowsFontEvidenceError(ValueError):
    """The provider could not safely interpret local font evidence."""


@dataclass(frozen=True)
class WindowsFontRegistryEntry:
    """One local Windows font registry value.

    Tests may inject these records on any host platform.  Production uses the
    default read-only ``winreg`` enumerator.
    """

    hive: str
    key_path: str
    view: str
    value_name: str
    value_data: str
    value_type: str = "REG_SZ"


@dataclass(frozen=True)
class _FileFingerprint:
    size: int
    mtime_ns: int
    device: int
    inode: int


@dataclass(frozen=True)
class _FontSnapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: _FileFingerprint


@dataclass(frozen=True)
class _FontFace:
    face_index: int
    family_names: tuple[str, ...]
    style_names: tuple[str, ...]
    full_names: tuple[str, ...]
    postscript_names: tuple[str, ...]
    covered_code_points: frozenset[int]
    cmap_sha256: str

    @property
    def primary_family(self) -> str:
        return self.family_names[0]

    @property
    def primary_style(self) -> str:
        return self.style_names[0] if self.style_names else "Regular"


@dataclass(frozen=True)
class _RegisteredFace:
    entry: WindowsFontRegistryEntry
    snapshot: _FontSnapshot
    face: _FontFace


@dataclass(frozen=True)
class _Coverage:
    expected: int
    covered: int
    missing_code_points: tuple[str, ...]
    missing_occurrences: int
    requested_code_points: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return (
            self.expected == self.covered
            and not self.missing_code_points
            and self.missing_occurrences == 0
        )


@dataclass(frozen=True)
class _Selection:
    registered_face: _RegisteredFace
    requested_family: str
    resolved_family: str
    coverage: _Coverage


RegistryReader = Callable[[], Iterable[WindowsFontRegistryEntry]]


class WindowsFontEvidenceProvider:
    """Collect cryptographically bound local Windows font evidence.

    ``registry_reader``, ``allowed_font_roots`` and ``platform_name`` are
    injectable so the Windows trust policy can be tested on non-Windows CI.
    An injected platform name never enables the production registry reader;
    tests must also inject registry records.
    """

    def __init__(
        self,
        *,
        registry_reader: RegistryReader | None = None,
        allowed_font_roots: Sequence[str | Path] | None = None,
        evidence_root: str | Path | None = None,
        platform_name: str | None = None,
        max_font_bytes: int = _DEFAULT_MAX_FONT_BYTES,
        max_frame_bytes: int = _DEFAULT_MAX_FRAME_BYTES,
        max_registry_entries: int = _DEFAULT_MAX_REGISTRY_ENTRIES,
    ) -> None:
        self._host_platform = platform.system()
        self._platform_name = (
            self._host_platform if platform_name is None else platform_name
        )
        if not isinstance(self._platform_name, str):
            raise WindowsFontEvidenceError("platform_name must be text")
        self._platform_name = self._platform_name.strip()
        if not self._platform_name:
            raise WindowsFontEvidenceError("platform_name must not be empty")

        self._registry_reader = registry_reader
        roots = (
            tuple(allowed_font_roots)
            if allowed_font_roots is not None
            else _default_windows_font_roots()
        )
        if len(roots) > 32:
            raise WindowsFontEvidenceError(
                "allowed_font_roots exceeds the 32-root safety limit"
            )
        self._configured_roots = tuple(
            _coerce_path(root, label="allowed font root") for root in roots
        )
        self._evidence_root = (
            None
            if evidence_root is None
            else _coerce_path(evidence_root, label="evidence root")
        )
        self._max_font_bytes = _bounded_positive_integer(
            max_font_bytes,
            label="max_font_bytes",
            maximum=2 * 1024 * 1024 * 1024,
        )
        self._max_frame_bytes = _bounded_positive_integer(
            max_frame_bytes,
            label="max_frame_bytes",
            maximum=2 * 1024 * 1024 * 1024,
        )
        self._max_registry_entries = _bounded_positive_integer(
            max_registry_entries,
            label="max_registry_entries",
            maximum=100_000,
        )
        configuration = {
            "schemaVersion": WINDOWS_FONT_EVIDENCE_SCHEMA_VERSION,
            "platformPolicy": "windows-only-fail-closed",
            "allowedFontRoots": sorted(
                str(path.absolute()) for path in self._configured_roots
            ),
            "fontExtensions": sorted(_FONT_EXTENSIONS),
            "fontToolsVersion": _fonttools_version(),
            "maxFontBytes": self._max_font_bytes,
            "maxFrameBytes": self._max_frame_bytes,
            "maxRegistryEntries": self._max_registry_entries,
            "registrySource": (
                "injected-read-only"
                if registry_reader is not None
                else "windows-winreg-read-only"
            ),
        }
        self._descriptor = ComponentDescriptor(
            name=_PROVIDER_NAME,
            version=_PROVIDER_VERSION,
            configuration_sha256=_canonical_sha256(configuration),
        )

        self._index_signature: str | None = None
        self._index: tuple[_RegisteredFace, ...] = ()

    @property
    def descriptor(self) -> ComponentDescriptor:
        return self._descriptor

    def collect(
        self,
        *,
        cue: Mapping[str, Any],
        frame_id: str,
        timestamp_ms: int,
        rendered_frame_path: Path,
        source_frame_path: Path,
    ) -> FontEvidenceObservation | None:
        """Return a protocol-compatible observation or fail closed.

        A non-Windows platform returns ``None`` before touching registry or
        font state.  Invalid caller input raises ``WindowsFontEvidenceError``;
        the enclosing subtitle evidence collector converts that into its
        structured ``font-evidence-invalid`` failure.
        """

        parsed = _validate_collect_input(
            cue=cue,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            rendered_frame_path=rendered_frame_path,
            source_frame_path=source_frame_path,
        )
        if self._platform_name.casefold() != "windows":
            return None
        if self._registry_reader is None and self._host_platform != "Windows":
            return None

        roots = self._resolve_allowed_roots()
        if not roots:
            return None

        rendered_snapshot = _read_stable_file(
            parsed["rendered_frame_path"],
            maximum=self._max_frame_bytes,
            allowed_roots=None,
            label="rendered frame",
            allowed_extensions=None,
        )
        source_snapshot = _read_stable_file(
            parsed["source_frame_path"],
            maximum=self._max_frame_bytes,
            allowed_roots=None,
            label="source frame",
            allowed_extensions=None,
        )
        entries = self._read_registry_entries()
        registered_faces = self._registered_faces(
            entries,
            roots=roots,
            requested_families=parsed["requested_families"],
        )
        selection = _select_face(
            registered_faces,
            requested_families=parsed["requested_families"],
            text=parsed["text"],
        )

        # Re-read the selected file from its registered canonical path.  This
        # detects stale cache entries, same-size mutations and most path-swap
        # races before a positive claim is emitted.
        if selection is not None:
            selection = self._revalidate_selection(
                selection,
                roots=roots,
                requested_families=parsed["requested_families"],
                text=parsed["text"],
            )

        evidence = _build_evidence_document(
            descriptor=self._descriptor,
            platform_name=self._platform_name,
            parsed=parsed,
            rendered_snapshot=rendered_snapshot,
            source_snapshot=source_snapshot,
            selection=selection,
            scanned_face_count=len(registered_faces),
            registry_entry_count=len(entries),
        )
        evidence_path = self._publish_evidence(
            evidence,
            rendered_frame_path=parsed["rendered_frame_path"],
        )

        if selection is None:
            missing = _missing_code_points(parsed["text"], frozenset())
            return FontEvidenceObservation(
                resolved_family=None,
                resolution_verified=False,
                glyph_coverage_verified=False,
                verification_method="not-provided",
                evidence_artifact_path=evidence_path,
                covered_renderable_code_points=0,
                missing_code_points=missing.missing_code_points,
                tofu_glyph_count=missing.missing_occurrences,
                installation=None,
                embedding=None,
            )

        installation = VerifiedFontClaim(
            status="verified-installed",
            verification_method="font-cmap-and-shaping",
            evidence_artifact_path=evidence_path,
            font_artifact_path=selection.registered_face.snapshot.path,
        )
        return FontEvidenceObservation(
            resolved_family=selection.resolved_family,
            resolution_verified=True,
            glyph_coverage_verified=selection.coverage.complete,
            verification_method="font-cmap-and-shaping",
            evidence_artifact_path=evidence_path,
            covered_renderable_code_points=selection.coverage.covered,
            missing_code_points=selection.coverage.missing_code_points,
            tofu_glyph_count=selection.coverage.missing_occurrences,
            installation=installation,
            embedding=None,
        )

    def _resolve_allowed_roots(self) -> tuple[Path, ...]:
        resolved: list[Path] = []
        seen: set[str] = set()
        for configured in self._configured_roots:
            try:
                root = configured.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if not root.is_dir() or _is_link_or_reparse(root):
                continue
            key = os.path.normcase(str(root))
            if key in seen:
                continue
            seen.add(key)
            resolved.append(root)
        return tuple(sorted(resolved, key=lambda item: os.path.normcase(str(item))))

    def _read_registry_entries(self) -> tuple[WindowsFontRegistryEntry, ...]:
        reader = self._registry_reader or _read_windows_font_registry
        try:
            raw_entries = reader()
        except WindowsFontEvidenceError:
            raise
        except Exception as exc:
            raise WindowsFontEvidenceError(
                "local Windows font registry enumeration failed"
            ) from exc

        entries: list[WindowsFontRegistryEntry] = []
        for index, raw in enumerate(raw_entries):
            if index >= self._max_registry_entries:
                raise WindowsFontEvidenceError(
                    "local font registry exceeds the configured entry limit"
                )
            entries.append(_validate_registry_entry(raw, index=index))
        entries.sort(key=_registry_entry_sort_key)
        return tuple(entries)

    def _registered_faces(
        self,
        entries: tuple[WindowsFontRegistryEntry, ...],
        *,
        roots: tuple[Path, ...],
        requested_families: tuple[str, ...],
    ) -> tuple[_RegisteredFace, ...]:
        bindings: list[tuple[WindowsFontRegistryEntry, Path]] = []
        seen_bindings: set[tuple[str, str, str, str, str]] = set()
        for entry in entries:
            # This is only a candidate-reduction step.  A positive claim still
            # requires an exact family match from the font's own name table
            # and registry-label/style validation after parsing.  Therefore a
            # forged registry label can make us inspect a file, but can never
            # make that file pass.
            if not any(
                _registry_label_can_reference_family(
                    entry.value_name,
                    requested_family,
                )
                for requested_family in requested_families
            ):
                continue
            path = _resolve_registered_font_path(entry, roots=roots)
            if path is None:
                continue
            key = (
                entry.hive.casefold(),
                entry.key_path.casefold(),
                entry.view.casefold(),
                entry.value_name.casefold(),
                os.path.normcase(str(path)),
            )
            if key in seen_bindings:
                continue
            seen_bindings.add(key)
            bindings.append((entry, path))

        fingerprints: list[dict[str, Any]] = []
        usable_bindings: list[tuple[WindowsFontRegistryEntry, Path]] = []
        for entry, path in bindings:
            try:
                fingerprint = _path_fingerprint(
                    path,
                    allowed_roots=roots,
                    label="registered font",
                    allowed_extensions=_FONT_EXTENSIONS,
                )
            except WindowsFontEvidenceError:
                continue
            fingerprints.append(
                {
                    "entry": _registry_entry_dict(entry),
                    "path": str(path),
                    "fingerprint": _fingerprint_dict(fingerprint),
                }
            )
            usable_bindings.append((entry, path))

        signature = _canonical_sha256(
            {
                "bindings": fingerprints,
                "fontToolsVersion": _fonttools_version(),
                "requestedFamilies": list(requested_families),
            }
        )
        if signature == self._index_signature:
            return self._index

        parsed_by_path: dict[str, tuple[_FontSnapshot, tuple[_FontFace, ...]]] = {}
        registered: list[_RegisteredFace] = []
        for entry, path in usable_bindings:
            path_key = os.path.normcase(str(path))
            parsed = parsed_by_path.get(path_key)
            if parsed is None:
                try:
                    snapshot = _read_stable_file(
                        path,
                        maximum=self._max_font_bytes,
                        allowed_roots=roots,
                        label="registered font",
                        allowed_extensions=_FONT_EXTENSIONS,
                    )
                    faces = _parse_font_faces(snapshot)
                except WindowsFontEvidenceError:
                    continue
                parsed = (snapshot, faces)
                parsed_by_path[path_key] = parsed
            snapshot, faces = parsed
            for face in faces:
                if not _registry_label_matches_face(entry.value_name, face):
                    continue
                registered.append(
                    _RegisteredFace(
                        entry=entry,
                        snapshot=snapshot,
                        face=face,
                    )
                )

        registered.sort(key=_registered_face_sort_key)
        self._index_signature = signature
        self._index = tuple(registered)
        return self._index

    def _revalidate_selection(
        self,
        selection: _Selection,
        *,
        roots: tuple[Path, ...],
        requested_families: tuple[str, ...],
        text: str,
    ) -> _Selection | None:
        path = selection.registered_face.snapshot.path
        fresh = _read_stable_file(
            path,
            maximum=self._max_font_bytes,
            allowed_roots=roots,
            label="selected registered font",
            allowed_extensions=_FONT_EXTENSIONS,
        )
        fresh_faces = _parse_font_faces(fresh)
        fresh_registered = tuple(
            _RegisteredFace(
                entry=selection.registered_face.entry,
                snapshot=fresh,
                face=face,
            )
            for face in fresh_faces
            if _registry_label_matches_face(
                selection.registered_face.entry.value_name,
                face,
            )
        )
        fresh_selection = _select_face(
            fresh_registered,
            requested_families=requested_families,
            text=text,
        )
        if fresh_selection is None:
            self._index_signature = None
            self._index = ()
            return None
        return fresh_selection

    def _publish_evidence(
        self,
        evidence: Mapping[str, Any],
        *,
        rendered_frame_path: Path,
    ) -> Path:
        payload = _canonical_json_bytes(evidence)
        if len(payload) > _MAX_EVIDENCE_BYTES:
            raise WindowsFontEvidenceError(
                "font evidence artifact exceeds the safety limit"
            )
        digest = hashlib.sha256(payload).hexdigest()
        if self._evidence_root is None:
            root = rendered_frame_path.parent / ".font-evidence"
            parent = rendered_frame_path.parent.resolve(strict=True)
        else:
            root = self._evidence_root
            try:
                parent = root.parent.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise WindowsFontEvidenceError(
                    "evidence root parent does not exist"
                ) from exc

        if root.exists():
            if not root.is_dir() or _is_link_or_reparse(root):
                raise WindowsFontEvidenceError(
                    "evidence root must be a regular local directory"
                )
        else:
            try:
                root.mkdir(mode=0o700)
            except OSError as exc:
                raise WindowsFontEvidenceError(
                    "could not create the local font evidence directory"
                ) from exc
        canonical_root = root.resolve(strict=True)
        if _is_link_or_reparse(canonical_root):
            raise WindowsFontEvidenceError(
                "evidence root must not be a link or reparse point"
            )
        if self._evidence_root is None and not _is_within(
            canonical_root, (parent,)
        ):
            raise WindowsFontEvidenceError(
                "default evidence root escaped the rendered frame directory"
            )

        target = canonical_root / f"font-evidence-{digest}.json"
        try:
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            existing = _read_stable_file(
                target,
                maximum=_MAX_EVIDENCE_BYTES,
                allowed_roots=(canonical_root,),
                label="existing font evidence",
                allowed_extensions=frozenset({".json"}),
            )
            if existing.payload != payload:
                raise WindowsFontEvidenceError(
                    "deterministic evidence path contains different bytes"
                )
        except OSError as exc:
            try:
                if target.exists() and target.stat().st_size == 0:
                    target.unlink()
            except OSError:
                pass
            raise WindowsFontEvidenceError(
                "could not publish the local font evidence artifact"
            ) from exc
        return target.resolve(strict=True)


def _validate_collect_input(
    *,
    cue: Mapping[str, Any],
    frame_id: str,
    timestamp_ms: int,
    rendered_frame_path: Path,
    source_frame_path: Path,
) -> dict[str, Any]:
    if not isinstance(cue, Mapping):
        raise WindowsFontEvidenceError("cue must be a mapping")
    text = _strict_text(
        cue.get("text"),
        label="cue.text",
        maximum=_MAX_TEXT_CHARACTERS,
        allow_newlines=True,
    )
    cue_id = _strict_text(
        cue.get("cueId"),
        label="cue.cueId",
        maximum=160,
    )
    style_id = _strict_text(
        cue.get("styleId"),
        label="cue.styleId",
        maximum=160,
    )
    families_raw = cue.get("requestedFontFamilies")
    if (
        not isinstance(families_raw, Sequence)
        or isinstance(families_raw, (str, bytes, bytearray))
        or not 1 <= len(families_raw) <= 64
    ):
        raise WindowsFontEvidenceError(
            "cue.requestedFontFamilies must contain 1..64 values"
        )
    requested_families: list[str] = []
    normalized_families: set[str] = set()
    for index, value in enumerate(families_raw):
        family = _strict_text(
            value,
            label=f"cue.requestedFontFamilies[{index}]",
            maximum=_MAX_FAMILY_CHARACTERS,
        )
        if (
            "/" in family
            or "\\" in family
            or ":" in family
            or _URL_SCHEME.match(family)
        ):
            raise WindowsFontEvidenceError(
                "requested font families must be names, not paths or URLs"
            )
        normalized = _normalize_name(family)
        if normalized in normalized_families:
            raise WindowsFontEvidenceError(
                "requested font families must be unique after normalization"
            )
        normalized_families.add(normalized)
        requested_families.append(family)

    frame_id_value = _strict_text(
        frame_id,
        label="frame_id",
        maximum=160,
    )
    if (
        isinstance(timestamp_ms, bool)
        or not isinstance(timestamp_ms, int)
        or not 0 <= timestamp_ms <= 604_800_000
    ):
        raise WindowsFontEvidenceError(
            "timestamp_ms must be an integer in the supported media range"
        )
    rendered = _coerce_path(
        rendered_frame_path,
        label="rendered_frame_path",
    )
    source = _coerce_path(source_frame_path, label="source_frame_path")
    if _same_path(rendered, source):
        raise WindowsFontEvidenceError(
            "rendered and source frame paths must be distinct"
        )
    return {
        "cue_id": cue_id,
        "style_id": style_id,
        "text": text,
        "requested_families": tuple(requested_families),
        "frame_id": frame_id_value,
        "timestamp_ms": timestamp_ms,
        "rendered_frame_path": rendered,
        "source_frame_path": source,
    }


def _read_windows_font_registry() -> tuple[WindowsFontRegistryEntry, ...]:
    if platform.system() != "Windows":
        return ()
    try:
        import winreg
    except ImportError:
        return ()

    hives = (
        ("HKLM", winreg.HKEY_LOCAL_MACHINE),
        ("HKCU", winreg.HKEY_CURRENT_USER),
    )
    view_specs = (
        ("registry64", getattr(winreg, "KEY_WOW64_64KEY", 0)),
        ("registry32", getattr(winreg, "KEY_WOW64_32KEY", 0)),
    )
    entries: list[WindowsFontRegistryEntry] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for hive_name, hive in hives:
        for key_path in _REGISTRY_KEYS:
            for view_name, view_flag in view_specs:
                try:
                    key = winreg.OpenKey(
                        hive,
                        key_path,
                        0,
                        winreg.KEY_READ | view_flag,
                    )
                except OSError:
                    continue
                try:
                    index = 0
                    while True:
                        try:
                            name, data, value_type = winreg.EnumValue(
                                key, index
                            )
                        except OSError:
                            break
                        index += 1
                        if value_type not in (
                            winreg.REG_SZ,
                            winreg.REG_EXPAND_SZ,
                        ):
                            continue
                        if not isinstance(name, str) or not isinstance(
                            data, str
                        ):
                            continue
                        type_name = (
                            "REG_EXPAND_SZ"
                            if value_type == winreg.REG_EXPAND_SZ
                            else "REG_SZ"
                        )
                        identity = (
                            hive_name,
                            key_path,
                            name,
                            data,
                            type_name,
                        )
                        if identity in seen:
                            continue
                        seen.add(identity)
                        entries.append(
                            WindowsFontRegistryEntry(
                                hive=hive_name,
                                key_path=key_path,
                                view=view_name,
                                value_name=name,
                                value_data=data,
                                value_type=type_name,
                            )
                        )
                finally:
                    winreg.CloseKey(key)
    entries.sort(key=_registry_entry_sort_key)
    return tuple(entries)


def _default_windows_font_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    windows = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if windows:
        roots.append(Path(windows) / "Fonts")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    return tuple(roots)


def _validate_registry_entry(
    value: Any,
    *,
    index: int,
) -> WindowsFontRegistryEntry:
    if not isinstance(value, WindowsFontRegistryEntry):
        raise WindowsFontEvidenceError(
            f"registry entry {index} has an invalid type"
        )
    hive = _strict_text(
        value.hive,
        label=f"registry[{index}].hive",
        maximum=16,
    ).upper()
    if hive not in {"HKLM", "HKCU"}:
        raise WindowsFontEvidenceError(
            f"registry[{index}].hive must be HKLM or HKCU"
        )
    key_path = _strict_text(
        value.key_path,
        label=f"registry[{index}].key_path",
        maximum=512,
    )
    if key_path not in _REGISTRY_KEYS:
        raise WindowsFontEvidenceError(
            f"registry[{index}].key_path is not an approved font key"
        )
    view = _strict_text(
        value.view,
        label=f"registry[{index}].view",
        maximum=32,
    )
    if view not in {"registry32", "registry64", "default"}:
        raise WindowsFontEvidenceError(
            f"registry[{index}].view is invalid"
        )
    name = _strict_text(
        value.value_name,
        label=f"registry[{index}].value_name",
        maximum=1_024,
    )
    data = _strict_text(
        value.value_data,
        label=f"registry[{index}].value_data",
        maximum=_MAX_REGISTRY_TEXT,
    )
    value_type = _strict_text(
        value.value_type,
        label=f"registry[{index}].value_type",
        maximum=32,
    )
    if value_type not in {"REG_SZ", "REG_EXPAND_SZ"}:
        raise WindowsFontEvidenceError(
            f"registry[{index}].value_type is invalid"
        )
    return WindowsFontRegistryEntry(
        hive=hive,
        key_path=key_path,
        view=view,
        value_name=name,
        value_data=data,
        value_type=value_type,
    )


def _resolve_registered_font_path(
    entry: WindowsFontRegistryEntry,
    *,
    roots: tuple[Path, ...],
) -> Path | None:
    value = entry.value_data
    if entry.value_type == "REG_EXPAND_SZ":
        try:
            value = _expand_windows_environment(value)
        except WindowsFontEvidenceError:
            return None
    if (
        not value
        or value != value.strip()
        or "\x00" in value
        or _URL_SCHEME.match(value)
        or value.startswith(('"', "'"))
        or value.endswith(('"', "'"))
    ):
        return None

    raw = Path(value)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        # A relative registry font value must be a bare file name.  Directory
        # traversal, drive-relative syntax and alternate data streams are
        # never interpreted.
        if (
            raw.name != value
            or "/" in value
            or "\\" in value
            or ":" in value
            or value in {".", ".."}
        ):
            return None
        candidates.extend(root / value for root in roots)

    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            canonical = candidate.resolve(strict=True)
            _path_fingerprint(
                canonical,
                allowed_roots=roots,
                label="registered font",
                allowed_extensions=_FONT_EXTENSIONS,
            )
        except WindowsFontEvidenceError:
            continue
        except (OSError, RuntimeError):
            continue
        key = os.path.normcase(str(canonical))
        if key not in seen:
            seen.add(key)
            resolved.append(canonical)
    if len(resolved) != 1:
        return None
    return resolved[0]


def _read_stable_file(
    path: Path,
    *,
    maximum: int,
    allowed_roots: tuple[Path, ...] | None,
    label: str,
    allowed_extensions: frozenset[str] | None,
) -> _FontSnapshot:
    canonical = _canonical_regular_file(
        path,
        allowed_roots=allowed_roots,
        label=label,
        allowed_extensions=allowed_extensions,
    )
    try:
        with canonical.open("rb") as handle:
            before = _fingerprint_from_stat(os.fstat(handle.fileno()))
            if before.size <= 0:
                raise WindowsFontEvidenceError(f"{label} must not be empty")
            if before.size > maximum:
                raise WindowsFontEvidenceError(
                    f"{label} exceeds the configured byte limit"
                )
            payload = handle.read(maximum + 1)
            after = _fingerprint_from_stat(os.fstat(handle.fileno()))
    except WindowsFontEvidenceError:
        raise
    except OSError as exc:
        raise WindowsFontEvidenceError(f"could not read {label}") from exc
    if len(payload) > maximum or len(payload) != before.size:
        raise WindowsFontEvidenceError(
            f"{label} changed or exceeded the configured byte limit"
        )
    if before != after:
        raise WindowsFontEvidenceError(f"{label} changed while being read")
    current = _path_fingerprint(
        canonical,
        allowed_roots=allowed_roots,
        label=label,
        allowed_extensions=allowed_extensions,
    )
    if current != before:
        raise WindowsFontEvidenceError(
            f"{label} path changed while evidence was collected"
        )
    return _FontSnapshot(
        path=canonical,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        fingerprint=before,
    )


def _canonical_regular_file(
    path: Path,
    *,
    allowed_roots: tuple[Path, ...] | None,
    label: str,
    allowed_extensions: frozenset[str] | None,
) -> Path:
    try:
        if _is_link_or_reparse(path):
            raise WindowsFontEvidenceError(
                f"{label} must not be a link or reparse point"
            )
        canonical = path.resolve(strict=True)
    except WindowsFontEvidenceError:
        raise
    except (OSError, RuntimeError) as exc:
        raise WindowsFontEvidenceError(f"{label} does not exist") from exc
    if _is_link_or_reparse(canonical):
        raise WindowsFontEvidenceError(
            f"{label} must not be a link or reparse point"
        )
    if allowed_roots is not None and not _is_within(
        canonical, allowed_roots
    ):
        raise WindowsFontEvidenceError(
            f"{label} escaped the trusted local font roots"
        )
    try:
        mode = canonical.stat().st_mode
    except OSError as exc:
        raise WindowsFontEvidenceError(f"could not stat {label}") from exc
    if not stat.S_ISREG(mode):
        raise WindowsFontEvidenceError(f"{label} must be a regular file")
    if (
        allowed_extensions is not None
        and canonical.suffix.casefold() not in allowed_extensions
    ):
        raise WindowsFontEvidenceError(
            f"{label} does not use a supported local artifact extension"
        )
    return canonical


def _path_fingerprint(
    path: Path,
    *,
    allowed_roots: tuple[Path, ...] | None,
    label: str,
    allowed_extensions: frozenset[str] | None,
) -> _FileFingerprint:
    canonical = _canonical_regular_file(
        path,
        allowed_roots=allowed_roots,
        label=label,
        allowed_extensions=allowed_extensions,
    )
    try:
        return _fingerprint_from_stat(canonical.stat())
    except OSError as exc:
        raise WindowsFontEvidenceError(f"could not stat {label}") from exc


def _parse_font_faces(snapshot: _FontSnapshot) -> tuple[_FontFace, ...]:
    try:
        from fontTools.ttLib import TTCollection, TTFont, TTLibError
    except ImportError as exc:
        raise WindowsFontEvidenceError(
            "fontTools is required for local font parsing"
        ) from exc

    fonts: list[Any] = []
    collection: Any | None = None
    try:
        if snapshot.path.suffix.casefold() in {".ttc", ".otc"}:
            collection = TTCollection(io.BytesIO(snapshot.payload), lazy=False)
            fonts = list(collection.fonts)
        else:
            fonts = [
                TTFont(
                    io.BytesIO(snapshot.payload),
                    lazy=False,
                    recalcBBoxes=False,
                    recalcTimestamp=False,
                )
            ]
        faces = tuple(
            _parse_font_face(font, face_index=index)
            for index, font in enumerate(fonts)
        )
    except (TTLibError, KeyError, ValueError, OverflowError, OSError) as exc:
        raise WindowsFontEvidenceError(
            f"registered font is not a valid supported OpenType font: {snapshot.path.name}"
        ) from exc
    finally:
        if collection is not None:
            try:
                collection.close()
            except Exception:
                pass
        else:
            for font in fonts:
                try:
                    font.close()
                except Exception:
                    pass
    valid = tuple(face for face in faces if face.family_names)
    if not valid:
        raise WindowsFontEvidenceError(
            "registered font has no trustworthy family metadata"
        )
    return valid


def _parse_font_face(font: Any, *, face_index: int) -> _FontFace:
    name_table = font["name"]
    typographic_families = _font_names(name_table, {16})
    legacy_families = _font_names(name_table, {1})
    family_names = _ordered_unique_names(
        typographic_families + legacy_families
    )
    style_names = _ordered_unique_names(
        _font_names(name_table, {17}) + _font_names(name_table, {2})
    )
    full_names = _ordered_unique_names(_font_names(name_table, {4}))
    postscript_names = _ordered_unique_names(_font_names(name_table, {6}))

    cmap: dict[int, str] = {}
    cmap_table = font["cmap"]
    for table in cmap_table.tables:
        try:
            is_unicode = bool(table.isUnicode())
        except Exception:
            is_unicode = False
        if not is_unicode or not isinstance(getattr(table, "cmap", None), dict):
            continue
        for code_point, glyph_name in table.cmap.items():
            if (
                not isinstance(code_point, int)
                or not _is_unicode_scalar(code_point)
                or not isinstance(glyph_name, str)
                or code_point in cmap
            ):
                continue
            try:
                glyph_id = font.getGlyphID(glyph_name)
            except (KeyError, ValueError):
                continue
            if glyph_id != 0 and glyph_name != ".notdef":
                cmap[code_point] = glyph_name
    coverage = frozenset(cmap)
    cmap_sha256 = _canonical_sha256(
        [
            [f"U+{code_point:04X}", cmap[code_point]]
            for code_point in sorted(cmap)
        ]
    )
    return _FontFace(
        face_index=face_index,
        family_names=family_names,
        style_names=style_names,
        full_names=full_names,
        postscript_names=postscript_names,
        covered_code_points=coverage,
        cmap_sha256=cmap_sha256,
    )


def _font_names(name_table: Any, name_ids: set[int]) -> list[str]:
    values: list[tuple[int, int, int, str]] = []
    for record in getattr(name_table, "names", ()):
        if getattr(record, "nameID", None) not in name_ids:
            continue
        try:
            value = record.toUnicode(errors="strict")
        except (UnicodeDecodeError, LookupError, TypeError, ValueError):
            continue
        if not isinstance(value, str):
            continue
        value = unicodedata.normalize("NFC", value).strip()
        if (
            not value
            or len(value) > _MAX_FAMILY_CHARACTERS
            or "\x00" in value
            or any(_is_forbidden_control(character) for character in value)
        ):
            continue
        language_id = int(getattr(record, "langID", 0))
        platform_id = int(getattr(record, "platformID", 0))
        encoding_id = int(getattr(record, "platEncID", 0))
        values.append((language_id, platform_id, encoding_id, value))
    values.sort(
        key=lambda item: (
            0 if item[0] in {0x0409, 0} else 1,
            item[1],
            item[2],
            _normalize_name(item[3]),
            item[3],
        )
    )
    return [item[3] for item in values]


def _ordered_unique_names(values: Iterable[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_name(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(value)
    return tuple(output)


def _registry_label_matches_face(label: str, face: _FontFace) -> bool:
    base = _REGISTRY_FONT_SUFFIX.sub("", label).strip()
    normalized_label = _normalize_name(base)
    if not normalized_label:
        return False
    metadata_names = (
        face.family_names + face.full_names + face.postscript_names
    )
    normalized_names = tuple(_normalize_name(name) for name in metadata_names)
    if not any(
        name == normalized_label
        or _name_has_bounded_component(normalized_label, name)
        for name in normalized_names
    ):
        return False

    label_style_words = _style_words(base)
    if not label_style_words:
        return True
    metadata_style_words = _style_words(
        " ".join(face.style_names + face.full_names)
    )
    return label_style_words <= metadata_style_words


def _registry_label_can_reference_family(
    label: str,
    requested_family: str,
) -> bool:
    base = _REGISTRY_FONT_SUFFIX.sub("", label).strip()
    normalized_label = _normalize_name(base)
    normalized_family = _normalize_name(requested_family)
    return (
        normalized_label == normalized_family
        or _name_has_bounded_component(
            normalized_label,
            normalized_family,
        )
    )


def _select_face(
    registered_faces: Sequence[_RegisteredFace],
    *,
    requested_families: tuple[str, ...],
    text: str,
) -> _Selection | None:
    best_partial: _Selection | None = None
    best_partial_key: tuple[int, int] | None = None
    for requested_index, requested in enumerate(requested_families):
        normalized = _normalize_name(requested)
        matches: list[tuple[_RegisteredFace, str, _Coverage]] = []
        for registered in registered_faces:
            matching_family = next(
                (
                    family
                    for family in registered.face.family_names
                    if _normalize_name(family) == normalized
                ),
                None,
            )
            if matching_family is None:
                continue
            coverage = _missing_code_points(
                text,
                registered.face.covered_code_points,
            )
            matches.append((registered, matching_family, coverage))
        matches.sort(
            key=lambda item: (
                0 if item[2].complete else 1,
                -item[2].covered,
                _style_rank(item[0].face),
                _registered_face_sort_key(item[0]),
            )
        )
        if not matches:
            continue
        registered, family, coverage = matches[0]
        selection = _Selection(
            registered_face=registered,
            requested_family=requested,
            resolved_family=family,
            coverage=coverage,
        )
        if coverage.complete:
            return selection
        candidate_key = (coverage.covered, -requested_index)
        if best_partial_key is None or candidate_key > best_partial_key:
            best_partial = selection
            best_partial_key = candidate_key
    return best_partial


def _missing_code_points(
    text: str,
    covered_code_points: frozenset[int],
) -> _Coverage:
    renderable = [
        ord(character) for character in text if not character.isspace()
    ]
    missing_occurrences = [
        code_point
        for code_point in renderable
        if code_point not in covered_code_points
    ]
    missing_unique = tuple(
        _format_code_point(code_point)
        for code_point in sorted(set(missing_occurrences))
    )
    requested_unique = tuple(
        _format_code_point(code_point)
        for code_point in sorted(set(renderable))
    )
    return _Coverage(
        expected=len(renderable),
        covered=len(renderable) - len(missing_occurrences),
        missing_code_points=missing_unique,
        missing_occurrences=len(missing_occurrences),
        requested_code_points=requested_unique,
    )


def _build_evidence_document(
    *,
    descriptor: ComponentDescriptor,
    platform_name: str,
    parsed: Mapping[str, Any],
    rendered_snapshot: _FontSnapshot,
    source_snapshot: _FontSnapshot,
    selection: _Selection | None,
    scanned_face_count: int,
    registry_entry_count: int,
) -> dict[str, Any]:
    if selection is None:
        coverage = _missing_code_points(parsed["text"], frozenset())
        resolution: dict[str, Any] = {
            "verified": False,
            "requestedFamily": None,
            "resolvedFamily": None,
            "familyAliases": [],
            "style": None,
            "styleAliases": [],
            "fullNames": [],
            "postscriptNames": [],
            "faceIndex": None,
        }
        installation: dict[str, Any] | None = None
    else:
        registered = selection.registered_face
        face = registered.face
        coverage = selection.coverage
        resolution = {
            "verified": True,
            "requestedFamily": selection.requested_family,
            "resolvedFamily": selection.resolved_family,
            "familyAliases": list(face.family_names),
            "style": face.primary_style,
            "styleAliases": list(face.style_names),
            "fullNames": list(face.full_names),
            "postscriptNames": list(face.postscript_names),
            "faceIndex": face.face_index,
        }
        installation = {
            "verified": True,
            "registry": _registry_entry_dict(registered.entry),
            "registryLabelMatchesMetadata": True,
            "fontArtifactPath": str(registered.snapshot.path),
            "fontArtifactSha256": registered.snapshot.sha256,
            "fontArtifactBytes": registered.snapshot.fingerprint.size,
            "fontFileFingerprint": _fingerprint_dict(
                registered.snapshot.fingerprint
            ),
            "fontCmapSha256": face.cmap_sha256,
        }
    return {
        "kind": "windows-font-evidence",
        "schemaVersion": WINDOWS_FONT_EVIDENCE_SCHEMA_VERSION,
        "provider": {
            "name": descriptor.name,
            "version": descriptor.version,
            "configurationSha256": descriptor.configuration_sha256,
        },
        "platform": platform_name,
        "cue": {
            "cueId": parsed["cue_id"],
            "styleId": parsed["style_id"],
            "textSha256": hashlib.sha256(
                parsed["text"].encode("utf-8")
            ).hexdigest(),
            "requestedFamilies": list(parsed["requested_families"]),
            "expectedRenderableCodePoints": coverage.expected,
        },
        "frameBinding": {
            "frameId": parsed["frame_id"],
            "timestampMs": parsed["timestamp_ms"],
            "renderedFramePath": str(rendered_snapshot.path),
            "renderedFrameSha256": rendered_snapshot.sha256,
            "sourceFramePath": str(source_snapshot.path),
            "sourceFrameSha256": source_snapshot.sha256,
        },
        "localDiscovery": {
            "registryEntryCount": registry_entry_count,
            "validatedRegisteredFaceCount": scanned_face_count,
            "networkUsed": False,
        },
        "resolution": resolution,
        "installation": installation,
        "glyphCoverage": {
            "verified": bool(selection is not None and coverage.complete),
            "method": "unicode-cmap-non-notdef-per-codepoint-v1",
            "expectedRenderableCodePoints": coverage.expected,
            "coveredRenderableCodePoints": coverage.covered,
            "missingCodePoints": list(coverage.missing_code_points),
            "missingOccurrences": coverage.missing_occurrences,
            "requestedUniqueCodePoints": list(
                coverage.requested_code_points
            ),
        },
    }


def _style_rank(face: _FontFace) -> tuple[int, str]:
    style = " ".join(face.style_names)
    words = _style_words(style)
    return (
        0 if words & _REGULAR_STYLE_WORDS else 1,
        _normalize_name(style),
    )


def _registered_face_sort_key(
    registered: _RegisteredFace,
) -> tuple[str, int, str, str, str, str]:
    return (
        os.path.normcase(str(registered.snapshot.path)),
        registered.face.face_index,
        registered.entry.hive.casefold(),
        registered.entry.key_path.casefold(),
        registered.entry.view.casefold(),
        registered.entry.value_name.casefold(),
    )


def _registry_entry_sort_key(
    entry: WindowsFontRegistryEntry,
) -> tuple[str, str, str, str, str, str]:
    return (
        entry.hive.casefold(),
        entry.key_path.casefold(),
        entry.view.casefold(),
        entry.value_name.casefold(),
        entry.value_data.casefold(),
        entry.value_type.casefold(),
    )


def _registry_entry_dict(
    entry: WindowsFontRegistryEntry,
) -> dict[str, str]:
    return {
        "hive": entry.hive,
        "keyPath": entry.key_path,
        "view": entry.view,
        "valueName": entry.value_name,
        "valueData": entry.value_data,
        "valueType": entry.value_type,
    }


def _fingerprint_from_stat(value: os.stat_result) -> _FileFingerprint:
    return _FileFingerprint(
        size=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
        device=int(value.st_dev),
        inode=int(value.st_ino),
    )


def _fingerprint_dict(value: _FileFingerprint) -> dict[str, int]:
    return {
        "size": value.size,
        "mtimeNs": value.mtime_ns,
        "device": value.device,
        "inode": value.inode,
    }


def _expand_windows_environment(value: str) -> str:
    environment = {key.casefold(): item for key, item in os.environ.items()}

    def replacement(match: re.Match[str]) -> str:
        name = match.group(1)
        replacement_value = environment.get(name.casefold())
        if replacement_value is None:
            raise WindowsFontEvidenceError(
                f"unknown registry environment variable: %{name}%"
            )
        return replacement_value

    expanded = _WINDOWS_ENVIRONMENT_VARIABLE.sub(replacement, value)
    if "%" in expanded:
        raise WindowsFontEvidenceError(
            "registry environment value contains an unmatched percent sign"
        )
    return expanded


def _fonttools_version() -> str:
    try:
        return metadata.version("fonttools")
    except metadata.PackageNotFoundError:
        return "unavailable"


def _strict_text(
    value: Any,
    *,
    label: str,
    maximum: int,
    allow_newlines: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        raise WindowsFontEvidenceError(
            f"{label} must be trimmed non-empty text"
        )
    for character in value:
        code_point = ord(character)
        if not _is_unicode_scalar(code_point):
            raise WindowsFontEvidenceError(
                f"{label} contains a non-scalar Unicode value"
            )
        if _is_forbidden_control(character):
            if allow_newlines and character in {"\n", "\r", "\t"}:
                continue
            raise WindowsFontEvidenceError(
                f"{label} contains a forbidden control character"
            )
    return value


def _coerce_path(value: str | Path, *, label: str) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        if (
            not value
            or value != value.strip()
            or "\x00" in value
            or _URL_SCHEME.match(value)
        ):
            raise WindowsFontEvidenceError(
                f"{label} must be a local filesystem path"
            )
        path = Path(value)
    else:
        raise WindowsFontEvidenceError(
            f"{label} must be a local filesystem path"
        )
    if not path.is_absolute():
        raise WindowsFontEvidenceError(f"{label} must be absolute")
    return path


def _bounded_positive_integer(
    value: Any,
    *,
    label: str,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise WindowsFontEvidenceError(
            f"{label} must be an integer between 1 and {maximum}"
        )
    return value


def _is_link_or_reparse(path: Path) -> bool:
    try:
        stat_result = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(stat_result.st_mode):
        return True
    attributes = getattr(stat_result, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _is_within(path: Path, roots: Sequence[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=True) == right.resolve(strict=True)
    except (OSError, RuntimeError):
        return os.path.normcase(str(left.absolute())) == os.path.normcase(
            str(right.absolute())
        )


def _normalize_name(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", value).casefold().split()
    )


def _name_has_bounded_component(container: str, component: str) -> bool:
    if not component:
        return False
    escaped = re.escape(component)
    return bool(
        re.search(
            rf"(?<![\w]){escaped}(?![\w])",
            container,
            flags=re.UNICODE,
        )
    )


def _style_words(value: str) -> frozenset[str]:
    words = re.findall(
        r"[0-9A-Za-z]+",
        unicodedata.normalize("NFKC", value).casefold(),
    )
    combined: set[str] = set(words)
    for first, second in zip(words, words[1:]):
        combined.add(first + second)
    return frozenset(word for word in combined if word in _STYLE_WORDS)


def _is_forbidden_control(character: str) -> bool:
    return unicodedata.category(character) in {"Cc", "Cs"}


def _is_unicode_scalar(code_point: int) -> bool:
    return (
        0 <= code_point <= 0x10FFFF
        and not 0xD800 <= code_point <= 0xDFFF
    )


def _format_code_point(code_point: int) -> str:
    return f"U+{code_point:04X}"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    digest = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    if not _SHA256.fullmatch(digest):
        raise AssertionError("internal SHA-256 invariant failed")
    return digest


__all__ = [
    "WINDOWS_FONT_EVIDENCE_SCHEMA_VERSION",
    "WindowsFontEvidenceError",
    "WindowsFontEvidenceProvider",
    "WindowsFontRegistryEntry",
]
