"""Executable, repository-local Design Pack quality policy.

This module is intentionally independent from the external Design Pack runtime.
The upstream V3 materializer/evaluator contract is currently blocked, so this
gate never represents itself as an external Design Pack passage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .css import (
    CssDeclaration,
    CssRule,
    context_contains,
    iter_declarations,
    parse_css,
    selector_parts,
    split_top_level,
)
from .evidence import (
    EvidenceValidationError,
    validate_ocr_document,
    validate_screenshot_document,
)


SCHEMA_VERSION = "1.0.0"
MAX_UI_MOTION_MS = 300.0
EXPECTED_LOCALES = (
    "en",
    "zh-Hans",
    "zh-Hant",
    "ja",
    "ko",
    "es",
    "fr",
    "de",
    "pt-BR",
)
ALLOWED_TRANSITION_PROPERTIES = frozenset(
    {
        "transform",
        "opacity",
        "color",
        "background-color",
        "border-color",
        "box-shadow",
    }
)
ALLOWED_KEYFRAME_PROPERTIES = frozenset({"transform", "opacity"})
REQUIRED_GLASS_TOKENS = frozenset(
    {
        "--surface-glass",
        "--surface-strong",
        "--surface-soft",
        "--surface-control",
        "--glass-border",
        "--glass-highlight",
        "--background-overlay",
    }
)
REQUIRED_PALETTE_TOKENS = frozenset(
    {
        "--violet-700",
        "--violet-600",
        "--violet-100",
        "--rose-700",
        "--rose-600",
        "--rose-100",
        "--mint-700",
        "--mint-600",
        "--mint-100",
    }
)
CANONICAL_EASE_OUT = "cubic-bezier(0.23, 1, 0.32, 1)"
CANONICAL_EASE_IN_OUT = "cubic-bezier(0.77, 0, 0.175, 1)"
CANONICAL_EASE_DRAWER = "cubic-bezier(0.32, 0.72, 0, 1)"
CANONICAL_EASINGS = {
    "--ease-out": CANONICAL_EASE_OUT,
    "--ease-in-out": CANONICAL_EASE_IN_OUT,
    "--ease-drawer": CANONICAL_EASE_DRAWER,
}
REQUIRED_IMPORTED_CSS = frozenset(
    {
        "apps/desktop/src/styles/tokens.css",
        "apps/desktop/src/styles/global.css",
        "apps/desktop/src/components/SceneBackdrop.css",
        "apps/desktop/src/components/DesktopCompanion.css",
    }
)
COMPANION_SHA256 = (
    "84555A0B2AD4B96C0282C50B5A3FD92D6AB3933E0FA72935E3FD59322389C4EB"
)
TIME_PATTERN = re.compile(r"(?<![\w.-])(\d+(?:\.\d+)?|\.\d+)(ms|s)\b", re.I)
MOTION_PSEUDO_PATTERN = re.compile(
    r":(?:hover|active|focus|focus-visible|checked|open)\b|\[open\]|\[data-state",
    re.I,
)
PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z][A-Za-z0-9]*)\}")
TS_CSS_IMPORT_PATTERN = re.compile(
    r"""^\s*import\s+(?:(?:[^"'();\n]+?)\s+from\s+)?"""
    r"""["'](?P<path>[^"']+\.css(?:[?#][^"']*)?)["']\s*;?""",
    re.M,
)
CSS_IMPORT_PATTERN = re.compile(
    r"""^\s*@import\s+(?:url\(\s*)?["']"""
    r"""(?P<path>[^"']+\.css(?:[?#][^"']*)?)["']\s*\)?[^;]*;""",
    re.M | re.I,
)
UPSTREAM_DESIGN_PACK = {
    "passageClaimed": False,
    "status": "blocked-upstream-v3-contract",
    "blocker": "repairCatalog.passPolicy is required",
    "detail": (
        "The external Design Pack canonical prepare path emits a V3 context "
        "whose materializer output does not match the evaluator and JSON "
        "Schema contracts. This local gate is not a substitute for, and does "
        "not claim, an external Design Pack passage."
    ),
}


@dataclass(frozen=True)
class CssSource:
    path: Path
    source: str
    imported_by: tuple[Path, ...]


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    path: str
    line: int | None = None
    severity: str = "error"
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.line is None:
            value.pop("line")
        if not self.evidence:
            value.pop("evidence")
        return value


@dataclass(frozen=True)
class CheckResult:
    id: str
    status: str
    findings: tuple[Finding, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "findings": [finding.as_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class AuditReport:
    project_root: str
    checks: tuple[CheckResult, ...]

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(
            finding
            for check in self.checks
            for finding in check.findings
        )

    @property
    def status(self) -> str:
        return "pass" if not self.findings else "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "media-transcribe-studio/design-quality-report",
            "status": self.status,
            "projectRoot": self.project_root,
            "policy": {
                "failClosed": True,
                "maximumUiMotionMs": MAX_UI_MOTION_MS,
                "allowedTransitionProperties": sorted(
                    ALLOWED_TRANSITION_PROPERTIES
                ),
                "allowedKeyframeProperties": sorted(ALLOWED_KEYFRAME_PROPERTIES),
            },
            "externalDesignPack": dict(UPSTREAM_DESIGN_PACK),
            "checks": [check.as_dict() for check in self.checks],
            "summary": {
                "checks": len(self.checks),
                "passed": sum(check.status == "pass" for check in self.checks),
                "failed": sum(check.status == "fail" for check in self.checks),
                "errors": len(self.findings),
            },
        }


def _relative(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _read_required(project_root: Path, relative: str) -> tuple[Path, str]:
    path = project_root / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"required source is missing: {relative}: {exc}") from exc
    if not resolved.is_file():
        raise ValueError(f"required source is not a file: {relative}")
    try:
        content = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"required source is not readable UTF-8: {relative}: {exc}") from exc
    if not content.strip():
        raise ValueError(f"required source is empty: {relative}")
    return resolved, content


def _finding(
    *,
    code: str,
    message: str,
    project_root: Path,
    path: Path,
    line: int | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> Finding:
    return Finding(
        code=code,
        message=message,
        path=_relative(project_root, path),
        line=line,
        evidence=evidence or {},
    )


def _strip_import_suffix(specifier: str) -> str:
    return re.split(r"[?#]", specifier, maxsplit=1)[0]


def _resolve_css_import(
    *,
    project_root: Path,
    source_root: Path,
    importer: Path,
    specifier: str,
) -> tuple[Path | None, Finding | None]:
    import_path = _strip_import_suffix(specifier).replace("\\", "/")
    if not import_path.startswith(("./", "../")):
        return None, _finding(
            code="DQ-CSS-IMPORT-UNSUPPORTED",
            message=(
                "desktop CSS imports must be repository-local relative paths; "
                f"found {specifier!r}"
            ),
            project_root=project_root,
            path=importer,
        )
    normalized = Path(*import_path.split("/"))
    candidate = (importer.parent / normalized).resolve()
    try:
        candidate.relative_to(source_root.resolve())
    except ValueError:
        return None, _finding(
            code="DQ-CSS-IMPORT-BOUNDARY",
            message=f"CSS import escapes apps/desktop/src: {specifier!r}",
            project_root=project_root,
            path=importer,
        )
    if candidate.suffix.casefold() != ".css":
        return None, _finding(
            code="DQ-CSS-IMPORT-UNSUPPORTED",
            message=f"CSS import does not resolve to a .css file: {specifier!r}",
            project_root=project_root,
            path=importer,
        )
    if not candidate.is_file():
        return None, _finding(
            code="DQ-CSS-IMPORT-MISSING",
            message=f"imported CSS file is missing: {specifier!r}",
            project_root=project_root,
            path=importer,
        )
    return candidate, None


def _discover_imported_css(
    project_root: Path,
) -> tuple[tuple[CssSource, ...], tuple[Finding, ...]]:
    """Find every repository-local CSS file imported below desktop/src.

    Component CSS is intentionally discovered from every TypeScript/JavaScript
    module, not only modules reachable from ``main.tsx``. This keeps dormant
    but product-bound components such as DesktopCompanion under the same gate
    before their final App integration lands.
    """

    source_root = project_root / "apps/desktop/src"
    findings: list[Finding] = []
    importers: dict[Path, set[Path]] = {}
    queue: list[Path] = []

    try:
        modules = tuple(
            sorted(
                (
                    path
                    for path in source_root.rglob("*")
                    if path.is_file()
                    and path.suffix.casefold() in {".ts", ".tsx", ".js", ".jsx"}
                ),
                key=lambda path: path.as_posix().casefold(),
            )
        )
    except OSError as exc:
        return (), (
            _finding(
                code="DQ-CSS-SOURCE-GRAPH",
                message=f"desktop source tree could not be enumerated: {exc}",
                project_root=project_root,
                path=source_root,
            ),
        )

    for module in modules:
        try:
            source = module.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            findings.append(
                _finding(
                    code="DQ-CSS-SOURCE-READ",
                    message=f"source module is not readable UTF-8: {exc}",
                    project_root=project_root,
                    path=module,
                )
            )
            continue
        for match in TS_CSS_IMPORT_PATTERN.finditer(source):
            resolved, finding = _resolve_css_import(
                project_root=project_root,
                source_root=source_root,
                importer=module,
                specifier=match.group("path"),
            )
            if finding is not None:
                findings.append(finding)
                continue
            assert resolved is not None
            if resolved not in importers:
                queue.append(resolved)
                importers[resolved] = set()
            importers[resolved].add(module)

    css_sources: dict[Path, str] = {}
    while queue:
        path = queue.pop(0)
        if path in css_sources:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            findings.append(
                _finding(
                    code="DQ-CSS-IMPORT-READ",
                    message=f"imported CSS is not readable UTF-8: {exc}",
                    project_root=project_root,
                    path=path,
                )
            )
            continue
        if not source.strip():
            findings.append(
                _finding(
                    code="DQ-CSS-IMPORT-EMPTY",
                    message="imported CSS must not be empty",
                    project_root=project_root,
                    path=path,
                )
            )
            continue
        css_sources[path] = source
        for match in CSS_IMPORT_PATTERN.finditer(source):
            resolved, finding = _resolve_css_import(
                project_root=project_root,
                source_root=source_root,
                importer=path,
                specifier=match.group("path"),
            )
            if finding is not None:
                findings.append(finding)
                continue
            assert resolved is not None
            if resolved not in importers:
                queue.append(resolved)
                importers[resolved] = set()
            importers[resolved].add(path)

    discovered_relative = {
        _relative(project_root, path)
        for path in css_sources
    }
    for required in sorted(REQUIRED_IMPORTED_CSS):
        if required not in discovered_relative:
            findings.append(
                _finding(
                    code="DQ-CSS-REQUIRED-IMPORT",
                    message=f"required product CSS is not imported: {required}",
                    project_root=project_root,
                    path=project_root / required,
                )
            )

    ordered = tuple(
        CssSource(
            path=path,
            source=css_sources[path],
            imported_by=tuple(
                sorted(
                    importers.get(path, ()),
                    key=lambda item: item.as_posix().casefold(),
                )
            ),
        )
        for path in sorted(css_sources, key=lambda item: item.as_posix().casefold())
    )
    return ordered, tuple(findings)


def _duration_ms(token: str) -> float:
    match = TIME_PATTERN.fullmatch(token.strip())
    if match is None:
        raise ValueError(f"duration must be a literal ms/s value, got {token!r}")
    value = float(match.group(1))
    return value * 1000.0 if match.group(2).casefold() == "s" else value


def _times(value: str) -> tuple[tuple[str, float], ...]:
    return tuple(
        (match.group(0), _duration_ms(match.group(0)))
        for match in TIME_PATTERN.finditer(value)
    )


def _transition_property(segment: str) -> str | None:
    for token in re.split(r"\s+", segment.strip()):
        lowered = token.casefold()
        if not lowered or TIME_PATTERN.fullmatch(lowered):
            continue
        if lowered.startswith(("cubic-bezier(", "steps(", "linear(")):
            continue
        if lowered in {
            "ease",
            "ease-in",
            "ease-out",
            "ease-in-out",
            "linear",
            "step-start",
            "step-end",
            "allow-discrete",
            "normal",
        }:
            continue
        return lowered
    return None


def _animation_name(segment: str, keyframe_names: set[str]) -> str | None:
    for name in sorted(keyframe_names, key=len, reverse=True):
        if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", segment):
            return name
    return None


def _animation_is_ambient(segment: str) -> bool:
    return re.search(r"(?<![-\w])infinite(?![-\w])", segment, re.I) is not None


def _rule_has_ambient_animation(rule: CssRule) -> bool:
    return any(
        _animation_is_ambient(value)
        for value in (
            *rule.values("animation"),
            *rule.values("animation-iteration-count"),
        )
    )


def _is_reduced_motion(rule: CssRule) -> bool:
    return context_contains(rule, "prefers-reduced-motion:reduce")


def _is_hover_capable(rule: CssRule) -> bool:
    return (
        context_contains(rule, "hover:hover")
        and context_contains(rule, "pointer:fine")
    )


def _selector_anchor(selector: str) -> str:
    value = selector.strip()
    value = re.sub(r"::?[a-zA-Z-]+(?:\([^()]*(?:\([^()]*\)[^()]*)*\))?", "", value)
    value = re.sub(r"\[(?:data-state|aria-[^\]]+|open|class)[^\]]*\]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _selector_is_covered(anchor: str, protected: set[str]) -> bool:
    if "*" in protected or anchor in protected:
        return True
    return any(
        anchor.startswith(candidate + " ")
        or anchor.startswith(candidate + ">")
        or anchor.startswith(candidate + " +")
        or anchor.startswith(candidate + " ~")
        for candidate in protected
        if candidate
    )


def _motion_findings(project_root: Path, css_path: Path, css: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        rules = parse_css(css)
    except ValueError as exc:
        return [
            _finding(
                code="DQ-CSS-PARSE",
                message=f"CSS could not be deterministically parsed: {exc}",
                project_root=project_root,
                path=css_path,
            )
        ]

    keyframe_properties: dict[str, set[str]] = {}
    for rule in rules:
        if rule.keyframes is None:
            continue
        properties = keyframe_properties.setdefault(rule.keyframes, set())
        for declaration in rule.declarations:
            property_name = declaration.name.casefold()
            properties.add(property_name)
            if property_name not in ALLOWED_KEYFRAME_PROPERTIES:
                findings.append(
                    _finding(
                        code="DQ-MOTION-KEYFRAME-PROPERTY",
                        message=(
                            f"@keyframes {rule.keyframes!r} animates "
                            f"disallowed property {property_name!r}; allowed: "
                            f"{', '.join(sorted(ALLOWED_KEYFRAME_PROPERTIES))}"
                        ),
                        project_root=project_root,
                        path=css_path,
                        line=declaration.line,
                    )
                )

    keyframe_names = set(keyframe_properties)
    moving_keyframes = {
        name
        for name, properties in keyframe_properties.items()
        if "transform" in properties
    }

    reduced_transform_none: set[str] = set()
    reduced_animation_none: set[str] = set()
    for rule in rules:
        if not _is_reduced_motion(rule):
            continue
        for selector in selector_parts(rule.selector):
            anchor = _selector_anchor(selector)
            if any(
                value.casefold().replace("!important", "").strip() == "none"
                for value in rule.values("transform")
            ):
                reduced_transform_none.add(anchor)
            if any(
                value.casefold().replace("!important", "").strip() == "none"
                for value in rule.values("animation")
            ):
                reduced_animation_none.add(anchor)

    movement_requirements: list[tuple[CssRule, str, str]] = []

    for rule, declaration in iter_declarations(rules, "animation-name"):
        if rule.keyframes is not None or _is_reduced_motion(rule):
            continue
        ambient = _rule_has_ambient_animation(rule)
        for token in split_top_level(declaration.value):
            animation_name = token.strip()
            if animation_name.casefold() == "none":
                continue
            if animation_name not in keyframe_names:
                findings.append(
                    _finding(
                        code="DQ-MOTION-KEYFRAME-UNKNOWN",
                        message=(
                            "animation-name does not resolve to a local "
                            f"keyframe: {animation_name!r}"
                        ),
                        project_root=project_root,
                        path=css_path,
                        line=declaration.line,
                    )
                )
                continue
            if animation_name in moving_keyframes or ambient:
                detail = (
                    f"ambient loop {animation_name}"
                    if ambient
                    else animation_name
                )
                movement_requirements.append((rule, "animation", detail))

    for rule, declaration in iter_declarations(
        rules,
        "transition",
        "transition-property",
        "transition-duration",
        "animation",
        "animation-duration",
    ):
        if rule.keyframes is not None:
            continue
        name = declaration.name.casefold()
        value = declaration.value
        lowered_value = value.casefold()

        if name == "transition":
            for segment in split_top_level(value):
                property_name = _transition_property(segment)
                if property_name is None:
                    findings.append(
                        _finding(
                            code="DQ-MOTION-UNPARSEABLE",
                            message=f"transition segment is missing an explicit property: {segment!r}",
                            project_root=project_root,
                            path=css_path,
                            line=declaration.line,
                        )
                    )
                    continue
                if property_name == "none":
                    continue
                if property_name == "all" or property_name not in ALLOWED_TRANSITION_PROPERTIES:
                    findings.append(
                        _finding(
                            code="DQ-MOTION-PROPERTY",
                            message=(
                                f"transition property {property_name!r} is not allowed; "
                                f"allowed: {', '.join(sorted(ALLOWED_TRANSITION_PROPERTIES))}"
                            ),
                            project_root=project_root,
                            path=css_path,
                            line=declaration.line,
                        )
                    )
                times = _times(segment)
                if not times:
                    findings.append(
                        _finding(
                            code="DQ-MOTION-DURATION-UNKNOWN",
                            message=f"transition duration must be a literal value: {segment!r}",
                            project_root=project_root,
                            path=css_path,
                            line=declaration.line,
                        )
                    )
                elif times[0][1] > MAX_UI_MOTION_MS:
                    findings.append(
                        _finding(
                            code="DQ-MOTION-DURATION",
                            message=(
                                f"transition duration {times[0][0]} exceeds "
                                f"{MAX_UI_MOTION_MS:g}ms"
                            ),
                            project_root=project_root,
                            path=css_path,
                            line=declaration.line,
                            evidence={"property": property_name},
                        )
                    )
                if re.search(r"(?<![-\w])ease-in(?![-\w])", segment):
                    findings.append(
                        _finding(
                            code="DQ-MOTION-EASING",
                            message="ease-in is forbidden for responsive UI motion",
                            project_root=project_root,
                            path=css_path,
                            line=declaration.line,
                        )
                    )
        elif name == "transition-property":
            for property_name in split_top_level(lowered_value):
                if property_name == "none":
                    continue
                if property_name == "all" or property_name not in ALLOWED_TRANSITION_PROPERTIES:
                    findings.append(
                        _finding(
                            code="DQ-MOTION-PROPERTY",
                            message=f"transition-property {property_name!r} is not allowed",
                            project_root=project_root,
                            path=css_path,
                            line=declaration.line,
                        )
                    )
        elif name == "transition-duration":
            durations = _times(value)
            if not durations:
                findings.append(
                    _finding(
                        code="DQ-MOTION-DURATION-UNKNOWN",
                        message="transition-duration must use literal ms/s values",
                        project_root=project_root,
                        path=css_path,
                        line=declaration.line,
                    )
                )
            for token, duration in durations:
                if duration > MAX_UI_MOTION_MS:
                    findings.append(
                        _finding(
                            code="DQ-MOTION-DURATION",
                            message=f"transition duration {token} exceeds 300ms",
                            project_root=project_root,
                            path=css_path,
                            line=declaration.line,
                        )
                    )
        elif name == "animation":
            if lowered_value.replace("!important", "").strip() == "none":
                continue
            for segment in split_top_level(value):
                times = _times(segment)
                ambient = _animation_is_ambient(segment)
                if not times:
                    findings.append(
                        _finding(
                            code="DQ-MOTION-DURATION-UNKNOWN",
                            message=f"animation duration must be a literal value: {segment!r}",
                            project_root=project_root,
                            path=css_path,
                            line=declaration.line,
                        )
                    )
                elif times[0][1] > MAX_UI_MOTION_MS and not ambient:
                    findings.append(
                        _finding(
                            code="DQ-MOTION-DURATION",
                            message=(
                                f"animation duration {times[0][0]} exceeds "
                                f"{MAX_UI_MOTION_MS:g}ms"
                            ),
                            project_root=project_root,
                            path=css_path,
                            line=declaration.line,
                        )
                    )
                if re.search(r"(?<![-\w])ease-in(?![-\w])", segment):
                    findings.append(
                        _finding(
                            code="DQ-MOTION-EASING",
                            message="ease-in is forbidden for UI animation",
                            project_root=project_root,
                            path=css_path,
                            line=declaration.line,
                        )
                    )
                animation_name = _animation_name(segment, keyframe_names)
                if animation_name is None:
                    findings.append(
                        _finding(
                            code="DQ-MOTION-KEYFRAME-UNKNOWN",
                            message=f"animation does not resolve to a local keyframe: {segment!r}",
                            project_root=project_root,
                            path=css_path,
                            line=declaration.line,
                        )
                    )
                elif not _is_reduced_motion(rule):
                    if animation_name in moving_keyframes:
                        movement_requirements.append(
                            (rule, "animation", animation_name)
                        )
                    if ambient:
                        movement_requirements.append(
                            (
                                rule,
                                "animation",
                                f"ambient loop {animation_name}",
                            )
                        )
        elif name == "animation-duration":
            durations = _times(value)
            ambient = _rule_has_ambient_animation(rule)
            if not durations:
                findings.append(
                    _finding(
                        code="DQ-MOTION-DURATION-UNKNOWN",
                        message="animation-duration must use literal ms/s values",
                        project_root=project_root,
                        path=css_path,
                        line=declaration.line,
                    )
                )
            for token, duration in durations:
                if duration > MAX_UI_MOTION_MS and not ambient:
                    findings.append(
                        _finding(
                            code="DQ-MOTION-DURATION",
                            message=f"animation duration {token} exceeds 300ms",
                            project_root=project_root,
                            path=css_path,
                            line=declaration.line,
                        )
                    )
            if ambient and not _is_reduced_motion(rule):
                movement_requirements.append(
                    (rule, "animation", "ambient loop")
                )

    for rule in rules:
        if rule.keyframes is not None or _is_reduced_motion(rule):
            continue
        transform_values = rule.values("transform")
        animation_values = rule.values("animation")
        has_movement = any(
            value.casefold().replace("!important", "").strip() != "none"
            and re.search(r"\b(?:translate|scale|rotate|matrix|perspective)", value, re.I)
            for value in transform_values
        )
        if has_movement and (
            MOTION_PSEUDO_PATTERN.search(rule.selector)
            or any("transform" in value.casefold() for value in rule.values("transition"))
        ):
            movement_requirements.append((rule, "transform", "interactive movement"))

        if ":hover" in rule.selector and (
            has_movement
            or any(
                _animation_name(segment, moving_keyframes) is not None
                for value in animation_values
                for segment in split_top_level(value)
            )
        ) and not _is_hover_capable(rule):
            findings.append(
                _finding(
                    code="DQ-HOVER-MOTION-GATE",
                    message=(
                        "hover movement must be inside "
                        "@media (hover: hover) and (pointer: fine)"
                    ),
                    project_root=project_root,
                    path=css_path,
                    line=rule.line,
                    evidence={"selector": rule.selector},
                )
            )

    seen_requirements: set[tuple[int, str, str]] = set()
    for rule, kind, detail in movement_requirements:
        for selector in selector_parts(rule.selector):
            anchor = _selector_anchor(selector)
            identity = (rule.line, anchor, kind)
            if identity in seen_requirements:
                continue
            seen_requirements.add(identity)
            protected = (
                reduced_animation_none
                if kind == "animation"
                else reduced_transform_none
            )
            if not _selector_is_covered(anchor, protected):
                findings.append(
                    _finding(
                        code="DQ-REDUCED-MOTION-MOVEMENT",
                        message=(
                            f"reduced-motion must disable {kind} movement for "
                            f"{selector.strip()!r} ({detail})"
                        ),
                        project_root=project_root,
                        path=css_path,
                        line=rule.line,
                    )
                )

    if "scroll-behavior: smooth" in re.sub(r"\s+", " ", css.casefold()):
        findings.append(
            _finding(
                code="DQ-NAV-SMOOTH-SCROLL",
                message="frequent navigation must not use CSS smooth scrolling",
                project_root=project_root,
                path=css_path,
            )
        )
    return findings


def _navigation_findings(
    project_root: Path,
    css_path: Path,
    css: str,
    app_path: Path,
    app_source: str,
) -> list[Finding]:
    findings: list[Finding] = []
    if re.search(r"scroll-behavior\s*:\s*smooth", css, re.I):
        findings.append(
            _finding(
                code="DQ-NAV-SMOOTH-SCROLL",
                message="CSS smooth scrolling is forbidden for frequent navigation",
                project_root=project_root,
                path=css_path,
            )
        )
    if re.search(r"behavior\s*:\s*[\"']smooth[\"']", app_source):
        findings.append(
            _finding(
                code="DQ-NAV-SMOOTH-SCROLL",
                message="programmatic smooth scrolling is forbidden for frequent navigation",
                project_root=project_root,
                path=app_path,
            )
        )
    required_markers = (
        'mainRef.current?.scrollTo({ top: 0, behavior: "auto" })',
        'setOverviewRoom("home")',
        'if (section === "overview")',
    )
    for marker in required_markers:
        if marker not in app_source:
            findings.append(
                _finding(
                    code="DQ-NAV-RESET-CONTRACT",
                    message=f"navigation reset contract is missing marker {marker!r}",
                    project_root=project_root,
                    path=app_path,
                )
            )
    if not re.search(r"\.workspace-main\s*\{[^}]*scroll-behavior\s*:\s*auto", css, re.S):
        findings.append(
            _finding(
                code="DQ-NAV-AUTO-SCROLL",
                message=".workspace-main must explicitly use scroll-behavior: auto",
                project_root=project_root,
                path=css_path,
            )
        )
    return findings


def _rules_with_selector(
    rules: Sequence[CssRule],
    selector: str,
) -> tuple[CssRule, ...]:
    return tuple(
        rule
        for rule in rules
        if selector in {part.strip() for part in selector_parts(rule.selector)}
    )


def _cascaded_declarations(
    rules: Sequence[CssRule],
    selector: str,
    *,
    context_fragment: str | None = None,
    base_only: bool = False,
) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for rule in _rules_with_selector(rules, selector):
        if rule.keyframes is not None:
            continue
        if base_only and rule.contexts:
            continue
        if context_fragment is not None and not context_contains(
            rule,
            context_fragment,
        ):
            continue
        for declaration in rule.declarations:
            declarations[declaration.name.casefold()] = declaration.value
    return declarations


def _normalized_declarations(values: Mapping[str, str]) -> str:
    return re.sub(
        r"\s+",
        "",
        ";".join(
            f"{name}:{value}"
            for name, value in sorted(values.items())
        ).casefold(),
    )


def _forced_colors_findings(
    project_root: Path,
    css_path: Path,
    css: str,
) -> list[Finding]:
    findings: list[Finding] = []
    try:
        rules = parse_css(css)
    except ValueError as exc:
        return [
            _finding(
                code="DQ-CSS-PARSE",
                message=f"forced-colors CSS could not be parsed: {exc}",
                project_root=project_root,
                path=css_path,
            )
        ]
    forced_rules = tuple(
        rule for rule in rules if context_contains(rule, "forced-colors:active")
    )
    if not forced_rules:
        return [
            _finding(
                code="DQ-FORCED-COLORS-MISSING",
                message="@media (forced-colors: active) is required",
                project_root=project_root,
                path=css_path,
            )
        ]

    required_surfaces = (
        "body",
        ".navigation-rail",
        ".top-bar",
        ".studio-hub",
        ".studio-room-grid",
        ".studio-room-portal",
        ".studio-room-header",
        ".media-drop-overlay",
        ".media-drop-overlay__card",
    )
    for selector in required_surfaces:
        matched = _rules_with_selector(forced_rules, selector)
        if not matched:
            findings.append(
                _finding(
                    code="DQ-FORCED-COLORS-SURFACE",
                    message=f"forced-colors coverage is missing for {selector}",
                    project_root=project_root,
                    path=css_path,
                )
            )
            continue
        merged = " ".join(
            f"{declaration.name}:{declaration.value}"
            for rule in matched
            for declaration in rule.declarations
        ).casefold()
        if "canvas" not in merged:
            findings.append(
                _finding(
                    code="DQ-FORCED-COLORS-CANVAS",
                    message=f"{selector} must map to Canvas/CanvasText in forced colors",
                    project_root=project_root,
                    path=css_path,
                    line=matched[0].line,
                )
            )
        if selector != "body" and "forced-color-adjust:auto" not in re.sub(
            r"\s+", "", merged
        ):
            findings.append(
                _finding(
                    code="DQ-FORCED-COLORS-ADJUST",
                    message=f"{selector} must explicitly use forced-color-adjust: auto",
                    project_root=project_root,
                    path=css_path,
                    line=matched[0].line,
                )
            )
    return findings


def _background_findings(
    project_root: Path,
    app_path: Path,
    app: str,
    component_path: Path,
    component: str,
    css_path: Path,
    css: str,
) -> list[Finding]:
    findings: list[Finding] = []
    asset_paths = {
        "day": project_root / "apps/desktop/src/assets/day.jpg",
        "night": project_root / "apps/desktop/src/assets/night.jpg",
        "scene": project_root / "apps/desktop/src/assets/scene.png",
    }
    for role, path in asset_paths.items():
        try:
            if not path.resolve(strict=True).is_file() or path.stat().st_size <= 0:
                raise OSError("not a non-empty file")
        except OSError as exc:
            findings.append(
                _finding(
                    code="DQ-BACKGROUND-ASSET",
                    message=f"required {role} background asset is unavailable: {exc}",
                    project_root=project_root,
                    path=path,
                )
            )

    required_component_markers = (
        'import dayArtwork from "../assets/day.jpg"',
        'import nightArtwork from "../assets/night.jpg"',
        'import sceneArtwork from "../assets/scene.png"',
        'data-background-contract="remote-day-night-scene"',
        'className="scene-backdrop__theme scene-backdrop__theme--light"',
        'className="scene-backdrop__theme scene-backdrop__theme--dark"',
        'className="scene-backdrop__illustration"',
        "src={dayArtwork}",
        "src={nightArtwork}",
        "src={sceneArtwork}",
        'aria-hidden="true"',
    )
    for marker in required_component_markers:
        if marker not in component:
            findings.append(
                _finding(
                    code="DQ-BACKGROUND-LAYER-CONTRACT",
                    message=f"SceneBackdrop is missing explicit layer marker {marker!r}",
                    project_root=project_root,
                    path=component_path,
                )
            )

    if (
        'import { SceneBackdrop } from "./components/SceneBackdrop"' not in app
        or app.count("<SceneBackdrop") < 3
    ):
        findings.append(
            _finding(
                code="DQ-BACKGROUND-APP-INTEGRATION",
                message=(
                    "App must import SceneBackdrop and mount it in loading, "
                    "failure, and ready states"
                ),
                project_root=project_root,
                path=app_path,
                evidence={"mountCount": app.count("<SceneBackdrop")},
            )
        )

    try:
        rules = parse_css(css)
    except ValueError as exc:
        findings.append(
            _finding(
                code="DQ-CSS-PARSE",
                message=f"SceneBackdrop CSS could not be parsed: {exc}",
                project_root=project_root,
                path=css_path,
            )
        )
        return findings

    root = _cascaded_declarations(
        rules,
        ".scene-backdrop",
        base_only=True,
    )
    normalized_root = _normalized_declarations(root)
    for marker in ("position:fixed", "inset:0", "overflow:hidden"):
        if marker not in normalized_root:
            findings.append(
                _finding(
                    code="DQ-BACKGROUND-FULLSCREEN",
                    message=f"SceneBackdrop full-screen root is missing {marker}",
                    project_root=project_root,
                    path=css_path,
                )
            )

    cover = _normalized_declarations(
        _cascaded_declarations(
            rules,
            ".scene-backdrop__theme",
            base_only=True,
        )
    )
    illustration_cover = _normalized_declarations(
        _cascaded_declarations(
            rules,
            ".scene-backdrop__illustration",
            base_only=True,
        )
    )
    if (
        "object-fit:cover" not in cover
        or "object-fit:cover" not in illustration_cover
    ):
        findings.append(
            _finding(
                code="DQ-BACKGROUND-COVER",
                message="day, night, and scene image layers must use object-fit: cover",
                project_root=project_root,
                path=css_path,
            )
        )

    base_theme = _cascaded_declarations(
        rules,
        ".scene-backdrop__theme",
        base_only=True,
    )
    if re.sub(r"\s+", "", base_theme.get("opacity", "")) != "0":
        findings.append(
            _finding(
                code="DQ-BACKGROUND-THEME-VISIBILITY",
                message="inactive day/night layers must default to opacity: 0",
                project_root=project_root,
                path=css_path,
            )
        )
    themes = (
        ':root[data-theme="light"] .scene-backdrop__theme--light',
        ':root[data-theme="dark"] .scene-backdrop__theme--dark',
    )
    for selector in themes:
        values = _cascaded_declarations(rules, selector, base_only=True)
        if re.sub(r"\s+", "", values.get("opacity", "")) != "1":
            findings.append(
                _finding(
                    code="DQ-BACKGROUND-THEME-VISIBILITY",
                    message=f"{selector} must make its theme layer visible",
                    project_root=project_root,
                    path=css_path,
                )
            )

    illustration = _cascaded_declarations(
        rules,
        ".scene-backdrop__illustration",
        base_only=True,
    )
    try:
        illustration_opacity = float(illustration.get("opacity", "0"))
    except ValueError:
        illustration_opacity = 0
    if illustration_opacity <= 0:
        findings.append(
            _finding(
                code="DQ-BACKGROUND-SCENE-VISIBILITY",
                message="scene.png must remain visibly layered above the theme artwork",
                project_root=project_root,
                path=css_path,
            )
        )

    reduced_theme = _normalized_declarations(
        _cascaded_declarations(
            rules,
            ".scene-backdrop__theme",
            context_fragment="prefers-reduced-motion:reduce",
        )
    )
    reduced_scene = _normalized_declarations(
        _cascaded_declarations(
            rules,
            ".scene-backdrop__illustration",
            context_fragment="prefers-reduced-motion:reduce",
        )
    )
    if "transform:none" not in reduced_theme or "transform:none" not in reduced_scene:
        findings.append(
            _finding(
                code="DQ-BACKGROUND-REDUCED-MOTION",
                message=(
                    "reduced motion must remove positional transforms from "
                    "theme and scene layers"
                ),
                project_root=project_root,
                path=css_path,
            )
        )

    forced = _normalized_declarations(
        _cascaded_declarations(
            rules,
            ".scene-backdrop",
            context_fragment="forced-colors:active",
        )
    )
    if "display:none" not in forced:
        findings.append(
            _finding(
                code="DQ-BACKGROUND-FORCED-COLORS",
                message="forced colors must hide decorative background artwork",
                project_root=project_root,
                path=css_path,
            )
        )
    return findings


def _desktop_companion_findings(
    project_root: Path,
    app_path: Path,
    app: str,
    component_path: Path,
    component: str,
    css_path: Path,
    css: str,
) -> list[Finding]:
    findings: list[Finding] = []
    asset_path = project_root / "apps/desktop/src/assets/companion.gif"
    try:
        digest = hashlib.sha256(asset_path.read_bytes()).hexdigest().upper()
    except OSError as exc:
        findings.append(
            _finding(
                code="DQ-COMPANION-ASSET",
                message=f"companion.gif is unavailable: {exc}",
                project_root=project_root,
                path=asset_path,
            )
        )
    else:
        if digest != COMPANION_SHA256:
            findings.append(
                _finding(
                    code="DQ-COMPANION-ASSET-HASH",
                    message=(
                        "companion.gif must remain byte-identical to the "
                        f"approved remote artwork; found SHA-256 {digest}"
                    ),
                    project_root=project_root,
                    path=asset_path,
                    evidence={
                        "expectedSha256": COMPANION_SHA256,
                        "actualSha256": digest,
                    },
                )
            )

    required_component_markers = (
        'import "./DesktopCompanion.css"',
        'new URL("../assets/companion.gif", import.meta.url).href',
        "export interface DesktopCompanionProps",
        "collapseLabel: string",
        "expandLabel: string",
        "imageAlt: string",
        "regionLabel: string",
        "statusText: string",
        "aria-label={regionLabel}",
        'role="status"',
        'aria-live="polite"',
        "aria-expanded={!collapsed}",
        'aria-keyshortcuts="ArrowUp ArrowDown ArrowLeft ArrowRight Enter Space"',
        "aria-label={collapsed ? expandLabel : collapseLabel}",
        "alt={imageAlt}",
        'type="button"',
        "setPointerCapture(event.pointerId)",
        "releasePointerCapture(event.pointerId)",
    )
    for marker in required_component_markers:
        if marker not in component:
            findings.append(
                _finding(
                    code="DQ-COMPANION-ACCESSIBILITY",
                    message=f"DesktopCompanion is missing required marker {marker!r}",
                    project_root=project_root,
                    path=component_path,
                )
            )

    try:
        rules = parse_css(css)
    except ValueError as exc:
        findings.append(
            _finding(
                code="DQ-CSS-PARSE",
                message=f"DesktopCompanion CSS could not be parsed: {exc}",
                project_root=project_root,
                path=css_path,
            )
        )
    else:
        reduced = _normalized_declarations(
            _cascaded_declarations(
                rules,
                ".desktop-companion",
                context_fragment="prefers-reduced-motion:reduce",
            )
        )
        reduced_button = _normalized_declarations(
            _cascaded_declarations(
                rules,
                ".desktop-companion__button",
                context_fragment="prefers-reduced-motion:reduce",
            )
        )
        if not (
            "animation:none!important" in reduced
            and "transition:none!important" in reduced
            and "animation:none!important" in reduced_button
            and "transition:none!important" in reduced_button
        ):
            findings.append(
                _finding(
                    code="DQ-COMPANION-REDUCED-MOTION",
                    message=(
                        "DesktopCompanion root and interactive button must "
                        "disable animation and transition under reduced motion"
                    ),
                    project_root=project_root,
                    path=css_path,
                )
            )

        forced_status = _normalized_declarations(
            _cascaded_declarations(
                rules,
                ".desktop-companion__status",
                context_fragment="forced-colors:active",
            )
        )
        forced_visual = _normalized_declarations(
            _cascaded_declarations(
                rules,
                ".desktop-companion__visual",
                context_fragment="forced-colors:active",
            )
        )
        required_forced = (
            "color:canvastext",
            "background:canvas",
            "forced-color-adjust:auto",
        )
        if any(
            marker not in values
            for values in (forced_status, forced_visual)
            for marker in required_forced
        ):
            findings.append(
                _finding(
                    code="DQ-COMPANION-FORCED-COLORS",
                    message=(
                        "DesktopCompanion status and visual must use "
                        "Canvas/CanvasText with forced-color-adjust: auto"
                    ),
                    project_root=project_root,
                    path=css_path,
                )
            )

    import_marker = (
        'import { DesktopCompanion } from "./components/DesktopCompanion"'
    )
    match = re.search(r"<DesktopCompanion\b(?P<props>.*?)\/>", app, re.S)
    if import_marker not in app or match is None:
        findings.append(
            _finding(
                code="DQ-COMPANION-APP-INTEGRATION",
                message="App must import and mount the localized DesktopCompanion",
                project_root=project_root,
                path=app_path,
            )
        )
    else:
        props = match.group("props")
        for prop in (
            "collapseLabel",
            "expandLabel",
            "imageAlt",
            "regionLabel",
            "statusText",
        ):
            if re.search(
                rf"\b{prop}\s*=\s*\{{\s*t\(\s*[\"'][^\"']+[\"']",
                props,
            ) is None:
                findings.append(
                    _finding(
                        code="DQ-COMPANION-I18N",
                        message=f"DesktopCompanion {prop} must come from t(...)",
                        project_root=project_root,
                        path=app_path,
                    )
                )
    return findings


def _custom_properties(rules: Sequence[CssRule], selector: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for rule in _rules_with_selector(rules, selector):
        for declaration in rule.declarations:
            if declaration.name.startswith("--"):
                values[declaration.name] = declaration.value
    return values


def _glass_findings(
    project_root: Path,
    tokens_path: Path,
    tokens: str,
    css_path: Path,
    css: str,
) -> list[Finding]:
    findings: list[Finding] = []
    try:
        token_rules = parse_css(tokens)
    except ValueError as exc:
        findings.append(
            _finding(
                code="DQ-CSS-PARSE",
                message=f"design token CSS could not be parsed: {exc}",
                project_root=project_root,
                path=tokens_path,
            )
        )
        return findings
    try:
        css_rules = parse_css(css)
    except ValueError as exc:
        findings.append(
            _finding(
                code="DQ-CSS-PARSE",
                message=f"glass surface CSS could not be parsed: {exc}",
                project_root=project_root,
                path=css_path,
            )
        )
        return findings
    light = _custom_properties(token_rules, ":root")
    dark = _custom_properties(token_rules, ':root[data-theme="dark"]')
    for theme, values in (("light", light), ("dark", dark)):
        for token in sorted(REQUIRED_GLASS_TOKENS | REQUIRED_PALETTE_TOKENS):
            if token not in values or not values[token].strip():
                findings.append(
                    _finding(
                        code="DQ-TOKEN-MISSING",
                        message=f"{theme} theme is missing required token {token}",
                        project_root=project_root,
                        path=tokens_path,
                    )
                )

    for token, canonical in CANONICAL_EASINGS.items():
        ease_value = re.sub(r"\s+", " ", light.get(token, "")).strip()
        if ease_value != canonical:
            findings.append(
                _finding(
                    code="DQ-EASING-TOKEN",
                    message=(
                        f"{token} must use the Design Pack canonical curve "
                        f"{canonical}; found {ease_value!r}"
                    ),
                    project_root=project_root,
                    path=tokens_path,
                    evidence={"token": token, "expected": canonical},
                )
            )

    frosted_selectors = (
        ".navigation-rail",
        ".top-bar",
        ".panel",
        ".studio-hub",
        ".studio-room-grid",
        ".studio-room-header",
        ".media-drop-overlay__card",
    )
    for selector in frosted_selectors:
        matched = _rules_with_selector(css_rules, selector)
        merged = " ".join(
            f"{declaration.name}:{declaration.value}"
            for rule in matched
            for declaration in rule.declarations
        ).casefold()
        if "backdrop-filter" not in merged or "blur(" not in merged:
            findings.append(
                _finding(
                    code="DQ-GLASS-SURFACE",
                    message=f"{selector} must use a blur-based backdrop-filter",
                    project_root=project_root,
                    path=css_path,
                    line=matched[0].line if matched else None,
                )
            )
        if not any(
            token in merged
            for token in (
                "var(--surface-glass)",
                "var(--surface-strong)",
                "var(--surface-soft)",
                "color-mix(",
            )
        ):
            findings.append(
                _finding(
                    code="DQ-GLASS-BACKGROUND",
                    message=f"{selector} must use a translucent surface token",
                    project_root=project_root,
                    path=css_path,
                    line=matched[0].line if matched else None,
                )
            )
    if not any(
        context_contains(rule, "@supportsnot((backdrop-filter")
        or (
            any("supports not" in context.casefold() for context in rule.contexts)
            and any(
                declaration.name.casefold() == "background-color"
                for declaration in rule.declarations
            )
        )
        for rule in css_rules
    ):
        findings.append(
            _finding(
                code="DQ-GLASS-FALLBACK",
                message="frosted glass requires an opaque @supports not fallback",
                project_root=project_root,
                path=css_path,
            )
        )
    return findings


def _find_balanced(source: str, opening: int, opener: str, closer: str) -> int:
    depth = 1
    quote: str | None = None
    escaped = False
    for index in range(opening + 1, len(source)):
        char = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {'"', "'", "`"}:
            quote = char
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unclosed {opener}{closer} block")


def _extract_array_strings(source: str, name: str) -> tuple[str, ...]:
    match = re.search(
        rf"\b(?:export\s+)?const\s+{re.escape(name)}\b[^=]*=\s*\[",
        source,
    )
    if match is None:
        raise ValueError(f"array {name} was not found")
    opening = source.find("[", match.start())
    closing = _find_balanced(source, opening, "[", "]")
    return tuple(re.findall(r'["\']([^"\']+)["\']', source[opening + 1 : closing]))


def _extract_object(source: str, name: str) -> str:
    match = re.search(
        rf"\b(?:export\s+)?const\s+{re.escape(name)}\b[^=]*=\s*\{{",
        source,
    )
    if match is None:
        raise ValueError(f"object {name} was not found")
    opening = source.find("{", match.start())
    closing = _find_balanced(source, opening, "{", "}")
    return source[opening + 1 : closing]


def _skip_typescript_trivia(source: str, index: int) -> int:
    while index < len(source):
        if source[index].isspace() or source[index] == ",":
            index += 1
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            return len(source) if newline < 0 else _skip_typescript_trivia(
                source,
                newline + 1,
            )
        if source.startswith("/*", index):
            closing = source.find("*/", index + 2)
            if closing < 0:
                raise ValueError("unclosed TypeScript block comment")
            index = closing + 2
            continue
        return index
    return index


def _read_typescript_string(source: str, index: int) -> tuple[str, int]:
    if index >= len(source) or source[index] not in {'"', "'", "`"}:
        raise ValueError("expected a TypeScript string literal")
    quote = source[index]
    index += 1
    value: list[str] = []
    while index < len(source):
        char = source[index]
        if char == "\\":
            if index + 1 >= len(source):
                raise ValueError("unterminated TypeScript string escape")
            value.extend((char, source[index + 1]))
            index += 2
            continue
        if char == quote:
            return "".join(value), index + 1
        if quote == "`" and source.startswith("${", index):
            raise ValueError("catalog messages must not use template interpolation")
        value.append(char)
        index += 1
    raise ValueError("unterminated TypeScript string literal")


def _read_typescript_identifier(source: str, index: int) -> tuple[str, int]:
    match = re.match(r"[A-Za-z_][A-Za-z0-9_-]*", source[index:])
    if match is None:
        raise ValueError(f"expected TypeScript identifier near {source[index:index + 40]!r}")
    return match.group(0), index + len(match.group(0))


def _read_typescript_reference(source: str, index: int) -> tuple[str, int]:
    """Read a static identifier with optional dot or string-index access."""

    root, index = _read_typescript_identifier(source, index)
    reference = root
    while True:
        while index < len(source) and source[index].isspace():
            index += 1
        if index < len(source) and source[index] == ".":
            property_name, index = _read_typescript_identifier(source, index + 1)
            reference += f".{property_name}"
            continue
        if index < len(source) and source[index] == "[":
            index = _skip_typescript_trivia(source, index + 1)
            property_name, index = _read_typescript_string(source, index)
            index = _skip_typescript_trivia(source, index)
            if index >= len(source) or source[index] != "]":
                raise ValueError("expected closing bracket in TypeScript reference")
            reference += f"[{json.dumps(property_name, ensure_ascii=False)}]"
            index += 1
            continue
        return reference, index


def _typescript_reference_parts(reference: str) -> tuple[str, tuple[str, ...]]:
    match = re.match(r"[A-Za-z_][A-Za-z0-9_-]*", reference)
    if match is None:
        raise ValueError(f"invalid TypeScript reference {reference!r}")
    root = match.group(0)
    index = match.end()
    properties: list[str] = []
    while index < len(reference):
        if reference[index] == ".":
            property_match = re.match(
                r"[A-Za-z_][A-Za-z0-9_-]*",
                reference[index + 1 :],
            )
            if property_match is None:
                raise ValueError(f"invalid TypeScript reference {reference!r}")
            properties.append(property_match.group(0))
            index += 1 + len(property_match.group(0))
            continue
        if reference[index] == "[":
            closing = reference.find("]", index + 1)
            if closing < 0:
                raise ValueError(f"invalid TypeScript reference {reference!r}")
            raw_property = reference[index + 1 : closing]
            try:
                property_name = json.loads(raw_property)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid TypeScript reference {reference!r}"
                ) from exc
            if not isinstance(property_name, str):
                raise ValueError(f"invalid TypeScript reference {reference!r}")
            properties.append(property_name)
            index = closing + 1
            continue
        raise ValueError(f"invalid TypeScript reference {reference!r}")
    return root, tuple(properties)


def _parse_typescript_object(
    body: str,
) -> tuple[dict[str, tuple[str, str]], tuple[str, ...]]:
    """Parse the static string/reference object subset used by catalog.ts."""

    entries: dict[str, tuple[str, str]] = {}
    spreads: list[str] = []
    index = 0
    while True:
        index = _skip_typescript_trivia(body, index)
        if index >= len(body):
            return entries, tuple(spreads)
        if body.startswith("...", index):
            spread, index = _read_typescript_reference(body, index + 3)
            spreads.append(spread)
            index = _skip_typescript_trivia(body, index)
            continue

        if body[index] in {'"', "'", "`"}:
            key, index = _read_typescript_string(body, index)
        else:
            key, index = _read_typescript_identifier(body, index)
        if key in entries:
            raise ValueError(f"duplicate object key {key!r}")

        index = _skip_typescript_trivia(body, index)
        if index >= len(body) or body[index] != ":":
            entries[key] = ("reference", key)
            continue
        index = _skip_typescript_trivia(body, index + 1)
        if index >= len(body):
            raise ValueError(f"missing value for object key {key!r}")
        if body[index] in {'"', "'", "`"}:
            value, index = _read_typescript_string(body, index)
            entries[key] = ("string", value)
        else:
            value, index = _read_typescript_identifier(body, index)
            entries[key] = ("reference", value)


def _catalog_values(
    source: str,
    name: str,
) -> tuple[dict[str, str], tuple[str, ...]]:
    entries, spreads = _parse_typescript_object(_extract_object(source, name))
    values: dict[str, str] = {}
    for key, (kind, value) in entries.items():
        if kind != "string":
            raise ValueError(f"{name}.{key} must be a static string literal")
        values[key] = value
    return values, spreads


def _resolve_typescript_import(
    *,
    project_root: Path,
    importer: Path,
    specifier: str,
) -> Path | None:
    if not specifier.startswith("."):
        return None
    base = importer.parent / specifier
    candidates = (
        base,
        base.with_suffix(".ts"),
        base.with_suffix(".tsx"),
        base / "index.ts",
        base / "index.tsx",
    )
    root = project_root.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _imported_catalog_objects(
    *,
    project_root: Path,
    catalog_path: Path,
    source: str,
) -> dict[str, tuple[str, str]]:
    """Map local named imports to source text and exported identifiers."""

    imported: dict[str, tuple[str, str]] = {}
    pattern = re.compile(
        r"\bimport\s*\{(?P<bindings>[^}]*)\}\s*from\s*"
        r'["\'](?P<specifier>[^"\']+)["\']\s*;?',
        re.S,
    )
    for match in pattern.finditer(source):
        imported_path = _resolve_typescript_import(
            project_root=project_root,
            importer=catalog_path,
            specifier=match.group("specifier"),
        )
        if imported_path is None:
            continue
        imported_source = imported_path.read_text(encoding="utf-8")
        for raw_binding in split_top_level(match.group("bindings")):
            binding = re.sub(r"^\s*type\s+", "", raw_binding).strip()
            if not binding:
                continue
            alias = re.fullmatch(
                r"(?P<exported>[A-Za-z_][A-Za-z0-9_]*)"
                r"(?:\s+as\s+(?P<local>[A-Za-z_][A-Za-z0-9_]*))?",
                binding,
            )
            if alias is None:
                continue
            exported = alias.group("exported")
            local = alias.group("local") or exported
            imported[local] = (imported_source, exported)
    return imported


def _imported_json_catalogs(
    *,
    project_root: Path,
    catalog_path: Path,
    source: str,
) -> dict[str, Mapping[str, Any]]:
    """Load repository-local JSON catalogs imported through a default binding."""

    imported: dict[str, Mapping[str, Any]] = {}
    pattern = re.compile(
        r"\bimport\s+(?P<local>[A-Za-z_][A-Za-z0-9_]*)\s+from\s*"
        r'["\'](?P<specifier>[^"\']+\.json)["\']\s*;?',
    )
    for match in pattern.finditer(source):
        imported_path = _resolve_typescript_import(
            project_root=project_root,
            importer=catalog_path,
            specifier=match.group("specifier"),
        )
        if imported_path is None or imported_path.suffix.casefold() != ".json":
            continue
        try:
            payload = json.loads(imported_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot load imported JSON catalog {match.group('specifier')!r}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"imported JSON catalog {match.group('specifier')!r} "
                "must be an object"
            )
        imported[match.group("local")] = payload
    return imported


def _json_catalog_values(
    *,
    reference: str,
    imported_json: Mapping[str, Mapping[str, Any]],
) -> dict[str, str] | None:
    root, properties = _typescript_reference_parts(reference)
    payload: Any = imported_json.get(root)
    if payload is None:
        return None
    for property_name in properties:
        if not isinstance(payload, dict) or property_name not in payload:
            raise ValueError(
                f"JSON catalog reference {reference!r} does not resolve"
            )
        payload = payload[property_name]
    if not isinstance(payload, dict):
        raise ValueError(
            f"JSON catalog reference {reference!r} must resolve to an object"
        )
    values: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                f"JSON catalog reference {reference!r} must contain "
                "only string keys and values"
            )
        values[key] = value
    return values


def _locale_findings(
    project_root: Path,
    catalog_path: Path,
    source: str,
    core_path: Path,
    core: str,
) -> list[Finding]:
    findings: list[Finding] = []
    try:
        locales = _extract_array_strings(source, "SUPPORTED_LOCALES")
        english, english_spreads = _catalog_values(source, "ENGLISH_MESSAGES")
        mapping_entries, mapping_spreads = _parse_typescript_object(
            _extract_object(source, "MESSAGE_CATALOGS")
        )
    except ValueError as exc:
        return [
            _finding(
                code="DQ-I18N-PARSE",
                message=f"i18n catalog could not be parsed: {exc}",
                project_root=project_root,
                path=catalog_path,
            )
        ]

    if locales != EXPECTED_LOCALES:
        findings.append(
            _finding(
                code="DQ-I18N-LOCALES",
                message=(
                    f"supported locales must be exactly {EXPECTED_LOCALES!r}; "
                    f"found {locales!r}"
                ),
                project_root=project_root,
                path=catalog_path,
            )
        )
    if not english:
        findings.append(
            _finding(
                code="DQ-I18N-ENGLISH-EMPTY",
                message="ENGLISH_MESSAGES must define the complete message contract",
                project_root=project_root,
                path=catalog_path,
            )
        )
    if mapping_spreads:
        findings.append(
            _finding(
                code="DQ-I18N-CATALOG-MAP",
                message="MESSAGE_CATALOGS must list every locale explicitly",
                project_root=project_root,
                path=catalog_path,
            )
        )

    mappings = {
        key: value
        for key, (kind, value) in mapping_entries.items()
        if kind == "reference"
    }
    invalid_mapping_values = sorted(
        key
        for key, (kind, _value) in mapping_entries.items()
        if kind != "reference"
    )
    if invalid_mapping_values:
        findings.append(
            _finding(
                code="DQ-I18N-CATALOG-MAP",
                message=(
                    "MESSAGE_CATALOGS values must be catalog references: "
                    + ", ".join(invalid_mapping_values)
                ),
                project_root=project_root,
                path=catalog_path,
            )
        )
    if set(mappings) != set(EXPECTED_LOCALES):
        findings.append(
            _finding(
                code="DQ-I18N-CATALOG-MAP",
                message=(
                    "MESSAGE_CATALOGS keys must be exactly the supported nine "
                    f"locales; found {tuple(sorted(mappings))!r}"
                ),
                project_root=project_root,
                path=catalog_path,
            )
        )
    if mappings.get("en") != "ENGLISH_MESSAGES":
        findings.append(
            _finding(
                code="DQ-I18N-CATALOG-MAP",
                message="the en locale must map directly to ENGLISH_MESSAGES",
                project_root=project_root,
                path=catalog_path,
            )
        )

    object_cache: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {
        "ENGLISH_MESSAGES": (english, english_spreads)
    }
    imported_objects = _imported_catalog_objects(
        project_root=project_root,
        catalog_path=catalog_path,
        source=source,
    )
    try:
        imported_json = _imported_json_catalogs(
            project_root=project_root,
            catalog_path=catalog_path,
            source=source,
        )
    except ValueError as exc:
        findings.append(
            _finding(
                code="DQ-I18N-PARSE",
                message=f"cannot resolve imported JSON catalogs: {exc}",
                project_root=project_root,
                path=catalog_path,
            )
        )
        imported_json = {}

    def resolved_catalog(
        name: str,
        stack: tuple[str, ...] = (),
    ) -> tuple[dict[str, str], set[str]]:
        if name in stack:
            raise ValueError("catalog spread cycle: " + " -> ".join(stack + (name,)))
        if name not in object_cache:
            json_values = _json_catalog_values(
                reference=name,
                imported_json=imported_json,
            )
            if json_values is not None:
                object_cache[name] = (json_values, ())
            else:
                imported = imported_objects.get(name)
                if imported is None:
                    object_cache[name] = _catalog_values(source, name)
                else:
                    imported_source, exported_name = imported
                    object_cache[name] = _catalog_values(
                        imported_source,
                        exported_name,
                    )
        direct, spreads = object_cache[name]
        resolved: dict[str, str] = {}
        dependencies: set[str] = set()
        for spread in spreads:
            spread_values, spread_dependencies = resolved_catalog(
                spread,
                stack + (name,),
            )
            resolved.update(spread_values)
            dependencies.add(spread)
            dependencies.update(spread_dependencies)
        resolved.update(direct)
        return resolved, dependencies

    try:
        resolved_english, _english_dependencies = resolved_catalog(
            "ENGLISH_MESSAGES"
        )
    except ValueError as exc:
        findings.append(
            _finding(
                code="DQ-I18N-PARSE",
                message=f"cannot resolve English catalog: {exc}",
                project_root=project_root,
                path=catalog_path,
            )
        )
        resolved_english = english
    english_keys = set(resolved_english)
    for locale in EXPECTED_LOCALES:
        variable = mappings.get(locale)
        if variable is None:
            findings.append(
                _finding(
                    code="DQ-I18N-CATALOG-MAP",
                    message=f"MESSAGE_CATALOGS does not map locale {locale}",
                    project_root=project_root,
                    path=catalog_path,
                )
            )
            continue
        try:
            values, dependencies = resolved_catalog(variable)
        except ValueError as exc:
            findings.append(
                _finding(
                    code="DQ-I18N-PARSE",
                    message=f"cannot resolve {locale} catalog: {exc}",
                    project_root=project_root,
                    path=catalog_path,
                )
            )
            continue
        if locale != "en" and (
            variable == "ENGLISH_MESSAGES" or "ENGLISH_MESSAGES" in dependencies
        ):
            findings.append(
                _finding(
                    code="DQ-I18N-ENGLISH-SPREAD-FALLBACK",
                    message=f"{locale} must not inherit untranslated English messages",
                    project_root=project_root,
                    path=catalog_path,
                    evidence={"locale": locale},
                )
            )

        keys = set(values)
        missing = sorted(english_keys - keys)
        if missing:
            findings.append(
                _finding(
                    code="DQ-I18N-KEY-MISSING",
                    message=f"{locale} is missing message keys: {', '.join(missing)}",
                    project_root=project_root,
                    path=catalog_path,
                    evidence={"locale": locale, "missingCount": len(missing)},
                )
            )
        extra = sorted(keys - english_keys)
        if extra:
            findings.append(
                _finding(
                    code="DQ-I18N-KEY-EXTRA",
                    message=f"{locale} has unknown message keys: {', '.join(extra)}",
                    project_root=project_root,
                    path=catalog_path,
                    evidence={"locale": locale, "extraCount": len(extra)},
                )
            )
        for key in sorted(english_keys & keys):
            value = values[key]
            if not value.strip():
                findings.append(
                    _finding(
                        code="DQ-I18N-EMPTY",
                        message=f"{locale}.{key} must not be empty",
                        project_root=project_root,
                        path=catalog_path,
                        evidence={"locale": locale, "key": key},
                    )
                )
                continue
            expected_placeholders = Counter(
                PLACEHOLDER_PATTERN.findall(resolved_english[key])
            )
            actual_placeholders = Counter(PLACEHOLDER_PATTERN.findall(value))
            if actual_placeholders != expected_placeholders:
                findings.append(
                    _finding(
                        code="DQ-I18N-PLACEHOLDER",
                        message=(
                            f"{locale}.{key} placeholders must match English; "
                            f"expected {dict(expected_placeholders)!r}, "
                            f"found {dict(actual_placeholders)!r}"
                        ),
                        project_root=project_root,
                        path=catalog_path,
                        evidence={"locale": locale, "key": key},
                    )
                )

    if re.search(
        r"\btype\s+MessageCatalog\s*=\s*Partial\s*<",
        source,
    ):
        findings.append(
            _finding(
                code="DQ-I18N-PARTIAL-CATALOG",
                message=(
                    "MessageCatalog must be total; Partial<Record<...>> permits "
                    "silent English fallback"
                ),
                project_root=project_root,
                path=catalog_path,
            )
        )

    fallback_patterns = (
        r"\blocalized\s*\?\?\s*english\b",
        r"MESSAGE_CATALOGS\s*\[\s*locale\s*\]\s*\[\s*key\s*\]\s*\?\?",
        r"\|\|\s*(?:english|ENGLISH_MESSAGES|MESSAGE_CATALOGS(?:\.en|\[.en.\]))",
    )
    for pattern in fallback_patterns:
        match = re.search(pattern, core, re.I)
        if match is not None:
            findings.append(
                _finding(
                    code="DQ-I18N-ENGLISH-FALLBACK",
                    message=(
                        "non-English locales must fail closed on a missing "
                        "message instead of falling back to English"
                    ),
                    project_root=project_root,
                    path=core_path,
                    line=core.count("\n", 0, match.start()) + 1,
                )
            )
            break
    return findings


def _hierarchy_findings(
    project_root: Path,
    app_path: Path,
    app: str,
    overview_path: Path,
    overview: str,
) -> list[Finding]:
    findings: list[Finding] = []
    union_match = re.search(
        r"export\s+type\s+OverviewRoom\s*=\s*([^;]+);",
        overview,
    )
    rooms = (
        tuple(re.findall(r'["\']([^"\']+)["\']', union_match.group(1)))
        if union_match
        else ()
    )
    if rooms != ("home", "speakers", "quality", "pipeline"):
        findings.append(
            _finding(
                code="DQ-HIERARCHY-ROOM-TYPE",
                message="OverviewRoom must define home plus speakers/quality/pipeline",
                project_root=project_root,
                path=overview_path,
            )
        )

    room_block = re.search(
        r"ROOM_DEFINITIONS[^=]*=\s*\[(.*?)\]\s*;",
        overview,
        re.S,
    )
    definition_ids = (
        tuple(re.findall(r'id\s*:\s*["\']([^"\']+)["\']', room_block.group(1)))
        if room_block
        else ()
    )
    if definition_ids != ("speakers", "quality", "pipeline"):
        findings.append(
            _finding(
                code="DQ-HIERARCHY-ROOM-DEFINITIONS",
                message="room definitions must be speakers, quality, pipeline in order",
                project_root=project_root,
                path=overview_path,
            )
        )

    required_overview_markers = (
        'if (room === "home")',
        'className="studio-hub"',
        'className="studio-room-grid studio-room-portals"',
        "studio-room-portal studio-room-portal--",
        'className="studio-room-portal__track"',
        'className="studio-room-header__breadcrumb"',
        'onClick={() => onRoomChange("home")}',
        '{room === "speakers" ? speakerStudio : null}',
        '{room === "quality" ? qualityLab : null}',
        '{room === "pipeline" ? pipelineObservatory : null}',
    )
    for marker in required_overview_markers:
        if marker not in overview:
            findings.append(
                _finding(
                    code="DQ-HIERARCHY-MARKER",
                    message=f"multi-level room contract is missing {marker!r}",
                    project_root=project_root,
                    path=overview_path,
                )
            )

    if len(re.findall(r"<h1\b", overview)) != 1:
        findings.append(
            _finding(
                code="DQ-HIERARCHY-H1",
                message="the focused room branch must contain exactly one h1",
                project_root=project_root,
                path=overview_path,
            )
        )
    if not re.search(r'if \(room === "home"\).*?<h2\b', overview, re.S):
        findings.append(
            _finding(
                code="DQ-HIERARCHY-HUB-HEADING",
                message="the overview hub must use h2 below the TaskHero h1",
                project_root=project_root,
                path=overview_path,
            )
        )

    required_app_markers = (
        'const [overviewRoom, setOverviewRoom] = useState<OverviewRoom>("home")',
        'studio.activeSection === "overview"',
        "room={overviewRoom}",
        "speakerStudio={",
        "qualityLab={",
        "pipelineObservatory={",
    )
    for marker in required_app_markers:
        if marker not in app:
            findings.append(
                _finding(
                    code="DQ-HIERARCHY-APP-WIRING",
                    message=f"App room wiring is missing {marker!r}",
                    project_root=project_root,
                    path=app_path,
                )
            )
    return findings


def _overview_structure_findings(
    project_root: Path,
    overview_path: Path,
    overview: str,
    css_path: Path,
    css: str,
) -> list[Finding]:
    findings: list[Finding] = []
    required_markup = (
        'className="studio-room-grid studio-room-portals"',
        "studio-room-portal studio-room-portal--",
        'className="studio-room-portal__track"',
        'className="studio-room-portal__footer"',
    )
    for marker in required_markup:
        if marker not in overview:
            findings.append(
                _finding(
                    code="DQ-STRUCTURE-PORTAL-MARKUP",
                    message=f"overview portal workspace is missing {marker!r}",
                    project_root=project_root,
                    path=overview_path,
                )
            )
    if "studio-room-card" in overview:
        findings.append(
            _finding(
                code="DQ-STRUCTURE-CARD-WALL",
                message=(
                    "OverviewWorkspace must not render independent "
                    "studio-room-card surfaces"
                ),
                project_root=project_root,
                path=overview_path,
            )
        )

    try:
        rules = parse_css(css)
    except ValueError as exc:
        return findings + [
            _finding(
                code="DQ-CSS-PARSE",
                message=f"overview structure CSS could not be parsed: {exc}",
                project_root=project_root,
                path=css_path,
            )
        ]

    workspace = _cascaded_declarations(
        rules,
        ".studio-room-grid",
        base_only=True,
    )
    portal = _cascaded_declarations(
        rules,
        ".studio-room-portal",
        base_only=True,
    )
    workspace_normalized = _normalized_declarations(workspace)
    portal_normalized = _normalized_declarations(portal)

    workspace_markers = (
        "gap:0",
        "border:",
        "background:",
        "backdrop-filter:",
    )
    for marker in workspace_markers:
        if marker not in workspace_normalized:
            findings.append(
                _finding(
                    code="DQ-STRUCTURE-WORKSPACE",
                    message=(
                        "studio-room-grid must be one continuous frosted "
                        f"workspace; missing {marker}"
                    ),
                    project_root=project_root,
                    path=css_path,
                )
            )
    if not any(
        marker in workspace_normalized
        for marker in ("overflow:clip", "overflow:hidden")
    ):
        findings.append(
            _finding(
                code="DQ-STRUCTURE-WORKSPACE",
                message="studio-room-grid must clip portal decoration as one surface",
                project_root=project_root,
                path=css_path,
            )
        )

    portal_markers = (
        "border-radius:0",
        "background:transparent",
        "box-shadow:none",
    )
    for marker in portal_markers:
        if marker not in portal_normalized:
            findings.append(
                _finding(
                    code="DQ-STRUCTURE-PORTAL-SURFACE",
                    message=(
                        "studio-room-portal must remain a segment of the shared "
                        f"workspace; missing {marker}"
                    ),
                    project_root=project_root,
                    path=css_path,
                )
            )
    if not any(
        marker in portal_normalized
        for marker in ("border-right:", "border-bottom:", "border-left:")
    ):
        findings.append(
            _finding(
                code="DQ-STRUCTURE-DIVIDER",
                message="studio-room-portal requires an explicit shared divider",
                project_root=project_root,
                path=css_path,
            )
        )

    track = _cascaded_declarations(
        rules,
        ".studio-room-portal__track",
        base_only=True,
    )
    if not {"position", "inset", "pointer-events"}.issubset(track):
        findings.append(
            _finding(
                code="DQ-STRUCTURE-PORTAL-TRACK",
                message=(
                    "portal track must be a non-interactive continuous depth "
                    "layer with position/inset/pointer-events"
                ),
                project_root=project_root,
                path=css_path,
            )
        )
    return findings


def validate_screenshot_evidence(
    project_root: Path,
    manifest_path: Path,
) -> Mapping[str, Any]:
    return validate_screenshot_document(
        project_root=project_root,
        manifest_path=manifest_path,
        schema_path=project_root
        / "contracts/design-quality-screenshot-evidence.schema.json",
    )


def validate_ocr_evidence(
    project_root: Path,
    manifest_path: Path,
) -> Mapping[str, Any]:
    return validate_ocr_document(
        project_root=project_root,
        manifest_path=manifest_path,
        schema_path=project_root / "contracts/design-quality-ocr-result.schema.json",
    )


def _run_check(
    check_id: str,
    action: Callable[[], Iterable[Finding]],
    *,
    project_root: Path,
) -> CheckResult:
    try:
        findings = tuple(action())
    except Exception as exc:  # fail closed at every local check boundary
        findings = (
            Finding(
                code="DQ-INTERNAL-FAIL-CLOSED",
                message=f"{check_id} could not complete deterministically: {exc}",
                path=".",
                evidence={"exceptionType": type(exc).__name__},
            ),
        )
    return CheckResult(
        id=check_id,
        status="pass" if not findings else "fail",
        findings=findings,
    )


def audit_project(
    project_root: Path | str,
    *,
    screenshot_manifest: Path | str | None = None,
    ocr_manifest: Path | str | None = None,
    require_evidence: bool = False,
) -> AuditReport:
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"project root is not a directory: {root}")

    css_path, css = _read_required(root, "apps/desktop/src/styles/global.css")
    tokens_path, tokens = _read_required(root, "apps/desktop/src/styles/tokens.css")
    app_path, app = _read_required(root, "apps/desktop/src/App.tsx")
    overview_path, overview = _read_required(
        root,
        "apps/desktop/src/components/OverviewWorkspace.tsx",
    )
    catalog_path, catalog = _read_required(
        root,
        "apps/desktop/src/i18n/catalog.ts",
    )
    core_path, core = _read_required(
        root,
        "apps/desktop/src/i18n/core.ts",
    )
    backdrop_path, backdrop = _read_required(
        root,
        "apps/desktop/src/components/SceneBackdrop.tsx",
    )
    backdrop_css_path, backdrop_css = _read_required(
        root,
        "apps/desktop/src/components/SceneBackdrop.css",
    )
    companion_path, companion = _read_required(
        root,
        "apps/desktop/src/components/DesktopCompanion.tsx",
    )
    companion_css_path, companion_css = _read_required(
        root,
        "apps/desktop/src/components/DesktopCompanion.css",
    )
    imported_css, css_graph_findings = _discover_imported_css(root)

    def motion_check() -> Iterable[Finding]:
        for source in imported_css:
            yield from _motion_findings(root, source.path, source.source)

    checks = [
        _run_check(
            "css-source-graph",
            lambda: css_graph_findings,
            project_root=root,
        ),
        _run_check(
            "motion",
            motion_check,
            project_root=root,
        ),
        _run_check(
            "frequent-navigation",
            lambda: _navigation_findings(root, css_path, css, app_path, app),
            project_root=root,
        ),
        _run_check(
            "forced-colors",
            lambda: _forced_colors_findings(root, css_path, css),
            project_root=root,
        ),
        _run_check(
            "background-assets",
            lambda: _background_findings(
                root,
                app_path,
                app,
                backdrop_path,
                backdrop,
                backdrop_css_path,
                backdrop_css,
            ),
            project_root=root,
        ),
        _run_check(
            "desktop-companion",
            lambda: _desktop_companion_findings(
                root,
                app_path,
                app,
                companion_path,
                companion,
                companion_css_path,
                companion_css,
            ),
            project_root=root,
        ),
        _run_check(
            "glass-and-palette",
            lambda: _glass_findings(
                root,
                tokens_path,
                tokens,
                css_path,
                css,
            ),
            project_root=root,
        ),
        _run_check(
            "page-hierarchy",
            lambda: _hierarchy_findings(
                root,
                app_path,
                app,
                overview_path,
                overview,
            ),
            project_root=root,
        ),
        _run_check(
            "overview-structure",
            lambda: _overview_structure_findings(
                root,
                overview_path,
                overview,
                css_path,
                css,
            ),
            project_root=root,
        ),
        _run_check(
            "locales",
            lambda: _locale_findings(
                root,
                catalog_path,
                catalog,
                core_path,
                core,
            ),
            project_root=root,
        ),
    ]

    screenshot_path = Path(screenshot_manifest) if screenshot_manifest else None
    ocr_path = Path(ocr_manifest) if ocr_manifest else None

    def screenshot_check() -> Iterable[Finding]:
        if screenshot_path is None:
            if require_evidence:
                return (
                    Finding(
                        code="DQ-SCREENSHOT-EVIDENCE-MISSING",
                        message="release audit requires a real native screenshot manifest",
                        path=".",
                    ),
                )
            return ()
        try:
            validate_screenshot_evidence(root, screenshot_path)
        except EvidenceValidationError as exc:
            return (
                Finding(
                    code="DQ-SCREENSHOT-EVIDENCE",
                    message=str(exc),
                    path=_relative(root, screenshot_path),
                ),
            )
        return ()

    def ocr_check() -> Iterable[Finding]:
        if ocr_path is None:
            if require_evidence:
                return (
                    Finding(
                        code="DQ-OCR-EVIDENCE-MISSING",
                        message="release audit requires real background-asset OCR results",
                        path=".",
                    ),
                )
            return ()
        try:
            validate_ocr_evidence(root, ocr_path)
        except EvidenceValidationError as exc:
            return (
                Finding(
                    code="DQ-OCR-EVIDENCE",
                    message=str(exc),
                    path=_relative(root, ocr_path),
                ),
            )
        return ()

    checks.extend(
        (
            _run_check("native-screenshot-evidence", screenshot_check, project_root=root),
            _run_check("background-ocr-evidence", ocr_check, project_root=root),
        )
    )
    return AuditReport(project_root=str(root), checks=tuple(checks))


def report_json(report: AuditReport) -> str:
    return json.dumps(
        report.as_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
