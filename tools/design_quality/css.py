"""Small deterministic CSS reader used by the design-quality gate.

The validator intentionally does not depend on a browser or Node package.  It
supports the CSS constructs used by the desktop application and fails closed
when a motion declaration cannot be interpreted.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class CssDeclaration:
    name: str
    value: str
    line: int


@dataclass(frozen=True)
class CssRule:
    selector: str
    declarations: tuple[CssDeclaration, ...]
    contexts: tuple[str, ...]
    line: int
    keyframes: str | None = None

    def values(self, name: str) -> tuple[str, ...]:
        normalized = name.casefold()
        return tuple(
            declaration.value
            for declaration in self.declarations
            if declaration.name.casefold() == normalized
        )


def _mask_comments(source: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    return re.sub(r"/\*.*?\*/", replace, source, flags=re.DOTALL)


def _line_number(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def _find_matching_brace(source: str, opening: int, limit: int) -> int:
    depth = 1
    quote: str | None = None
    escaped = False
    index = opening + 1
    while index < limit:
        char = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError(
        f"unclosed CSS block beginning on line {_line_number(source, opening)}"
    )


def split_top_level(value: str, separator: str = ",") -> tuple[str, ...]:
    """Split a CSS value without breaking functions or quoted strings."""

    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == separator and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return tuple(part for part in parts if part)


def _find_top_level_colon(value: str) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            return index
    return -1


def _parse_declarations(
    source: str,
    body_start: int,
    body_end: int,
) -> tuple[CssDeclaration, ...]:
    declarations: list[CssDeclaration] = []
    body = source[body_start:body_end]
    segments = split_top_level(body, separator=";")
    search_from = body_start
    for segment in segments:
        stripped = segment.strip()
        if not stripped:
            continue
        relative = source.find(stripped, search_from, body_end)
        if relative < 0:
            relative = search_from
        search_from = relative + len(stripped)
        colon = _find_top_level_colon(stripped)
        if colon <= 0:
            raise ValueError(
                "malformed CSS declaration on line "
                f"{_line_number(source, relative)}: {stripped[:80]}"
            )
        name = stripped[:colon].strip()
        value = stripped[colon + 1 :].strip()
        if not name or not value:
            raise ValueError(
                "empty CSS declaration on line "
                f"{_line_number(source, relative)}: {stripped[:80]}"
            )
        declarations.append(
            CssDeclaration(
                name=name,
                value=value,
                line=_line_number(source, relative),
            )
        )
    return tuple(declarations)


def parse_css(source: str) -> tuple[CssRule, ...]:
    """Parse style rules, media/support contexts, and keyframe frames."""

    masked = _mask_comments(source)
    rules: list[CssRule] = []

    def walk(
        start: int,
        end: int,
        contexts: tuple[str, ...],
        keyframes: str | None,
    ) -> None:
        cursor = start
        while cursor < end:
            while cursor < end and (masked[cursor].isspace() or masked[cursor] == ";"):
                cursor += 1
            if cursor >= end:
                return

            header_start = cursor
            quote: str | None = None
            escaped = False
            paren_depth = 0
            bracket_depth = 0
            opening = -1
            while cursor < end:
                char = masked[cursor]
                if quote is not None:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == quote:
                        quote = None
                elif char in {'"', "'"}:
                    quote = char
                elif char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth = max(0, paren_depth - 1)
                elif char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth = max(0, bracket_depth - 1)
                elif char == "{" and paren_depth == 0 and bracket_depth == 0:
                    opening = cursor
                    break
                elif char == ";" and paren_depth == 0 and bracket_depth == 0:
                    # A blockless at-rule such as @charset or @import.
                    cursor += 1
                    break
                cursor += 1

            if opening < 0:
                if cursor >= end:
                    trailing = masked[header_start:end].strip()
                    if trailing:
                        raise ValueError(
                            "unexpected trailing CSS on line "
                            f"{_line_number(masked, header_start)}: {trailing[:80]}"
                        )
                continue

            header = masked[header_start:opening].strip()
            if not header:
                raise ValueError(
                    f"empty CSS rule on line {_line_number(masked, header_start)}"
                )
            closing = _find_matching_brace(masked, opening, end)
            body_start = opening + 1
            lowered = header.casefold()

            if lowered.startswith("@media ") or lowered.startswith("@supports "):
                walk(body_start, closing, contexts + (header,), keyframes)
            elif re.match(r"^@(?:-[a-z]+-)?keyframes\s+", lowered):
                name = header.split(None, 1)[1].strip()
                if not name:
                    raise ValueError(
                        f"unnamed keyframes on line {_line_number(masked, header_start)}"
                    )
                walk(body_start, closing, contexts, name)
            elif lowered.startswith("@") and "{" in masked[body_start:closing]:
                # Handle grouping at-rules such as @layer without treating their
                # nested rules as declarations.
                walk(body_start, closing, contexts + (header,), keyframes)
            else:
                rules.append(
                    CssRule(
                        selector=header,
                        declarations=_parse_declarations(
                            masked,
                            body_start,
                            closing,
                        ),
                        contexts=contexts,
                        line=_line_number(masked, header_start),
                        keyframes=keyframes,
                    )
                )
            cursor = closing + 1

    walk(0, len(masked), (), None)
    return tuple(rules)


def selector_parts(selector: str) -> tuple[str, ...]:
    return split_top_level(selector)


def context_contains(rule: CssRule, fragment: str) -> bool:
    needle = re.sub(r"\s+", "", fragment.casefold())
    return any(
        needle in re.sub(r"\s+", "", context.casefold())
        for context in rule.contexts
    )


def iter_declarations(
    rules: Iterable[CssRule],
    *names: str,
) -> Iterable[tuple[CssRule, CssDeclaration]]:
    wanted = {name.casefold() for name in names}
    for rule in rules:
        for declaration in rule.declarations:
            if declaration.name.casefold() in wanted:
                yield rule, declaration
