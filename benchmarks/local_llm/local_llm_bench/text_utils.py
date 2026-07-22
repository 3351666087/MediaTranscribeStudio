from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_PROTECTED_TOKEN_RE = re.compile(
    r"(?<![\w])(?:[A-Za-z]+(?:[A-Za-z0-9._:/+-]*[A-Za-z0-9])?|\d+(?:[.,:/-]\d+)*)(?![\w])"
)
_PUNCTUATION_ONLY_RE = re.compile(r"^[\s，。！？；：、,.!?;:'\"“”‘’（）()\-\u2014…]*$")


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row_index, right_char in enumerate(right, start=1):
        current = [row_index]
        for column_index, left_char in enumerate(left, start=1):
            insertion = current[column_index - 1] + 1
            deletion = previous[column_index] + 1
            substitution = previous[column_index - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def normalized_distance(left: str, right: str) -> float:
    denominator = max(len(left), len(right), 1)
    return levenshtein_distance(left, right) / denominator


def extract_cjk(text: str) -> str:
    return "".join(_CJK_RE.findall(text))


def lcs_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_char in right:
        current = [0]
        for index, left_char in enumerate(left, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def cjk_retention(source: str, output: str) -> float:
    source_cjk = extract_cjk(source)
    if not source_cjk:
        return 1.0
    return lcs_length(source_cjk, extract_cjk(output)) / len(source_cjk)


def protected_tokens(text: str) -> Counter[str]:
    return Counter(match.group(0).casefold() for match in _PROTECTED_TOKEN_RE.finditer(text))


def protected_tokens_preserved(source: str, output: str) -> bool:
    source_tokens = protected_tokens(source)
    output_tokens = protected_tokens(output)
    return all(output_tokens[token] >= count for token, count in source_tokens.items())


def changed_source_spans(source: str, revised: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    matcher = SequenceMatcher(a=source, b=revised, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag != "equal":
            spans.append((i1, i2))
    return spans


def span_overlap_score(
    predicted: Iterable[tuple[int, int]],
    expected: Iterable[tuple[int, int]],
) -> tuple[float, float, float]:
    predicted_set = _expand_spans(predicted)
    expected_set = _expand_spans(expected)
    if not predicted_set and not expected_set:
        return 1.0, 1.0, 1.0
    if not predicted_set:
        return 0.0, 0.0, 0.0
    if not expected_set:
        return 0.0, 0.0, 0.0
    overlap = len(predicted_set & expected_set)
    precision = overlap / len(predicted_set)
    recall = overlap / len(expected_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _expand_spans(spans: Iterable[tuple[int, int]]) -> set[int]:
    expanded: set[int] = set()
    for start, end in spans:
        if start == end:
            expanded.add(start)
        else:
            expanded.update(range(start, end))
    return expanded


def is_punctuation_only_change(source: str, output: str) -> bool:
    matcher = SequenceMatcher(a=source, b=output, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if not _PUNCTUATION_ONLY_RE.fullmatch(source[i1:i2] + output[j1:j2]):
            return False
    return source != output
