"""Regression tests for shared and cyclic exception links."""

from __future__ import annotations

import sys
from io import BytesIO

import pytest

from offline_debug import ExceptionData, ExceptionGroupData, load_traceback, save_traceback


def roundtrip(exc: BaseException) -> BaseException:
    buffer = BytesIO()
    save_traceback(exc, buffer)
    buffer.seek(0)
    return load_traceback(buffer, should_raise=False)


def node_count(root: ExceptionData) -> int:
    seen: set[int] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        pending.extend(edge for edge in (node.cause, node.context) if edge is not None)
        if isinstance(node, ExceptionGroupData):
            pending.extend(node.exceptions)
    return len(seen)


@pytest.mark.parametrize("link", ["__cause__", "__context__"])
def test_self_reference_roundtrips(link: str) -> None:
    original = ValueError("cycle")
    setattr(original, link, original)

    restored = roundtrip(original)

    assert getattr(restored, link) is restored


def test_mutual_cycle_roundtrips() -> None:
    first = ValueError("first")
    second = TypeError("second")
    first.__cause__ = second
    second.__context__ = first

    restored = roundtrip(first)

    assert isinstance(restored.__cause__, TypeError)
    assert restored.__cause__.__context__ is restored


def test_shared_cause_and_context_is_serialized_once() -> None:
    cause = ValueError("cause")
    original = RuntimeError("wrapper")
    original.__cause__ = cause
    original.__context__ = cause

    data = save_traceback(original, None)
    restored = roundtrip(original)

    assert data.cause is data.context
    assert restored.__cause__ is restored.__context__


def test_exception_chain_stays_linear() -> None:
    links = 12
    original: BaseException = ValueError("root")
    for _ in range(links):
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = original
        wrapper.__context__ = original
        original = wrapper

    assert node_count(save_traceback(original, None)) == links + 1


def test_deep_chain_is_not_recursion_bound() -> None:
    links = 400
    original: BaseException = ValueError("root")
    for _ in range(links):
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = original
        original = wrapper

    recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        restored = roundtrip(original)
    finally:
        sys.setrecursionlimit(recursion_limit)

    depth = 0
    while restored.__cause__ is not None:
        depth += 1
        restored = restored.__cause__
    assert depth == links


def test_group_member_can_link_back_to_its_group() -> None:
    member = ValueError("member")
    group = ExceptionGroup("group", [member])
    member.__context__ = group

    restored = roundtrip(member)

    assert isinstance(restored.__context__, ExceptionGroup)
    assert restored.__context__.exceptions[0] is restored
