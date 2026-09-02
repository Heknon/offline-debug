"""
Tests for traversal of the exception graph: cause, context and group members.

The graph is generally not a tree. ``raise X from Y`` inside ``except Y`` makes
``Y`` both the cause and the context of ``X``, and manual assignment can create
cycles (``raise e from e``). These tests pin down that every exception is
serialized once, that object identity survives a round-trip, and that cycles and
chains deeper than the recursion limit load without blowing up.
"""

from __future__ import annotations

import sys
from io import BytesIO
from typing import Never

import pytest

from offline_debug import ExceptionData, ExceptionGroupData, load_traceback, save_traceback


def _fail(exc: BaseException) -> Never:
    """Raise ``exc`` from a helper so tests can raise inside ``try`` bodies."""
    raise exc


def _round_trip(exc: BaseException) -> tuple[bytes, BaseException]:
    """Save ``exc`` to a buffer and load it back without raising."""
    buffer = BytesIO()
    save_traceback(exc, buffer)
    buffer.seek(0)
    return buffer.getvalue(), load_traceback(buffer, should_raise=False)


def _wrap_chain(depth: int) -> ValueError:
    """Build ``depth`` nested ``raise ... from e`` wrappers, each sharing cause and context."""

    def rec(n: int) -> Never:
        if n == 0:
            raise ValueError("leaf")
        try:
            rec(n - 1)
        except ValueError as e:
            msg = f"level {n}"
            raise ValueError(msg) from e

    with pytest.raises(ValueError, match=f"level {depth}") as exc_info:
        rec(depth)
    return exc_info.value


def _context_chain(depth: int) -> ValueError:
    """Build a linear ``__context__`` chain of ``depth`` exceptions, no cause links."""
    prev: ValueError | None = None
    for i in range(depth):
        try:
            _fail(ValueError(i))
        except ValueError as e:
            e.__context__ = prev
            prev = e
    assert prev is not None
    return prev


def _count_nodes(data: ExceptionData) -> int:
    """Count distinct ``ExceptionData`` objects reachable from ``data``."""
    seen: set[int] = set()
    pending = [data]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        pending.extend(n for n in (node.cause, node.context) if n is not None)
        if isinstance(node, ExceptionGroupData):
            pending.extend(node.exceptions)
    return len(seen)


def test_shared_cause_and_context_serialized_once() -> None:
    """``raise X from Y`` in ``except Y`` must serialize ``Y`` once, not twice per level."""
    depth = 12
    data = save_traceback(_wrap_chain(depth), None)

    assert _count_nodes(data) == depth + 1
    node: ExceptionData | None = data
    while node is not None and node.cause is not None:
        assert node.cause is node.context
        node = node.cause


def test_dump_size_grows_linearly_with_chain_depth() -> None:
    """A naive walk doubles the dump per level; the memoized walk stays linear."""
    small_depth, large_depth = 8, 24
    small, _ = _round_trip(_wrap_chain(small_depth))
    large, _ = _round_trip(_wrap_chain(large_depth))

    # Linear growth gives a ratio near 3; exponential growth would be ~2**16.
    ratio_upper_bound = 4
    assert len(large) < ratio_upper_bound * len(small)


def test_shared_cause_and_context_identity_after_load() -> None:
    depth = 5
    _, loaded = _round_trip(_wrap_chain(depth))

    node: BaseException | None = loaded
    levels = 0
    while node is not None and node.__cause__ is not None:
        assert node.__cause__ is node.__context__
        assert node.__suppress_context__
        node = node.__cause__
        levels += 1
    assert levels == depth


def _raise_from_self() -> Never:
    try:
        _fail(ValueError("inner"))
    except ValueError as e:
        raise e from e


def test_self_cause_cycle_round_trips() -> None:
    """``raise e from e`` points an exception's cause at itself."""
    with pytest.raises(ValueError, match="inner") as exc_info:
        _raise_from_self()
    exc = exc_info.value
    assert exc.__cause__ is exc
    # CPython never lets an exception become its own context, so only the cause cycles.
    assert exc.__context__ is None

    _, loaded = _round_trip(exc)
    assert loaded.__cause__ is loaded
    assert loaded.__context__ is None


def test_mutual_cause_cycle_round_trips() -> None:
    first = ValueError("first")
    second = KeyError("second")
    first.__cause__ = second
    second.__cause__ = first

    _, loaded = _round_trip(first)

    assert isinstance(loaded, ValueError)
    assert isinstance(loaded.__cause__, KeyError)
    assert loaded.__cause__.__cause__ is loaded


def test_group_member_context_pointing_at_group_round_trips() -> None:
    """A member whose context is its own group forms a cycle through ``exceptions``."""
    member = ValueError("member")
    group = ExceptionGroup("group", [member])
    member.__context__ = group

    data = save_traceback(group, None)
    assert isinstance(data, ExceptionGroupData)
    assert data.exceptions[0].context is data

    _, loaded = _round_trip(group)
    assert isinstance(loaded, ExceptionGroup)
    assert loaded.exceptions[0].__context__ is loaded


def test_group_members_share_context_between_them() -> None:
    """Members raised while handling one another keep their links to each other."""
    try:
        try:
            _fail(ValueError("first"))
        except ValueError as first:
            try:
                _fail(TypeError("second"))
            except TypeError as second:
                raise ExceptionGroup("group", [first, second]) from None
    except ExceptionGroup as group:
        _, loaded = _round_trip(group)

    assert isinstance(loaded, ExceptionGroup)
    first_loaded, second_loaded = loaded.exceptions
    assert second_loaded.__context__ is first_loaded
    assert loaded.__context__ is second_loaded
    assert loaded.__cause__ is None


def test_context_chain_deeper_than_recursion_limit() -> None:
    depth = sys.getrecursionlimit() * 2
    _, loaded = _round_trip(_context_chain(depth))

    length = 0
    node: BaseException | None = loaded
    while node is not None:
        length += 1
        node = node.__context__
    assert length == depth
