"""
Regression tests for exception graphs that are not trees.

``__cause__``/``__context__`` form an arbitrary directed graph, not a tree:

- ``raise e from e`` makes an exception its own cause, and cycles can equally
  span several exceptions. Traversing those recursively never terminates.
- ``raise X from Y`` sets *both* ``X.__cause__`` and ``X.__context__`` to ``Y``.
  Traversing that as a tree revisits the shared node down both edges, so a chain
  of ``n`` chained exceptions costs ``2**n`` — it terminates, but only after
  hours once ``n`` passes ~20, and it writes an equally oversized dump.

Both are fixed by memoizing on identity, so every exception is visited once and
the saved graph mirrors the original one.
"""

from __future__ import annotations

import json
import pickle
import sys
from io import BytesIO
from typing import TYPE_CHECKING

import pytest

from offline_debug import (
    ExceptionData,
    ExceptionGroupData,
    load_traceback,
    parse_traceback,
    save_traceback,
    walk_exception_data,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

MAX_NODES_PER_LINK = 2
DEEP_CHAIN_LINKS = 400


def roundtrip(exc: BaseException) -> BaseException:
    """Save an exception to a buffer and load it back without raising."""
    buffer = BytesIO()
    save_traceback(exc, buffer)
    buffer.seek(0)
    return load_traceback(buffer, should_raise=False)


def walk_nodes(data: ExceptionData) -> Iterator[ExceptionData]:
    """Yield every distinct node of a saved exception graph exactly once."""
    seen: set[int] = set()
    stack = [data]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(link for link in (node.cause, node.context) if link is not None)


def self_caused() -> BaseException:
    """Return an exception produced by ``raise e from e``."""
    try:
        try:
            msg = "original"
            raise ValueError(msg)
        except ValueError as err:
            raise err from err
    except ValueError as err:
        return err


def test_raise_from_self_saves_and_loads() -> None:
    """``raise e from e`` must not recurse forever while being saved."""
    restored = roundtrip(self_caused())

    assert isinstance(restored, ValueError)
    assert str(restored) == "original"


def test_self_cause_stays_a_self_cause() -> None:
    """The cycle is preserved, not silently cut or unrolled into copies."""
    restored = roundtrip(self_caused())

    assert restored.__cause__ is restored


def test_mutual_cycle_is_preserved() -> None:
    """A two-exception cycle round-trips as the same two-exception cycle."""
    try:
        msg_a = "a"
        raise ValueError(msg_a)
    except ValueError as first:
        try:
            msg_b = "b"
            raise TypeError(msg_b)
        except TypeError as second:
            first.__cause__ = second
            second.__cause__ = first
            original = first

    restored = roundtrip(original)

    assert isinstance(restored, ValueError)
    assert isinstance(restored.__cause__, TypeError)
    assert restored.__cause__.__cause__ is restored


def test_self_context_saves_and_loads() -> None:
    """A self-referential ``__context__`` is handled just like ``__cause__``."""
    try:
        msg = "ctx"
        raise ValueError(msg)
    except ValueError as err:
        err.__context__ = err
        original = err

    restored = roundtrip(original)

    assert restored.__context__ is restored


def chain(links: int) -> BaseException:
    """Build ``links`` exceptions chained with the ordinary ``raise ... from ...``."""
    current: BaseException | None = None
    for i in range(links):
        try:
            if current is None:
                msg = "root"
                raise ValueError(msg)
            raise current
        except BaseException as prev:  # noqa: BLE001
            try:
                msg = f"lvl{i}"
                raise RuntimeError(msg) from prev
            except RuntimeError as wrapped:
                current = wrapped
    assert current is not None
    return current


def test_shared_cause_and_context_is_one_node() -> None:
    """
    ``raise X from Y`` aliases cause and context; the dump must not duplicate ``Y``.

    Duplicating it is what made a chain cost ``2**n`` to save and to load.
    """
    original = chain(1)
    assert original.__cause__ is original.__context__

    buffer = BytesIO()
    save_traceback(original, buffer)
    buffer.seek(0)
    data = parse_traceback(buffer)

    assert data.cause is data.context

    restored = roundtrip(original)
    assert restored.__cause__ is restored.__context__


def test_chained_exceptions_stay_linear() -> None:
    """A chain of ``n`` exceptions must produce ``O(n)`` nodes, not ``2**n``."""
    links = 16
    buffer = BytesIO()
    save_traceback(chain(links), buffer)
    buffer.seek(0)

    node_count = sum(1 for _ in walk_nodes(parse_traceback(buffer)))

    assert node_count <= links * MAX_NODES_PER_LINK


def test_deep_chain_is_not_recursion_bound() -> None:
    """A long but acyclic cause chain must not exhaust the interpreter stack."""
    deepest: BaseException = ValueError("root")
    for i in range(DEEP_CHAIN_LINKS):
        nxt = RuntimeError(f"lvl{i}")
        nxt.__cause__ = deepest
        deepest = nxt

    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        restored = roundtrip(deepest)
    finally:
        sys.setrecursionlimit(original_limit)

    depth = 0
    curr: BaseException | None = restored
    while curr is not None:
        depth += 1
        curr = curr.__cause__
    assert depth == DEEP_CHAIN_LINKS + 1


def test_exception_group_with_cyclic_cause() -> None:
    """A group whose cause is itself round-trips with the cycle intact."""
    try:
        msg = "inner"
        raise ValueError(msg)
    except ValueError as inner:
        try:
            raise ExceptionGroup("group", [inner])
        except ExceptionGroup as group:
            group.__cause__ = group
            original = group

    restored = roundtrip(original)

    assert isinstance(restored, ExceptionGroup)
    assert restored.__cause__ is restored
    assert [type(e) for e in restored.exceptions] == [ValueError]


def test_exception_group_member_pointing_back_at_group() -> None:
    """A member whose cause is the enclosing group keeps that exact link."""
    try:
        msg = "inner"
        raise ValueError(msg)
    except ValueError as inner:
        try:
            raise ExceptionGroup("group", [inner])
        except ExceptionGroup as group:
            group.exceptions[0].__cause__ = group
            original = group

    restored = roundtrip(original)

    assert isinstance(restored, ExceptionGroup)
    assert restored.exceptions[0].__cause__ is restored


def test_self_containing_group_is_rejected() -> None:
    """
    A hand-crafted group listed among its own members is refused, not looped on.

    ``save_traceback`` cannot emit this shape, but a dump is just a pickle and can be
    edited. The members of a group have to exist before the group is rebuilt around
    them, so a group that (transitively) contains itself has no build order at all.
    """
    data = ExceptionGroupData(
        exc_pickle=pickle.dumps(ExceptionGroup("group", [ValueError("inner")])),
        tb_frames=[],
        exceptions=[],
    )
    data.exceptions.append(data)

    buffer = BytesIO()
    pickle.dump(data, buffer)
    buffer.seek(0)

    with pytest.raises(ValueError, match="contains itself"):
        load_traceback(buffer, should_raise=False)


def test_walk_visits_every_node_exactly_once() -> None:
    """The public traversal terminates on a cycle and yields each node once."""
    buffer = BytesIO()
    save_traceback(self_caused(), buffer)
    buffer.seek(0)
    data = parse_traceback(buffer)
    assert data.cause is data

    visited = list(walk_exception_data(data))

    assert visited == [data]


def test_walk_covers_causes_contexts_and_group_members() -> None:
    """Every reachable node is reported, including exception group members."""
    try:
        msg = "inner"
        raise ValueError(msg)
    except ValueError as inner:
        try:
            raise ExceptionGroup("group", [inner, TypeError("other")])
        except ExceptionGroup as group:
            original = group

    buffer = BytesIO()
    save_traceback(original, buffer)
    buffer.seek(0)
    data = parse_traceback(buffer)

    visited = list(walk_exception_data(data))
    assert data in visited
    assert len({id(node) for node in visited}) == len(visited)
    assert isinstance(data, ExceptionGroupData)
    for member in data.exceptions:
        assert any(node is member for node in visited)


def test_walk_enables_a_json_safe_projection() -> None:
    """A cyclic graph can be projected to JSON as ids and references."""
    buffer = BytesIO()
    save_traceback(self_caused(), buffer)
    buffer.seek(0)
    data = parse_traceback(buffer)

    ids = {id(node): i for i, node in enumerate(walk_exception_data(data))}
    payload = [
        {
            "id": ids[id(node)],
            "cause_id": ids[id(node.cause)] if node.cause is not None else None,
        }
        for node in walk_exception_data(data)
    ]

    # The self-cause survives as a reference rather than defeating serialization.
    assert json.loads(json.dumps(payload)) == [{"id": 0, "cause_id": 0}]


def test_walk_is_not_recursion_bound() -> None:
    """Walking a long chain must not exhaust the interpreter stack."""
    deepest: BaseException = ValueError("root")
    for i in range(DEEP_CHAIN_LINKS):
        nxt = RuntimeError(f"lvl{i}")
        nxt.__cause__ = deepest
        deepest = nxt

    buffer = BytesIO()
    save_traceback(deepest, buffer)
    buffer.seek(0)
    data = parse_traceback(buffer)

    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        count = sum(1 for _ in walk_exception_data(data))
    finally:
        sys.setrecursionlimit(original_limit)

    assert count == DEEP_CHAIN_LINKS + 1
