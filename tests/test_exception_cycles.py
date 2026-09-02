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
from typing import Never

import pytest

from offline_debug import (
    ExceptionData,
    ExceptionGroupData,
    load_traceback,
    parse_traceback,
    save_traceback,
    walk_exception_data,
)
from tests.helpers import roundtrip

MAX_NODES_PER_LINK = 2
DEEP_CHAIN_LINKS = 400


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

    node_count = sum(1 for _ in walk_exception_data(parse_traceback(buffer)))

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


def unwrap_member() -> Never:
    """``except ExceptionGroup as g: raise g.exceptions[0]``, the common unwrap idiom."""
    try:
        try:
            msg = "inner"
            raise ValueError(msg)
        except ValueError as inner:
            raise ExceptionGroup("group", [inner])  # noqa: B904 - wrapping is the idiom
    except ExceptionGroup as group:
        raise group.exceptions[0]  # noqa: B904 - unwrapping is the idiom


def unwrapped_member() -> BaseException:
    """
    Return what ``unwrap_member`` raises: a member whose ``__context__`` is its own group.

    Raising the member out of the ``except`` block records the group as the member's
    context while the group still lists the member, so the exception graph's root is a
    member of a group it links to.
    """
    try:
        unwrap_member()
    except ValueError as err:
        member = err
    else:
        msg = "unwrap_member() did not raise"
        raise AssertionError(msg)
    assert isinstance(member.__context__, ExceptionGroup)
    assert member.__context__.exceptions[0] is member
    return member


def test_member_raised_out_of_its_group_loads() -> None:
    """
    The unwrap idiom must load, with the group rebuilt around the very same member.

    The member is the root, and the loader can only build its group after it -- however
    the group is reached.
    """
    restored = roundtrip(unwrapped_member())

    assert isinstance(restored, ValueError)
    assert isinstance(restored.__context__, ExceptionGroup)
    assert restored.__context__.exceptions[0] is restored


def test_member_raised_out_of_except_star_loads() -> None:
    """The ``except*`` form of the unwrap idiom saves the same member-first shape."""

    def unwrap() -> Never:
        try:
            msg = "inner"
            raise ExceptionGroup("group", [ValueError(msg)])
        except* ValueError as matched:
            raise matched.exceptions[0]  # noqa: B904 - unwrapping is the idiom

    try:
        unwrap()
    except ValueError as err:
        original = err
    assert isinstance(original.__context__, ExceptionGroup)
    assert original.__context__.exceptions[0] is original

    restored = roundtrip(original)

    assert isinstance(restored, ValueError)
    assert isinstance(restored.__context__, ExceptionGroup)
    assert restored.__context__.exceptions[0] is restored


def test_wrapped_unwrapped_member_loads() -> None:
    """The member-first shape must load when only a ``raise ... from`` link reaches it."""
    try:
        msg = "wrapper"
        raise RuntimeError(msg) from unwrapped_member()
    except RuntimeError as err:
        original = err

    restored = roundtrip(original)

    member = restored.__cause__
    assert isinstance(member, ValueError)
    assert isinstance(member.__context__, ExceptionGroup)
    assert member.__context__.exceptions[0] is member


def test_group_reached_through_a_member_link_is_built_after_the_root() -> None:
    """
    A link from inside a group's members may lead to a group that contains the root.

    That outer group can only be built once the root is, however the link is found.
    """
    try:
        msg = "member"
        raise ValueError(msg)
    except ValueError as member:
        try:
            raise ExceptionGroup("inner group", [member])
        except ExceptionGroup as inner_group:
            try:
                raise ExceptionGroup("outer group", [inner_group])
            except ExceptionGroup as outer_group:
                member.__cause__ = outer_group
                original = inner_group

    restored = roundtrip(original)

    assert isinstance(restored, ExceptionGroup)
    outer = restored.exceptions[0].__cause__
    assert isinstance(outer, ExceptionGroup)
    assert outer.exceptions[0] is restored


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


def test_group_containing_itself_is_rejected() -> None:
    """
    A group that contains itself cannot be rebuilt, and must say so.

    ``save_traceback`` cannot produce this shape -- a real ``BaseExceptionGroup``
    cannot hold itself -- but a hand-crafted or corrupted dump can. Members must
    exist before the group is built, so a self-membership edge has no valid build
    order and is reported rather than surfacing as a bare ``KeyError``.
    """
    group = ExceptionGroupData(
        exc_pickle=pickle.dumps(ExceptionGroup("group", [ValueError("a")])),
        tb_frames=[],
        exceptions=[],
    )
    group.exceptions.append(group)

    buffer = BytesIO()
    pickle.dump(group, buffer)
    buffer.seek(0)

    with pytest.raises(ValueError, match="contains itself"):
        load_traceback(buffer, should_raise=False)


def test_mutually_containing_groups_are_rejected() -> None:
    """Two groups holding each other have no valid build order either."""
    outer_group = ExceptionGroupData(
        exc_pickle=pickle.dumps(ExceptionGroup("outer", [ValueError("a")])),
        tb_frames=[],
        exceptions=[],
    )
    inner_group = ExceptionGroupData(
        exc_pickle=pickle.dumps(ExceptionGroup("inner", [ValueError("b")])),
        tb_frames=[],
        exceptions=[outer_group],
    )
    outer_group.exceptions.append(inner_group)

    buffer = BytesIO()
    pickle.dump(outer_group, buffer)
    buffer.seek(0)

    with pytest.raises(ValueError, match="contains itself"):
        load_traceback(buffer, should_raise=False)


def test_nodes_compare_by_identity() -> None:
    """
    Nodes must compare by identity, since structural equality cannot end on a cycle.

    A dataclass-generated ``__eq__`` compares fields recursively, so comparing two
    parses of a ``raise e from e`` dump would recurse through ``cause`` forever.
    """

    def parsed() -> ExceptionData:
        buffer = BytesIO()
        save_traceback(self_caused(), buffer)
        buffer.seek(0)
        return parse_traceback(buffer)

    first, second = parsed(), parsed()
    assert first.cause is first

    assert first == first  # noqa: PLR0124 - identity is the semantics under test
    assert first != second
    by_node = {first: "first", second: "second"}
    assert by_node[first] == "first"
    assert by_node[second] == "second"
