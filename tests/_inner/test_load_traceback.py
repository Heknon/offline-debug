"""Tests for the load_traceback module."""

from __future__ import annotations

import pickle
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Never

import pytest

from offline_debug import load_traceback, save_traceback
from offline_debug._inner.load_traceback import _MAX_UNITS_PER_ENTRY, _line_only_linetable
from tests.helpers import roundtrip

if TYPE_CHECKING:
    import types

    from offline_debug import ExceptionData


def get_frames(tb: types.TracebackType | None) -> list[types.FrameType]:
    """Extract all frames from a traceback."""
    frames = []
    curr = tb
    while curr:
        frames.append(curr.tb_frame)
        curr = curr.tb_next
    return frames


def test_load_invalid_object(tmp_path: Path) -> None:
    """Test that load_traceback raises TypeError when loading an invalid object."""
    import pickle

    dump_file = tmp_path / "invalid.dump"
    with dump_file.open("wb") as f:
        pickle.dump("not an ExceptionData object", f)

    with pytest.raises(TypeError, match="Expected _ExceptionData, but got str"):
        load_traceback(dump_file)


def test_load_non_existent_file() -> None:
    """Test that load_traceback raises FileNotFoundError when the file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_traceback(Path("non_existent_file.dump"))


def test_reconstruct_invalid_exception_type() -> None:
    """Test that _reconstruct_exc_data raises TypeError when the pickled exception is invalid."""
    import pickle

    from offline_debug._inner.load_traceback import _reconstruct_exc_data
    from offline_debug._inner.models import ExceptionData

    data = ExceptionData(
        exc_pickle=pickle.dumps("not an exception"),
        tb_frames=[],
    )

    with pytest.raises(TypeError, match="Expected BaseException, but got str"):
        _reconstruct_exc_data(data)


def test_reconstruct_invalid_frame_type(monkeypatch) -> None:
    """Test that _reconstruct_exc_data raises TypeError when frame creation fails."""
    import offline_debug._inner.c_api._create_frame as _create_frame_module
    from offline_debug._inner.load_traceback import _reconstruct_exc_data
    from offline_debug._inner.models import ExceptionData, FrameData

    # Mock _get_py_frame_new to return a function that returns something that is not a FrameType
    monkeypatch.setattr(_create_frame_module, "_get_py_frame_new", lambda: lambda *_: "not a frame")

    import marshal
    import pickle

    def dummy() -> None:
        pass

    data = ExceptionData(
        exc_pickle=pickle.dumps(ValueError("test")),
        tb_frames=[
            FrameData(
                code=marshal.dumps(dummy.__code__),
                globals={},
                locals={},
                lasti=0,
                lineno=0,
                stack_depth=0,
            )
        ],
    )

    with pytest.raises(TypeError, match=r"Expected types.FrameType, but got str"):
        _reconstruct_exc_data(data)


def test_load_traceback_should_raise_false() -> None:
    """Test load_traceback when should_raise is False."""
    buffer = BytesIO()
    exc = ValueError("test_raise_false")
    save_traceback(exc, buffer)
    buffer.seek(0)

    loaded_exc = load_traceback(buffer, should_raise=False)
    assert isinstance(loaded_exc, ValueError)
    assert str(loaded_exc) == "test_raise_false"


@pytest.mark.parametrize(
    ("firstlineno", "lineno"),
    [
        (1, 1),  # zero delta
        (10, 12),  # small positive delta: one varint byte
        (10, 3),  # negative delta: the sign bit
        (1, 100),  # delta beyond one 6-bit payload: a continuation byte
        (1, 70_000),  # several continuation bytes
    ],
)
def test_line_only_linetable_matches_cpython_decoding(firstlineno: int, lineno: int) -> None:
    """CPython must decode the hand-built table as "every instruction on ``lineno``"."""
    base = compile("", "<synthetic>", "exec")

    code = base.replace(
        co_firstlineno=firstlineno,
        co_linetable=_line_only_linetable(base, firstlineno, lineno),
    )

    units = len(code.co_code) // 2
    assert list(code.co_positions()) == [(lineno, lineno, None, None)] * units
    assert [line for _, _, line in code.co_lines()] == [lineno]


def test_line_only_linetable_covers_code_longer_than_one_entry() -> None:
    """A code object needing several entries still maps every instruction."""
    code = compile("\n".join(f"v{i} = {i}" for i in range(20)), "<synthetic>", "exec")
    units = len(code.co_code) // 2
    assert units > _MAX_UNITS_PER_ENTRY

    code = code.replace(co_linetable=_line_only_linetable(code, code.co_firstlineno, 42))

    assert list(code.co_positions()) == [(42, 42, None, None)] * units


def _raise_on_load() -> Never:
    msg = "cannot load"
    raise TypeError(msg)


class LoadFailGroup(ExceptionGroup):
    """Group that pickles fine but whose reconstruction fails at load time."""

    def __reduce__(self) -> tuple:
        """Reduce to a callable that raises when the pickle is loaded."""
        return (_raise_on_load, ())


class LoadFailError(Exception):
    """Exception that pickles fine but whose reconstruction fails at load time."""

    def __reduce__(self) -> tuple:
        """Reduce to a callable that raises when the pickle is loaded."""
        return (_raise_on_load, ())


def record_rebuilt_exceptions(monkeypatch) -> list[str]:
    """Record the type name of every exception whose traceback the loader rebuilds."""
    import offline_debug._inner.load_traceback as load_module

    rebuilt: list[str] = []
    real_reconstruct_frames = load_module._reconstruct_frames

    def recording(data: ExceptionData) -> types.TracebackType | None:
        rebuilt.append(type(pickle.loads(data.exc_pickle)).__name__)  # noqa: S301
        return real_reconstruct_frames(data)

    monkeypatch.setattr(load_module, "_reconstruct_frames", recording)
    return rebuilt


def test_members_of_a_placeholder_group_are_not_rebuilt(monkeypatch) -> None:
    """
    A group that loads as the placeholder references none of its members.

    Rebuilding them anyway would create and link a real frame per traceback entry
    only to drop them, so the loader must not descend into them.
    """
    try:
        raise LoadFailGroup("group", [ValueError("a"), TypeError("b")])
    except LoadFailGroup as err:
        original = err
    rebuilt = record_rebuilt_exceptions(monkeypatch)

    restored = roundtrip(original)

    assert isinstance(restored, RuntimeError)
    assert "Unpicklable exception LoadFailGroup" in str(restored)
    assert rebuilt == ["RuntimeError"]


def test_placeholder_group_members_reached_by_a_link_are_still_rebuilt(monkeypatch) -> None:
    """Pruning must not drop a member that something else links to."""
    try:
        msg = "a"
        raise ValueError(msg)
    except ValueError as member:
        try:
            raise LoadFailGroup("group", [member, TypeError("b")])
        except LoadFailGroup as err:
            err.__cause__ = member
            original = err
    rebuilt = record_rebuilt_exceptions(monkeypatch)

    restored = roundtrip(original)

    assert isinstance(restored, RuntimeError)
    assert isinstance(restored.__cause__, ValueError)
    assert sorted(rebuilt) == ["RuntimeError", "ValueError"]


def test_group_survives_a_member_that_fails_only_on_load() -> None:
    """
    One member that cannot be loaded must not take its whole group down.

    The group's own pickle carries no members -- they are nodes of their own -- so a
    member's load failure is confined to that member's placeholder.
    """
    try:
        raise ExceptionGroup("group", [LoadFailError("bad"), ValueError("good")])
    except ExceptionGroup as err:
        original = err

    restored = roundtrip(original)

    assert isinstance(restored, ExceptionGroup)
    assert restored.message == "group"
    bad, good = restored.exceptions
    assert isinstance(bad, RuntimeError)
    assert "Unpicklable exception LoadFailError" in str(bad)
    assert isinstance(good, ValueError)
    assert str(good) == "good"
