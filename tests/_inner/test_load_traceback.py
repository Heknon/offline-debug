"""Tests for the load_traceback module."""

from __future__ import annotations

import pickle
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Never

import pytest

from offline_debug import FrameData, load_traceback, save_traceback
from offline_debug._inner.load_traceback import _original_position
from offline_debug._inner.synthetic_code import synthetic_code
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


def _frame_data(lasti: int, lineno: int) -> FrameData:
    """Build the parts of a ``FrameData`` that ``_original_position`` reads."""
    return FrameData(code=b"", globals={}, locals={}, lasti=lasti, lineno=lineno, stack_depth=1)


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator


def _failing_instruction() -> tuple[types.CodeType, int, int]:
    """Return ``_divide``'s code with the ``lasti`` and line of its division."""
    try:
        _divide(1, 0)
    except ZeroDivisionError as err:
        tb = err.__traceback__
        assert tb is not None
        tb = tb.tb_next
        assert tb is not None
        return tb.tb_frame.f_code, tb.tb_lasti, tb.tb_lineno
    msg = "_divide() did not raise"
    raise AssertionError(msg)


def test_original_position_recovers_the_columns_of_the_failing_instruction() -> None:
    """The saved ``lasti`` picks the original instruction, columns included."""
    code, lasti, lineno = _failing_instruction()
    source = "    return numerator / denominator"

    line, end_line, col, end_col = _original_position(code, _frame_data(lasti, lineno))

    assert (line, end_line) == (lineno, lineno)
    assert source[col:end_col] == "numerator / denominator"


@pytest.mark.parametrize("lasti", [-1, 10_000])
def test_original_position_falls_back_for_an_unusable_offset(lasti: int) -> None:
    """An offset the original bytecode cannot resolve yields the line alone."""
    code, _, lineno = _failing_instruction()

    assert _original_position(code, _frame_data(lasti, lineno)) == (lineno, lineno, None, None)


def test_original_position_falls_back_when_the_instruction_has_no_columns() -> None:
    """Code without column information (``-X no_debug_ranges``) yields the line alone."""
    code, _, lineno = _failing_instruction()
    columnless, lasti = synthetic_code(
        code.co_filename, code.co_name, code.co_qualname, (lineno, lineno, None, None)
    )

    assert _original_position(columnless, _frame_data(lasti, lineno)) == (
        lineno,
        lineno,
        None,
        None,
    )


def test_original_position_falls_back_when_the_line_disagrees() -> None:
    """A position on another line than the one recorded is not trusted."""
    code, lasti, lineno = _failing_instruction()

    assert _original_position(code, _frame_data(lasti, lineno + 1)) == (
        lineno + 1,
        lineno + 1,
        None,
        None,
    )


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
