"""Tests for the load_traceback module."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from offline_debug import load_traceback, save_traceback

if TYPE_CHECKING:
    import types


class Vanished:
    """Class removed from this module before loading, like one that moved between versions."""

    def __init__(self, value: int) -> None:
        """Keep ``value`` as both a constructor argument and pickled state."""
        self.value = value

    def __reduce__(self) -> tuple:
        """Pickle as a call with arguments followed by state."""
        return Vanished, (self.value,), {"value": self.value}


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
    from offline_debug._inner.models import CodeData, ExceptionData, FrameData

    # Mock _get_py_frame_new to return a function that returns something that is not a FrameType
    monkeypatch.setattr(_create_frame_module, "_get_py_frame_new", lambda: lambda *_: "not a frame")

    import pickle

    data = ExceptionData(
        exc_pickle=pickle.dumps(ValueError("test")),
        tb_frames=[
            FrameData(
                code=CodeData(filename="dummy.py", name="dummy", qualname="dummy", firstlineno=1),
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


def test_missing_class_loads_as_stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """A value whose class cannot be imported must not make the whole dump unloadable."""
    buffer = BytesIO()

    def raise_with_vanished() -> None:
        _vanished = Vanished(1)
        msg = "vanished"
        raise ValueError(msg)

    try:
        raise_with_vanished()
    except ValueError as e:
        save_traceback(e, buffer)
    buffer.seek(0)
    monkeypatch.delattr(sys.modules[__name__], "Vanished")

    loaded_exc = load_traceback(buffer, should_raise=False)
    stand_in = get_frames(loaded_exc.__traceback__)[-1].f_locals["_vanished"]
    assert repr(stand_in) == f"<missing {__name__}.Vanished(1,)>"
    assert stand_in.value == 1
