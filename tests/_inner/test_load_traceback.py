"""Tests for the load_traceback module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from offline_debug import load_traceback

if TYPE_CHECKING:
    import types


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
