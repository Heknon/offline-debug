"""Tests for the link_frame and f_back discovery logic."""

from __future__ import annotations

import ctypes
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from offline_debug._inner.c_api import create_frame
from offline_debug._inner.c_api._link_frame import _get_f_back_offset

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def clear_f_back_offset_cache() -> Iterator[None]:
    _get_f_back_offset.cache_clear()
    yield
    _get_f_back_offset.cache_clear()


def test_get_f_back_offset_logic() -> None:
    """Test the dynamic f_back offset discovery logic directly."""
    offset = _get_f_back_offset()
    # It should either find an offset or be None (if platform is weird)
    # But on standard CPython it should find something.
    assert offset is None or (offset > 0 and offset % 8 == 0)


def test_link_frame_no_offset(monkeypatch) -> None:
    """Test that link_frame raises an exception if the f back offset wasn't found."""
    import offline_debug._inner.c_api._link_frame as _link_frame_module

    monkeypatch.setattr(_link_frame_module, "_get_f_back_offset", lambda: None)

    f = sys._getframe()
    with pytest.raises(RuntimeError, match="Failed discovering"):
        _link_frame_module.link_frame(f, f)


def test_get_f_back_offset_success() -> None:
    """Test successful discovery of f_back offset."""
    # We don't need to mock much here, just verify it returns a plausible offset.
    offset = _get_f_back_offset()
    assert isinstance(offset, int)
    assert offset > 0
    # On 64-bit CPython it's usually 16 or 24.
    assert offset % ctypes.sizeof(ctypes.c_void_p) == 0


def test_get_f_back_offset_not_a_frame() -> None:
    """Test when PyFrame_New returns something that is not a FrameType."""
    import offline_debug._inner.c_api._create_frame as _create_frame_module

    with patch.object(
        _create_frame_module, "_get_py_frame_new", return_value=lambda *_: "not a frame"
    ):
        offset = _get_f_back_offset()
        assert offset is None


def test_get_f_back_offset_exception_in_try() -> None:
    """Test when an exception occurs early in the discovery process."""
    import offline_debug._inner.c_api._create_frame as _create_frame_module

    mock_func = MagicMock(side_effect=RuntimeError("thread error"))
    with patch.object(_create_frame_module, "_get_py_thread_state_get", return_value=mock_func):
        offset = _get_f_back_offset()
        assert offset is None


def test_get_f_back_offset_discovery_failure() -> None:
    """Test when the discovery loop completes without finding the offset."""

    class MockValue:
        def __init__(self, val: int) -> None:
            self.value = val

    with patch("ctypes.c_ssize_t.from_address", return_value=MockValue(1)):
        offset = _get_f_back_offset()
        assert offset is None


def test_get_f_back_offset_wrong_offset_restoration() -> None:
    """A candidate slot that holds 0 but is not ``f_back`` must be restored to 0."""
    import offline_debug._inner.c_api._create_frame as _create_frame_module
    import offline_debug._inner.c_api._link_frame as _link_frame_module

    # Build the frame through the package's own helper rather than calling
    # ``ctypes.pythonapi.PyFrame_New`` directly. That function pointer is cached and
    # process-global, and only carries argtypes/restype once ``_get_py_frame_new`` has
    # configured it -- so calling it raw passed only when some earlier test in the same
    # process happened to run first, and failed with ``ctypes.ArgumentError`` whenever
    # this test ran alone, under ``-k``, or under ``--lf``.
    frame = create_frame(code=compile("pass", "<dummy>", "exec"), frame_globals={}, frame_locals={})

    ptr_size = ctypes.sizeof(ctypes.c_void_p)
    # An arbitrary slot that is 0 but is not f_back, forcing the scan to reject it.
    probe_offset = ptr_size * 10
    # The probe writes raw memory, so assert the object is actually big enough rather
    # than trusting a frame layout that differs by version and platform.
    assert probe_offset + ptr_size <= sys.getsizeof(frame)

    with (
        patch.object(_create_frame_module, "_get_py_frame_new", return_value=lambda *_: frame),
        patch.object(_link_frame_module, "range", return_value=[probe_offset]),
    ):
        ctypes.c_ssize_t.from_address(id(frame) + probe_offset).value = 0
        offset = _get_f_back_offset()
        assert offset is None
        # The scan must put the slot back so the frame's refcounts stay sane.
        assert ctypes.c_ssize_t.from_address(id(frame) + probe_offset).value == 0


def test_get_f_back_offset_ctypes_error() -> None:
    """Test when ctypes.c_ssize_t.from_address raises an error."""
    with patch("ctypes.c_ssize_t.from_address", side_effect=ValueError("ctypes error")):
        offset = _get_f_back_offset()
        assert offset is None
