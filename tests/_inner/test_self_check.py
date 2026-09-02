"""Tests for the interpreter self-check that guards frame reconstruction."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

import offline_debug
from offline_debug import UnsupportedInterpreterError, ensure_supported
from offline_debug._inner import self_check
from tests.helpers import roundtrip

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def rerun_self_check() -> Iterator[None]:
    """Make every test run the probe afresh instead of reusing a cached verdict."""
    ensure_supported.cache_clear()
    yield
    ensure_supported.cache_clear()


def test_this_interpreter_passes() -> None:
    """The suite runs on a supported interpreter, so the probe must pass."""
    ensure_supported()


def test_verdict_is_cached() -> None:
    """The probe runs once; later calls do not rebuild frames again."""
    ensure_supported()

    assert ensure_supported.cache_info().hits == 0
    ensure_supported()
    assert ensure_supported.cache_info().hits == 1


def test_non_cpython_is_refused(monkeypatch) -> None:
    """``ctypes.pythonapi`` is CPython-only, so any other implementation is refused."""
    monkeypatch.setattr(sys.implementation, "name", "pypy")

    with pytest.raises(UnsupportedInterpreterError, match="cannot rebuild frames on pypy"):
        ensure_supported()


def test_a_probe_that_does_not_link_frames_is_refused(monkeypatch) -> None:
    """A frame linker whose write does not take effect fails the probe."""
    monkeypatch.setattr(self_check, "link_frame", lambda _frame, _f_back: None)

    with pytest.raises(UnsupportedInterpreterError, match="f_back did not take effect") as info:
        ensure_supported()

    assert isinstance(info.value.__cause__, RuntimeError)


def test_a_probe_that_crashes_is_refused(monkeypatch) -> None:
    """An exception anywhere in the probe becomes the same clear error."""

    def broken(*_args: object, **_kwargs: object) -> None:
        msg = "PyFrame_New is missing"
        raise AttributeError(msg)

    monkeypatch.setattr(self_check, "create_frame", broken)

    with pytest.raises(UnsupportedInterpreterError, match="PyFrame_New is missing"):
        ensure_supported()


def test_loading_a_dump_runs_the_probe_first(monkeypatch) -> None:
    """``load_traceback`` refuses to touch frames on an interpreter that fails the probe."""
    monkeypatch.setattr(sys.implementation, "name", "pypy")

    with pytest.raises(UnsupportedInterpreterError):
        roundtrip(ValueError("never rebuilt"))


def test_error_is_public() -> None:
    """Callers can catch the refusal by name from the package root."""
    assert offline_debug.UnsupportedInterpreterError is self_check.UnsupportedInterpreterError
    assert issubclass(UnsupportedInterpreterError, RuntimeError)
