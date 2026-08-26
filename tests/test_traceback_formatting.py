"""
A loaded exception must behave like a genuine one under introspection.

Reconstructed frames deliberately carry a *synthetic* code object, because
accessing ``f_locals`` on a frame built by ``PyFrame_New`` over real optimized
bytecode segfaults. The original ``tb_lasti`` indexes into the original
bytecode, so pairing it with that synthetic code object made
``traceback._get_code_position`` search a two-entry ``co_positions()`` for
instruction ``lasti // 2``. That raises StopIteration inside a generator, which
surfaced as ``RuntimeError: generator raised StopIteration`` from *any* attempt
to format a loaded exception -- the library's central promise.

Reconstructed frames now report no instruction offset, so the stdlib falls back
to ``tb_lineno``, which is restored accurately.
"""

from __future__ import annotations

import traceback
from io import BytesIO

from offline_debug import load_traceback, save_traceback


def roundtrip(exc: BaseException) -> BaseException:
    """Save an exception to a buffer and load it back without raising."""
    buffer = BytesIO()
    save_traceback(exc, buffer)
    buffer.seek(0)
    return load_traceback(buffer, should_raise=False)


def formatted(exc: BaseException) -> str:
    """Format an exception exactly as an unhandled traceback would print."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def positions(exc: BaseException) -> list[tuple[str, int | None, str]]:
    """Return ``(filename, lineno, function)`` for every frame of a traceback."""
    return [(f.filename, f.lineno, f.name) for f in traceback.extract_tb(exc.__traceback__)]


def inner() -> None:
    """Raise the exception under test."""
    local_value = 42
    raise ValueError(str(local_value))


def outer() -> None:
    """Add a second frame to the traceback."""
    inner()


def make_error() -> BaseException:
    """Return a ValueError carrying a two-function traceback."""
    try:
        outer()
    except ValueError as err:
        return err
    msg = "outer() did not raise"
    raise AssertionError(msg)


def test_loaded_exception_can_be_formatted() -> None:
    """Formatting a loaded exception must not raise."""
    restored = roundtrip(make_error())

    text = formatted(restored)

    assert "ValueError: 42" in text
    assert "Traceback (most recent call last):" in text


def test_loaded_traceback_keeps_original_frames() -> None:
    """
    The original frames survive verbatim as the tail of the loaded traceback.

    ``load_traceback`` splices the reconstructed frames onto the live stack so the
    exception appears to have been raised at the load site, so the loaded traceback
    is longer than the original by exactly the frames of the caller.
    """
    original = make_error()
    expected = positions(original)
    restored = roundtrip(original)

    actual = positions(restored)

    assert actual[-len(expected) :] == expected
    assert [name for _, _, name in expected] == ["make_error", "outer", "inner"]


def test_loaded_frames_report_source_lines() -> None:
    """Line numbers must be accurate enough for the stdlib to find the source."""
    restored = roundtrip(make_error())

    text = formatted(restored)

    assert "in inner" in text
    assert "raise ValueError(str(local_value))" in text


def test_chained_exception_keeps_cause_wording() -> None:
    """A loaded chain prints the same explanatory line as the original."""
    try:
        try:
            outer()
        except ValueError as err:
            msg = "wrapper"
            raise RuntimeError(msg) from err
    except RuntimeError as err:
        original = err

    assert "direct cause" in formatted(original)
    assert "direct cause" in formatted(roundtrip(original))


def test_self_caused_exception_can_be_formatted() -> None:
    """``raise e from e`` must format without looping, exactly as it does natively."""
    try:
        try:
            msg = "original"
            raise ValueError(msg)
        except ValueError as err:
            raise err from err
    except ValueError as err:
        original = err

    restored = roundtrip(original)
    assert restored.__cause__ is restored

    text = formatted(restored)

    assert "ValueError: original" in text


def test_exception_group_can_be_formatted() -> None:
    """A loaded group prints its sub-exceptions in the standard group layout."""
    try:
        raise ExceptionGroup("group", [ValueError("x"), TypeError("y")])
    except ExceptionGroup as err:
        original = err

    text = formatted(roundtrip(original))

    assert "ExceptionGroup: group (2 sub-exceptions)" in text
    assert "ValueError: x" in text
    assert "TypeError: y" in text
