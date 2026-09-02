"""Tests for the compiler-built code object reconstructed frames run on."""

from __future__ import annotations

import inspect
import traceback
import types

import pytest

from offline_debug._inner.c_api import create_frame
from offline_debug._inner.synthetic_code import CODE_UNIT_SIZE, Position, synthetic_code

FILENAME = "/somewhere/module.py"


@pytest.mark.parametrize(
    "position",
    [
        (1, 1, 0, 0),  # the first line, zero-width columns
        (0, 0, None, None),  # line 0, which a traceback entry without a line reports
        (12, 12, None, None),  # no column information (``-X no_debug_ranges``)
        (8, 8, 4, 20),  # an ordinary single-line expression
        (8, 11, 4, 1),  # a multi-line expression whose end column is below its start
        (70_000, 70_000, 100, 250),  # far beyond what one table entry can encode
    ],
)
def test_every_instruction_sits_on_the_failing_line(position: Position) -> None:
    """The compiler places the whole code object on the line and the entry at the columns."""
    code, lasti = synthetic_code(FILENAME, "inner", "outer.<locals>.inner", position)
    positions = list(code.co_positions())

    assert positions[lasti // CODE_UNIT_SIZE] == position
    assert {line for line, *_ in positions} == {position[0]}
    assert code.co_firstlineno == position[0]


def test_metadata_is_the_original_frames() -> None:
    """File and names come from the original code, even when not valid identifiers."""
    code, _ = synthetic_code(FILENAME, "<lambda>", "outer.<locals>.<lambda>", (3, 3, 0, 5))

    assert (code.co_filename, code.co_name, code.co_qualname) == (
        FILENAME,
        "<lambda>",
        "outer.<locals>.<lambda>",
    )


def test_code_reads_locals_from_a_dictionary() -> None:
    """The code must not be optimized: fast locals on a never-run frame are unsafe."""
    code, _ = synthetic_code(FILENAME, "f", "f", (3, 3, 0, 5))

    assert not code.co_flags & (inspect.CO_OPTIMIZED | inspect.CO_NEWLOCALS)
    assert code.co_nlocals == 0


def test_a_frame_over_the_code_reports_the_position() -> None:
    """A frame built on the code answers as a frame stopped at the failing instruction."""
    position = (8, 8, 4, 20)
    code, lasti = synthetic_code(FILENAME, "inner", "inner", position)
    frame = create_frame(code=code, frame_globals={"__name__": "m"}, frame_locals={"v": 1})
    tb = types.TracebackType(None, frame, lasti, position[0])

    (entry,) = traceback.extract_tb(tb)

    assert frame.f_lineno == position[0]
    assert frame.f_locals["v"] == 1
    assert (entry.filename, entry.lineno, entry.name) == (FILENAME, position[0], "inner")
    assert (entry.colno, entry.end_colno) == (position[2], position[3])


def test_a_position_the_compiler_rejects_is_an_error() -> None:
    """The caller must hand over a position CPython could have recorded."""
    with pytest.raises(ValueError, match="not a valid range"):
        synthetic_code(FILENAME, "f", "f", (5, 5, 9, 4))
