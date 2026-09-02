"""
Prove, once per process, that this interpreter can rebuild frames before a dump is loaded.

Frame reconstruction reaches into CPython internals: it creates frames through the C
API, writes a pointer into the frame object's memory to link ``f_back``, and relies on
the compiler and the frame's line-number logic agreeing on where a synthetic
instruction sits. None of that is guaranteed by any interpreter that has not been
tested, and the failure mode of a wrong guess is not always an exception: writing a
pointer at the wrong offset can corrupt memory silently.

The probe below rebuilds a tiny two-frame traceback with the same machinery a real
dump goes through, and checks every observable property a loaded exception relies on.
A probe that fails turns an unknown interpreter into a clear error at the first
``load_traceback`` call instead of a crash or a silently wrong traceback. A probe that
passes is evidence, not proof: it cannot detect memory corruption that happens not to
break the checked properties, which is why the supported Python range stays pinned.
"""

from __future__ import annotations

import platform
import sys
import traceback
import types
from functools import cache
from itertools import islice

from offline_debug._inner.c_api import create_frame, link_frame
from offline_debug._inner.synthetic_code import CODE_UNIT_SIZE, Position, synthetic_code

_PROBE_FILENAME = "<offline-debug self-check>"
_PROBE_POSITION: Position = (7, 7, 4, 9)
_PROBE_LOCALS = {"marker": "probe"}


class UnsupportedInterpreterError(RuntimeError):
    """Raised when this interpreter fails the frame reconstruction self-check."""


def _check(condition: bool, what: str) -> None:  # noqa: FBT001 - a bare assertion helper
    if not condition:
        raise RuntimeError(what)


def _probe() -> None:
    """Rebuild a two-frame traceback and verify everything a loaded exception relies on."""
    _check(
        sys.implementation.name == "cpython",
        "frame reconstruction uses the CPython C API through ctypes.pythonapi",
    )
    lineno, _end_lineno, col, end_col = _PROBE_POSITION

    code, lasti = synthetic_code(_PROBE_FILENAME, "inner", "outer.<locals>.inner", _PROBE_POSITION)
    position = next(islice(code.co_positions(), lasti // CODE_UNIT_SIZE, None), None)
    _check(position == _PROBE_POSITION, "the compiler did not place the instruction as asked")

    outer = create_frame(code=code, frame_globals={"__name__": __name__}, frame_locals={})
    inner = create_frame(
        code=code, frame_globals={"__name__": __name__}, frame_locals=dict(_PROBE_LOCALS)
    )
    _check(dict(inner.f_locals) == _PROBE_LOCALS, "frame locals did not survive frame creation")
    _check(inner.f_lineno == lineno, "the frame reports a line other than the one compiled in")

    link_frame(inner, outer)
    _check(inner.f_back is outer, "linking f_back did not take effect")
    _check(outer.f_back is None, "linking f_back touched the parent frame")

    tb = types.TracebackType(None, inner, lasti, lineno)
    (entry,) = traceback.extract_tb(tb)
    _check(
        (entry.filename, entry.lineno, entry.name) == (_PROBE_FILENAME, lineno, "inner"),
        "the traceback module does not see the file, line and name compiled in",
    )
    _check(
        (entry.colno, entry.end_colno) == (col, end_col),
        "the traceback module does not see the columns compiled in",
    )
    _check(bool("".join(traceback.format_tb(tb))), "the traceback module rendered nothing")


@cache
def ensure_supported() -> None:
    """
    Raise :class:`UnsupportedInterpreterError` unless frame reconstruction works here.

    The probe runs once per process; later calls return immediately.
    """
    try:
        _probe()
    except Exception as err:
        msg = (
            f"offline-debug cannot rebuild frames on {sys.implementation.name} "
            f"{platform.python_version()} ({platform.system()} {platform.machine()}): "
            f"{err}. Frame reconstruction relies on CPython internals, and this interpreter "
            "did not pass the self-check for them. Dumps can still be inspected with "
            "parse_traceback()."
        )
        raise UnsupportedInterpreterError(msg) from err
