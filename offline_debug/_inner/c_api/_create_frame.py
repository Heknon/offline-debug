"""
Abstraction of the CPython api for creating a traceback frame object.

The reason we have to implement it, is that cpython doesn't expose
the api to create a FrameType using pure python.
"""

from __future__ import annotations

import ctypes
from functools import cache
from types import CodeType, FrameType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import _ctypes


@cache
def _get_py_frame_new() -> ctypes._NamedFuncPointer:
    """Get the PyFrame_New C function configured with ctypes."""
    func: ctypes._NamedFuncPointer = ctypes.pythonapi.PyFrame_New
    func.argtypes = (
        ctypes.c_void_p,  # PyThreadState *tstate
        ctypes.py_object,  # PyCodeObject *code
        ctypes.py_object,  # PyObject *globals
        ctypes.py_object,  # PyObject *locals
    )
    func.restype = ctypes.py_object

    def errcheck[T](
        result: T | None,
        _func: _ctypes.CFuncPtr,
        _args: tuple[Any, ...],
    ) -> T:  # pragma: no cover
        if not result:
            msg = "failed to create a new frame while calling"
            raise RuntimeError(msg)
        return result

    func.errcheck = errcheck
    return func


@cache
def _get_py_thread_state_get() -> ctypes._NamedFuncPointer:
    """Get the PyThreadState_Get C function configured with ctypes."""
    func = ctypes.pythonapi.PyThreadState_Get
    func.argtypes = ()
    func.restype = ctypes.c_void_p

    def errcheck[T](
        result: T | None, _func: _ctypes.CFuncPtr, _args: tuple[Any, ...]
    ) -> T:  # pragma: no cover
        if not result:
            msg = "failed to get the current thread state"
            raise RuntimeError(msg)
        return result

    func.errcheck = errcheck
    return func


def create_frame(
    code: CodeType,
    frame_globals: dict[str, Any],
    frame_locals: dict[str, Any],
    thread_state: int | None = None,
) -> FrameType:
    """
    Create a FrameType object using the received parameters.

    If the thread state received is None, use the current thread state automatically.
    """
    py_frame_new = _get_py_frame_new()
    py_thread_state_get = _get_py_thread_state_get()

    if thread_state is None:
        thread_state = py_thread_state_get()

    frame: FrameType = py_frame_new(thread_state, code, frame_globals, frame_locals)

    if not isinstance(frame, FrameType):
        msg = f"Expected types.FrameType, but got {type(frame).__name__}"
        raise TypeError(msg)

    # Whether PyFrame_New adopts ``frame_locals`` as the frame's own dictionary
    # (3.12) or ignores it and exposes a write-through proxy instead (3.13+, PEP 667),
    # writing the locals through ``f_locals`` afterwards lands them where the frame
    # reads them; when the dictionary was adopted this merely updates it with itself.
    if frame_locals:
        frame.f_locals.update(frame_locals)
    return frame
