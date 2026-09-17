"""Functions for serializing and reconstructing exceptions with their tracebacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CodeData:
    """The metadata of a frame's code object that a reconstructed frame carries."""

    filename: str
    name: str
    qualname: str
    firstlineno: int


@dataclass
class FrameData:
    """Serialized data for a single stack frame."""

    code: CodeData
    globals: dict[str, Any]
    locals: dict[str, Any]
    lasti: int
    lineno: int
    stack_depth: int
    module_name: str | None = None


@dataclass(kw_only=True)
class ExceptionData:
    """Serialized data for an exception and its traceback."""

    exc_pickle: bytes
    tb_frames: list[FrameData]
    cause: ExceptionData | None = None
    context: ExceptionData | None = None


@dataclass(kw_only=True)
class ExceptionGroupData(ExceptionData):
    """Serialized data for an ExceptionGroup."""

    exceptions: list[ExceptionData]
