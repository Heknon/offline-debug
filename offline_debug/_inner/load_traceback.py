"""Load traceback object from a dump file."""

import marshal
import pickle
import sys
import types
from io import BytesIO
from pathlib import Path
from types import CodeType

from offline_debug._inner._pickle_helpers import reconstruct_exception_group
from offline_debug._inner.c_api import (
    create_frame,
    link_frame,
)
from offline_debug._inner.models import (
    ExceptionData,
    ExceptionGroupData,
    FrameData,
)


def _reconstruct_traceback(data: ExceptionData) -> types.TracebackType | None:
    """
    Reconstruct one exception's traceback frames.

    Note on Python Locals:
    Python uses two ways to store local variables:
    1. "Slow" locals: A dictionary used for module-level code and class definitions.
    2. "Fast" locals: A fixed-size array used for functions. This is faster than
       dictionary lookups because variables are accessed by index.

    During reconstruction, we must explicitly synchronize these because PyFrame_New
    does not automatically populate the "fast" locals array from a dictionary.
    """
    reconstructed_frames: list[tuple[types.FrameType, FrameData]] = []
    for f_data in data.tb_frames:
        code: CodeType = marshal.loads(f_data.code)  # noqa: S302

        # In Python 3.11+, accessing f_locals on a frame created via
        # PyFrame_New for optimized code (functions) causes a segmentation fault
        # because the internal 'fast' locals array is not initialized.
        # As a workaround, we create a 'non-optimized' version of the code object
        # by compiling a dummy string. This ensures the bytecode is safe
        # (no LOAD_FAST) while preserving metadata like name and filename.
        # A simple module-level code object never has fast locals.
        # Since the source is empty, no optimized locals will be created.
        # Instead, python will go to the unoptimized dictionary we set under frame_locals later.
        unoptimized_code = compile("", code.co_filename, "exec")
        code = unoptimized_code.replace(
            co_name=code.co_name,
            co_firstlineno=code.co_firstlineno,
            co_qualname=code.co_qualname,
        )

        # PyFrame_New returns a new reference to a PyFrameObject.
        if f_data.module_name:
            f_data.globals["__name__"] = f_data.module_name

        frame: types.FrameType = create_frame(
            code=code, frame_globals=f_data.globals, frame_locals=f_data.locals
        )

        if reconstructed_frames:
            # link the frame back to the previously constructed frame.
            link_frame(frame, reconstructed_frames[-1][0])

        reconstructed_frames.append((frame, f_data))

    tb_next: types.TracebackType | None = None
    for frame, f_data in reversed(reconstructed_frames):
        tb = types.TracebackType(
            tb_next=tb_next,
            tb_frame=frame,
            tb_lasti=f_data.lasti,
            tb_lineno=f_data.lineno,
        )
        tb_next = tb

    return tb_next


def _exception_nodes(data: ExceptionData) -> list[ExceptionData]:
    """Return each node in an exception graph once."""
    nodes: list[ExceptionData] = []
    seen: set[int] = set()
    pending = [data]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        nodes.append(node)
        pending.extend(edge for edge in (node.cause, node.context) if edge is not None)
        if isinstance(node, ExceptionGroupData):
            pending.extend(node.exceptions)
    return nodes


def _reconstruct_exc_data(data: ExceptionData) -> BaseException:
    """Reconstruct an exception graph, preserving cycles and shared nodes."""
    nodes = _exception_nodes(data)
    built: dict[int, BaseException] = {}

    def build(node: ExceptionData) -> BaseException:
        existing = built.get(id(node))
        if existing is not None:
            return existing

        exc = pickle.loads(node.exc_pickle)  # noqa: S301
        if not isinstance(exc, BaseException):
            msg = f"Expected BaseException, but got {type(exc).__name__}"
            raise TypeError(msg)

        if isinstance(node, ExceptionGroupData) and isinstance(exc, BaseExceptionGroup):
            members = tuple(build(member) for member in node.exceptions)
            # derive() may discard an exception-group subclass and its custom state.
            exc = reconstruct_exception_group(
                type(exc), exc.message, members, exc.__dict__.copy() or None
            )

        exc = exc.with_traceback(_reconstruct_traceback(node))
        built[id(node)] = exc
        return exc

    for node in nodes:
        build(node)

    for node in nodes:
        exc = built[id(node)]
        if node.cause is not None:
            exc.__cause__ = built[id(node.cause)]
        if node.context is not None:
            exc.__context__ = built[id(node.context)]

    return built[id(data)]


def parse_traceback(file: Path | BytesIO) -> ExceptionData:
    if isinstance(file, Path):
        with file.open("rb") as f:
            data = pickle.load(f)  # noqa: S301
    else:
        data = pickle.load(file)  # noqa: S301

    if not isinstance(data, ExceptionData):
        msg = f"Expected _ExceptionData, but got {type(data).__name__}"
        raise TypeError(msg)
    return data


def load_traceback(file: Path | BytesIO, should_raise: bool = True) -> BaseException:  # noqa: FBT001, FBT002
    """Load an exception and its traceback from a file and raise it."""
    data = parse_traceback(file)

    exc = _reconstruct_exc_data(data)

    current_frames: list[types.FrameType] = []
    curr: types.FrameType | None = sys._getframe(1)  # noqa: SLF001
    while curr:
        current_frames.append(curr)
        curr = curr.f_back

    if exc.__traceback__ and current_frames:
        reconstructed_outer = exc.__traceback__.tb_frame
        link_frame(reconstructed_outer, current_frames[0])

    tb_chain: types.TracebackType | None = exc.__traceback__
    for frame in current_frames:
        tb_chain = types.TracebackType(
            tb_next=tb_chain,
            tb_frame=frame,
            tb_lasti=frame.f_lasti,
            tb_lineno=frame.f_lineno,
        )

    exc = exc.with_traceback(tb_chain)
    if should_raise:
        raise exc
    return exc
