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

# Sentinel ``tb_lasti`` meaning "no bytecode offset applies to this frame".
# ``traceback._get_code_position`` returns no position for a negative offset, which
# makes the stdlib fall back to ``tb_lineno`` instead of indexing ``co_positions()``.
_NO_INSTRUCTION_OFFSET = -1


def _reconstruct_frames(data: ExceptionData) -> types.TracebackType | None:
    """
    Rebuild the traceback of a single exception from its serialized frames.

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
        tb_next = types.TracebackType(
            tb_next=tb_next,
            tb_frame=frame,
            # The original ``lasti`` indexes into the original bytecode, but the frame
            # above deliberately carries a synthetic (empty) code object, so the two do
            # not correspond. Handing the real ``lasti`` to the stdlib makes
            # ``traceback._get_code_position`` walk ``co_positions()`` of a two-entry
            # code object looking for instruction ``lasti // 2``, which raises
            # StopIteration inside a generator -- surfacing as ``RuntimeError: generator
            # raised StopIteration`` from any attempt to format the exception.
            #
            # A negative ``lasti`` is the stdlib's own signal for "no instruction
            # position available": it then falls back to ``tb_lineno``, which we restore
            # accurately. ``FrameData.lasti`` stays in the dump as original metadata.
            tb_lasti=_NO_INSTRUCTION_OFFSET,
            tb_lineno=f_data.lineno,
        )
    return tb_next


def _unpickle_exception(data: ExceptionData) -> BaseException:
    """Unpickle the exception object stored on ``data``."""
    exc = pickle.loads(data.exc_pickle)  # noqa: S301
    if not isinstance(exc, BaseException):
        msg = f"Expected BaseException, but got {type(exc).__name__}"
        raise TypeError(msg)
    return exc


def _members(data: ExceptionData, exc: BaseException) -> list[ExceptionData]:
    """
    Return the sub-exceptions ``data`` has to be rebuilt around, if any.

    A group whose own pickle fell back to the ``RuntimeError`` placeholder loads as a
    plain exception that references none of its members, so rebuilding them (frames and
    all) would be wasted work. They are still built if something else links to them.
    """
    if isinstance(data, ExceptionGroupData) and isinstance(exc, BaseExceptionGroup):
        return data.exceptions
    return []


def _build_order(root: ExceptionData) -> tuple[list[ExceptionData], dict[int, BaseException]]:
    """
    List every node that has to be rebuilt, each group after its members.

    Only ``exceptions`` edges constrain build order: a group is rebuilt around
    already-built members, whereas ``cause``/``context`` are assigned afterwards (see
    :func:`_reconstruct_exc_data`). So the walk is a post-order depth-first search over
    ``exceptions`` edges alone, and every ``cause``/``context`` target is queued as a new
    *starting point* that is only taken up once the current search has finished.
    Following a link mid-search would let a group that is first reached through one of
    its own members -- ``except ExceptionGroup as g: raise g.exceptions[0]`` saves
    exactly that shape -- be emitted before that member.

    Each node is unpickled exactly once, on discovery, so that the members of a group
    that loaded as a placeholder can be pruned (see :func:`_members`); the unpickled
    objects are returned alongside the order for the build pass. The walk is iterative
    and identity-memoized because the graph may contain cycles.

    This deliberately does not reuse :func:`walk_exception_data`: that traversal reports
    every node in no particular order, whereas this one prunes and orders.
    """
    seen: set[int] = set()
    order: list[ExceptionData] = []
    unpickled: dict[int, BaseException] = {}
    starts: list[ExceptionData] = [root]

    while starts:
        stack: list[tuple[ExceptionData, bool]] = [(starts.pop(), False)]
        while stack:
            node, members_done = stack.pop()
            if members_done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            exc = unpickled[id(node)] = _unpickle_exception(node)
            starts.extend(link for link in (node.cause, node.context) if link is not None)
            # Re-push the node below its members so it is emitted after them.
            stack.append((node, True))
            stack.extend((sub, False) for sub in _members(node, exc))

    return order, unpickled


def _build_exception(
    data: ExceptionData, exc: BaseException, built: dict[int, BaseException]
) -> BaseException:
    """Give the unpickled ``exc`` its traceback, rebuilding a group around ``built`` members."""
    if isinstance(data, ExceptionGroupData) and isinstance(exc, BaseExceptionGroup):
        try:
            inner_excs = [built[id(sub)] for sub in data.exceptions]
        except KeyError:
            # Only reachable from a hand-crafted dump: a real group's members are fixed
            # at construction, so ``save_traceback`` can never record a group that
            # (transitively) contains itself -- and such a group has no valid build
            # order, since its members must exist first.
            msg = "Cannot reconstruct an exception group that transitively contains itself"
            raise ValueError(msg) from None
        # The exceptions inside the unpickled exc object have incomplete data, so
        # rebuild the group around the fully reconstructed ones. We must not use
        # derive() for this: its default implementation returns a plain
        # ExceptionGroup, dropping the subclass type and its custom state.
        exc = reconstruct_exception_group(
            type(exc), exc.message, tuple(inner_excs), exc.__dict__.copy() or None
        )

    return exc.with_traceback(_reconstruct_frames(data))


def _reconstruct_exc_data(data: ExceptionData) -> BaseException:
    """
    Reconstruct an exception graph, restoring shared and cyclic references.

    ``save_traceback`` records the exception graph as-is, so ``cause``/``context`` may
    revisit a node or point back at it (``raise e from e``). Reconstruction therefore
    runs in two passes: first every exception is built, then the ``__cause__``/
    ``__context__`` links and ``__suppress_context__`` are restored from the identity
    map. Doing the links second is what lets a cycle be restored as a genuine cycle
    instead of an endless chain of copies, and it keeps a node that is reachable twice a
    single shared object.
    """
    nodes, unpickled = _build_order(data)

    built: dict[int, BaseException] = {}
    for node in nodes:
        built[id(node)] = _build_exception(node, unpickled[id(node)], built)

    for node in nodes:
        exc = built[id(node)]
        if node.cause is not None:
            exc.__cause__ = built[id(node.cause)]
        if node.context is not None:
            exc.__context__ = built[id(node.context)]
        # Assigning ``__cause__`` sets ``__suppress_context__`` as a side effect, so the
        # saved value must be restored *after* the links -- and unconditionally, since
        # ``raise X from None`` outside an ``except`` block suppresses a context that is
        # itself ``None``, leaving no assignment to piggyback on.
        exc.__suppress_context__ = node.suppress_context

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
