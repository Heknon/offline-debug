"""Functions for serializing and reconstructing exceptions with their tracebacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class FrameData:
    """Serialized data for a single stack frame."""

    code: bytes
    globals: dict[str, Any]
    locals: dict[str, Any]
    lasti: int
    lineno: int
    stack_depth: int
    module_name: str | None = None


@dataclass(kw_only=True, eq=False)
class ExceptionData:
    """
    Serialized data for an exception and its traceback.

    Nodes compare and hash by identity, like the exceptions they describe: the saved
    graph may contain cycles (``raise e from e``), on which a structural ``__eq__`` would
    recurse forever. Two dumps of the same exception therefore never compare equal;
    ``id()`` (or the node itself, as a dict key or set member) is the stable handle.
    """

    exc_pickle: bytes
    tb_frames: list[FrameData]
    cause: ExceptionData | None = None
    context: ExceptionData | None = None
    # Whether the original exception had its context suppressed (``raise X from None``).
    # Defaulted so that a dump written before this field existed reads as ``False``:
    # dataclass defaults live on the class, so the missing key resolves there.
    suppress_context: bool = False


@dataclass(kw_only=True, eq=False)
class ExceptionGroupData(ExceptionData):
    """Serialized data for an ExceptionGroup."""

    exceptions: list[ExceptionData]


def walk_exception_data(data: ExceptionData) -> Iterator[ExceptionData]:
    """
    Yield every exception node reachable from ``data`` exactly once.

    A saved graph is not a tree. ``raise X from Y`` aliases ``cause`` and ``context``
    onto one node, and ``raise e from e`` makes a node its own cause, so following the
    links naively either visits a node repeatedly or never terminates.

    This is the traversal any consumer should use to render or re-serialize a dump:
    it is iterative (so walking a deep chain cannot exhaust the stack) and it stops
    descending as soon as it reaches a node it has already yielded. Use ``id()`` of the
    yielded nodes to reference them — that identity is what encodes cycles and sharing,
    and it is what a JSON projection needs in order to emit references instead of
    nesting.
    """
    seen: set[int] = set()
    stack = [data]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node

        if isinstance(node, ExceptionGroupData):
            stack.extend(node.exceptions)
        stack.extend(link for link in (node.cause, node.context) if link is not None)
