"""Save traceback to a file."""

import marshal
import pickle
import types
from io import BytesIO
from pathlib import Path

from offline_debug._inner._pickle_helpers import (
    exception_safe_dump,
    exception_safe_dumps,
    reconstruct_exception_group,
)
from offline_debug._inner.models import ExceptionData, ExceptionGroupData, FrameData

# Internal attributes that are either unpicklable or redundant in a new process.
# We exclude these specifically because they are automatically recreated
# when the new frame is initialized or when the module is imported.
_INTERNAL_ATTRIBUTES_TO_SKIP = ("__builtins__", "__doc__", "__loader__", "__package__", "__spec__")


def _get_stack_depth(frame: types.FrameType) -> int:
    """Calculate the depth of the current stack frame."""
    depth = 0
    curr: types.FrameType | None = frame
    while curr:
        depth += 1
        curr = curr.f_back
    return depth


def _filter_dict(d: dict, roundtrip_cache: dict[int, str | None]) -> dict:
    """
    Filter dictionary to include only items that survive a pickle round-trip.

    ``roundtrip_cache`` maps ``id(value)`` to ``None`` (survives) or a placeholder
    string, so a value shared across frames (e.g. module globals) is only checked
    once per save. The cached objects stay alive for the whole save because the
    frames still reference them, so the ids are stable.
    """
    result = {}
    for k, v in d.items():
        if k in _INTERNAL_ATTRIBUTES_TO_SKIP:
            continue
        cache_key = id(v)
        if cache_key not in roundtrip_cache:
            try:
                # We must verify that the value survives a full pickle round-trip
                # because many globals (like open file handles, database connections,
                # or modules) cannot be saved to disk, and some values pickle but fail
                # to unpickle (e.g. a custom __reduce__ whose callable raises on load).
                # Such values would otherwise break the entire load, so we replace
                # them with a placeholder. We use the same pickler that serializes
                # these dicts so the check reflects what will actually be written.
                pickle.loads(exception_safe_dumps(v))  # noqa: S301
                roundtrip_cache[cache_key] = None
            except BaseException:  # noqa: BLE001 - even a KeyboardInterrupt raised by a
                # value's reconstruction must not abort capturing the traceback.
                roundtrip_cache[cache_key] = f"<unpicklable {type(v).__name__}: {v!r}>"
        placeholder = roundtrip_cache[cache_key]
        result[k] = v if placeholder is None else placeholder
    return result


def _serialize_frames(
    exc: BaseException, roundtrip_cache: dict[int, str | None]
) -> list[FrameData]:
    """Serialize every frame of ``exc``'s own traceback."""
    tb_frames: list[FrameData] = []
    curr_tb = exc.__traceback__
    while curr_tb:
        f = curr_tb.tb_frame

        # Try to get the "real" module name. If the module was run as a script,
        # __name__ will be "__main__", but __spec__.name might contain the
        # actual module path if run via `python -m`.
        mod_name = f.f_globals.get("__name__")
        if mod_name == "__main__":
            spec = f.f_globals.get("__spec__")
            if spec and hasattr(spec, "name"):
                mod_name = spec.name

        tb_frames.append(
            FrameData(
                code=marshal.dumps(f.f_code),
                globals=_filter_dict(f.f_globals, roundtrip_cache),
                locals=_filter_dict(f.f_locals, roundtrip_cache),
                lasti=curr_tb.tb_lasti,
                lineno=curr_tb.tb_lineno,
                stack_depth=_get_stack_depth(f),
                module_name=mod_name,
            )
        )
        curr_tb = curr_tb.tb_next
    return tb_frames


def _member_stand_in(group: BaseExceptionGroup[BaseException]) -> BaseException:
    """
    Return the single stand-in member that the pickled skeleton of ``group`` holds.

    A ``BaseExceptionGroup`` cannot be empty, so the skeleton needs one member, and it
    must be as "base" as the group itself: ``BaseExceptionGroup.__new__`` turns a plain
    ``BaseExceptionGroup`` holding only ``Exception`` members into an ``ExceptionGroup``,
    and rejects a ``BaseException`` member inside an ``Exception`` subclass -- either
    would record the wrong class in the dump.
    """
    cls = Exception if isinstance(group, Exception) else BaseException
    return cls("member saved as a separate node")


def _group_skeleton(group: BaseExceptionGroup[BaseException]) -> BaseExceptionGroup[BaseException]:
    """
    Return a copy of ``group`` whose members are replaced by a single stand-in.

    Members are saved as nodes of their own and the loader rebuilds the group around
    those (see ``load_traceback._build_exception``), so pickling them along with the
    group would only duplicate every member -- once per level for nested groups -- and
    let a single member that fails to load take the whole group down with it, since the
    placeholder replaces the entire pickle.
    """
    return reconstruct_exception_group(
        type(group), group.message, (_member_stand_in(group),), group.__dict__.copy() or None
    )


def _pickle_exception(exc: BaseException) -> bytes:
    """Pickle ``exc`` itself, falling back to a placeholder if it cannot round-trip."""
    try:
        to_pickle = _group_skeleton(exc) if isinstance(exc, BaseExceptionGroup) else exc
        exc_pickle = exception_safe_dumps(to_pickle)
        # A dump that cannot be loaded later is worse than a placeholder, so also
        # verify the exception survives loading (e.g. a custom __reduce__ whose
        # reconstruction fails only at load time).
        pickle.loads(exc_pickle)  # noqa: S301
    except Exception:  # noqa: BLE001
        exc_pickle = exception_safe_dumps(
            RuntimeError(f"Unpicklable exception {type(exc).__name__}: {exc!s}")
        )
    return exc_pickle


def _build_exc_node(exc: BaseException, roundtrip_cache: dict[int, str | None]) -> ExceptionData:
    """Build the node for a single exception, without following its links."""
    exc_pickle = _pickle_exception(exc)
    tb_frames = _serialize_frames(exc, roundtrip_cache)

    if isinstance(exc, BaseExceptionGroup):
        return ExceptionGroupData(
            exc_pickle=exc_pickle,
            tb_frames=tb_frames,
            suppress_context=exc.__suppress_context__,
            exceptions=[],
        )
    return ExceptionData(
        exc_pickle=exc_pickle,
        tb_frames=tb_frames,
        suppress_context=exc.__suppress_context__,
    )


def _serialize_exc_data(
    exc: BaseException, roundtrip_cache: dict[int, str | None]
) -> ExceptionData:
    """
    Serialize an exception graph, preserving shared and cyclic references.

    ``__cause__``/``__context__`` are not guaranteed to form a tree: ``raise e from e``
    makes an exception its own cause, and the links can equally form longer cycles or
    simply revisit the same exception twice. Walking them recursively therefore either
    never terminates or re-serializes the same exception over and over, so we walk the
    graph iteratively and memoize each exception by identity.

    Memoizing does more than break cycles: because every exception is serialized exactly
    once, the resulting node graph mirrors the original one (a cycle stays a cycle, and an
    exception reachable as both cause and context stays a single shared node). ``pickle``
    reproduces that sharing faithfully on load.

    Keying the memo on ``id()`` is safe because every exception we visit is reachable from
    ``exc`` through strong ``__cause__``/``__context__``/``exceptions`` references, so none
    of them can be collected (and have its id reused) while the walk is in progress.
    """
    memo: dict[int, ExceptionData] = {}
    pending: list[BaseException] = []

    def node_for(e: BaseException) -> ExceptionData:
        node = memo.get(id(e))
        if node is None:
            node = memo[id(e)] = _build_exc_node(e, roundtrip_cache)
            pending.append(e)
        return node

    root = node_for(exc)
    while pending:
        curr = pending.pop()
        node = memo[id(curr)]
        if curr.__cause__ is not None:
            node.cause = node_for(curr.__cause__)
        if curr.__context__ is not None:
            node.context = node_for(curr.__context__)
        if isinstance(curr, BaseExceptionGroup) and isinstance(node, ExceptionGroupData):
            node.exceptions = [node_for(sub) for sub in curr.exceptions]

    return root


def save_traceback(exc: BaseException, file: Path | BytesIO | None) -> ExceptionData:
    """
    Serialize an exception and its traceback to a file.

    The exception graph is walked iteratively, so the interpreter's recursion limit does
    not bound it. The dump, however, nests every node inside the node that links to it,
    and pickling that nesting is recursive at the C level: a ``__cause__``/``__context__``
    chain of roughly 3,000 links is the practical ceiling, past which this raises
    ``RecursionError`` regardless of ``sys.setrecursionlimit``.
    """
    data = _serialize_exc_data(exc, roundtrip_cache={})
    if file is None:
        return data

    if isinstance(file, Path):
        with file.open("wb") as f:
            exception_safe_dump(data, f)
    elif isinstance(file, BytesIO):
        exception_safe_dump(data, file)
    else:
        msg = f"Unexpected type for file {type(file).__name__}"
        raise TypeError(msg)
    return data
