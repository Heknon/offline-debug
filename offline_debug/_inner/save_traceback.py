"""Save traceback to a file."""

import marshal
import pickle
import types
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

from offline_debug._inner._pickle_helpers import exception_safe_dump, exception_safe_dumps
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


def _serialize_exc_node(
    exc: BaseException, roundtrip_cache: dict[int, str | None]
) -> ExceptionData:
    """
    Serialize a single exception and its traceback frames.

    The node is returned unlinked: ``cause``/``context`` are ``None`` and a group's
    ``exceptions`` list is empty. :func:`_serialize_exc_data` fills those in once
    every node of the graph exists, which is what lets shared and cyclic
    references be represented without re-serializing anything.
    """
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

    try:
        exc_pickle = exception_safe_dumps(exc)
        # A dump that cannot be loaded later is worse than a placeholder, so also
        # verify the exception survives loading (e.g. a custom __reduce__ whose
        # reconstruction fails only at load time).
        pickle.loads(exc_pickle)  # noqa: S301
    except Exception:  # noqa: BLE001
        exc_pickle = exception_safe_dumps(
            RuntimeError(f"Unpicklable exception {type(exc).__name__}: {exc!s}")
        )

    if isinstance(exc, BaseExceptionGroup):
        return ExceptionGroupData(exc_pickle=exc_pickle, tb_frames=tb_frames, exceptions=[])
    return ExceptionData(exc_pickle=exc_pickle, tb_frames=tb_frames)


def _neighbours(exc: BaseException) -> Iterator[BaseException]:
    """Yield the exceptions ``exc`` links to: its cause, its context and group members."""
    if exc.__cause__ is not None:
        yield exc.__cause__
    if exc.__context__ is not None:
        yield exc.__context__
    if isinstance(exc, BaseExceptionGroup):
        yield from exc.exceptions


def _serialize_exc_data(
    exc: BaseException, roundtrip_cache: dict[int, str | None]
) -> ExceptionData:
    """
    Serialize the whole exception graph reachable from ``exc`` into dataclasses.

    ``__cause__``, ``__context__`` and ``ExceptionGroup.exceptions`` form a graph,
    not a tree: ``raise X from Y`` inside ``except Y`` points both the cause and
    the context of ``X`` at the same ``Y`` (so a naive recursive walk doubles its
    work at every link of a chain), and manual assignment can even create cycles
    (``raise e from e``). The graph is therefore walked iteratively with a memo
    keyed on ``id(exc)`` so every exception is serialized exactly once, in O(n)
    time and memory, and links are resolved afterwards against the memo. The
    resulting ``ExceptionData`` graph mirrors the original object identity, which
    ``pickle`` preserves on disk and :func:`load_traceback` restores.

    The memo holds the exception objects themselves so their ids cannot be
    recycled while the walk is in progress.
    """
    memo: dict[int, tuple[BaseException, ExceptionData]] = {}
    pending: list[BaseException] = [exc]
    while pending:
        curr = pending.pop()
        if id(curr) in memo:
            continue
        memo[id(curr)] = (curr, _serialize_exc_node(curr, roundtrip_cache))
        pending.extend(_neighbours(curr))

    for curr, data in memo.values():
        if curr.__cause__ is not None:
            data.cause = memo[id(curr.__cause__)][1]
        if curr.__context__ is not None:
            data.context = memo[id(curr.__context__)][1]
        if isinstance(curr, BaseExceptionGroup) and isinstance(data, ExceptionGroupData):
            data.exceptions = [memo[id(e)][1] for e in curr.exceptions]

    return memo[id(exc)][1]


def save_traceback(exc: BaseException, file: Path | BytesIO | None) -> ExceptionData:
    """Serialize an exception and its traceback to a file."""
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
