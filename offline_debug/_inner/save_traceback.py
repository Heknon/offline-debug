"""Save traceback to a file."""

import contextlib
import marshal
import pickle
import types
from io import BytesIO
from pathlib import Path
from typing import Any

from offline_debug._inner._pickle_helpers import (
    exception_safe_dump,
    exception_safe_dumps,
    reconstruct_exception,
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


def _verified_dumps(obj: object) -> bytes:
    """
    Pickle ``obj`` and verify the result loads again.

    Some values pickle but fail to unpickle (e.g. a custom ``__reduce__`` whose
    callable raises on load), which would otherwise break the entire load.
    """
    data = exception_safe_dumps(obj)
    pickle.loads(data)  # noqa: S301
    return data


def _picklable(value: Any, roundtrip_cache: dict[int, str | None]) -> Any:  # noqa: ANN401
    """
    Return ``value`` if it survives a pickle round-trip, else a placeholder string.

    Many values (open file handles, database connections, modules) cannot be
    saved to disk. ``roundtrip_cache`` maps ``id(value)`` to ``None`` (survives)
    or its placeholder, so a value shared across frames (e.g. module globals) is
    only checked once per save. The cached objects stay alive for the whole save
    because the frames still reference them, so the ids are stable.
    """
    cache_key = id(value)
    if cache_key not in roundtrip_cache:
        try:
            _verified_dumps(value)
            roundtrip_cache[cache_key] = None
        except BaseException:  # noqa: BLE001 - even a KeyboardInterrupt raised by a
            # value's reconstruction must not abort capturing the traceback.
            roundtrip_cache[cache_key] = f"<unpicklable {type(value).__name__}: {value!r}>"
    placeholder = roundtrip_cache[cache_key]
    return value if placeholder is None else placeholder


def _filter_dict(d: dict, roundtrip_cache: dict[int, str | None]) -> dict:
    """Copy ``d`` with unpicklable values replaced by placeholders."""
    return {
        k: _picklable(v, roundtrip_cache)
        for k, v in d.items()
        if k not in _INTERNAL_ATTRIBUTES_TO_SKIP
    }


def _pickle_exception(exc: BaseException, roundtrip_cache: dict[int, str | None]) -> bytes:
    """
    Pickle ``exc`` so that it loads again.

    When the exception itself cannot round-trip, retry with a copy whose
    unpicklable args and attributes are replaced by placeholders, so its type
    and the rest of its state survive. Only if that fails too is the whole
    exception replaced by a ``RuntimeError`` describing it.
    """
    with contextlib.suppress(Exception):
        return _verified_dumps(exc)
    with contextlib.suppress(Exception):
        degraded = reconstruct_exception(
            type(exc),
            tuple(_picklable(arg, roundtrip_cache) for arg in exc.args),
            _filter_dict(vars(exc), roundtrip_cache) or None,
        )
        return _verified_dumps(degraded)
    return exception_safe_dumps(
        RuntimeError(f"Unpicklable exception {type(exc).__name__}: {exc!s}")
    )


def _serialize_exc_data(
    exc: BaseException, roundtrip_cache: dict[int, str | None]
) -> ExceptionData:
    """Serialize an exception graph, preserving cycles and shared nodes."""
    memo: dict[int, ExceptionData] = {}
    pending: list[BaseException] = []

    def node_for(current: BaseException) -> ExceptionData:
        node = memo.get(id(current))
        if node is not None:
            return node

        node = _serialize_exception(current, roundtrip_cache)
        memo[id(current)] = node
        pending.append(current)
        return node

    root = node_for(exc)
    while pending:
        current = pending.pop()
        node = memo[id(current)]
        if current.__cause__ is not None:
            node.cause = node_for(current.__cause__)
        if current.__context__ is not None:
            node.context = node_for(current.__context__)
        if isinstance(current, BaseExceptionGroup) and isinstance(node, ExceptionGroupData):
            node.exceptions = [node_for(member) for member in current.exceptions]

    return root


def _serialize_exception(
    exc: BaseException, roundtrip_cache: dict[int, str | None]
) -> ExceptionData:
    """Serialize one exception without following graph edges."""
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

    exc_pickle = _pickle_exception(exc, roundtrip_cache)

    if isinstance(exc, BaseExceptionGroup):
        return ExceptionGroupData(
            exc_pickle=exc_pickle,
            tb_frames=tb_frames,
            # This node is memoized before its edges are followed; the worklist in
            # _serialize_exc_data fills every group member before returning it.
            exceptions=[],
        )

    # cause and context start as None only while this node is being memoized. The
    # worklist in _serialize_exc_data fills both links before returning the graph.
    return ExceptionData(exc_pickle=exc_pickle, tb_frames=tb_frames)


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
