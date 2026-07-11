"""Save traceback to a file."""

import marshal
import pickle
import types
from io import BytesIO
from pathlib import Path

from offline_debug._inner._pickle_helpers import robust_dump, robust_dumps
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


def _filter_dict(d: dict) -> dict:
    """Filter dictionary to include only picklable items."""
    result = {}
    for k, v in d.items():
        if k in _INTERNAL_ATTRIBUTES_TO_SKIP:
            continue
        try:
            # We must verify that the value survives a full pickle round-trip
            # because many globals (like open file handles, database connections,
            # or modules) cannot be saved to disk, and some values pickle but fail
            # to unpickle (e.g. exceptions with keyword-only __init__ args). Such
            # values would otherwise break the entire load, so we replace them with
            # a placeholder. We use the same robust pickler that serializes these
            # dicts so the check reflects what will actually be written.
            pickle.loads(robust_dumps(v))  # noqa: S301
            result[k] = v
        except Exception:  # noqa: BLE001
            result[k] = f"<unpicklable {type(v).__name__}: {v!r}>"
    return result


def _serialize_exc_data(exc: BaseException) -> ExceptionData:
    """Recursively serialize exception data into dataclasses."""
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
                globals=_filter_dict(f.f_globals),
                locals=_filter_dict(f.f_locals),
                lasti=curr_tb.tb_lasti,
                lineno=curr_tb.tb_lineno,
                stack_depth=_get_stack_depth(f),
                module_name=mod_name,
            )
        )
        curr_tb = curr_tb.tb_next

    try:
        exc_pickle = robust_dumps(exc)
    except Exception:  # noqa: BLE001
        exc_pickle = robust_dumps(
            RuntimeError(f"Unpicklable exception {type(exc).__name__}: {exc!s}")
        )

    cause = _serialize_exc_data(exc.__cause__) if exc.__cause__ else None
    context = _serialize_exc_data(exc.__context__) if exc.__context__ else None

    if isinstance(exc, BaseExceptionGroup):
        return ExceptionGroupData(
            exc_pickle=exc_pickle,
            tb_frames=tb_frames,
            cause=cause,
            context=context,
            exceptions=[_serialize_exc_data(e) for e in exc.exceptions],
        )

    return ExceptionData(
        exc_pickle=exc_pickle,
        tb_frames=tb_frames,
        cause=cause,
        context=context,
    )


def save_traceback(exc: BaseException, file: Path | BytesIO | None) -> ExceptionData:
    """Serialize an exception and its traceback to a file."""
    data = _serialize_exc_data(exc)
    if file is None:
        return data

    if isinstance(file, Path):
        with file.open("wb") as f:
            robust_dump(data, f)
    elif isinstance(file, BytesIO):
        robust_dump(data, file)
    else:
        msg = f"Unexpected type for file {type(file).__name__}"
        raise TypeError(msg)
    return data
