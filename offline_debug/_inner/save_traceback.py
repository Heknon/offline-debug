"""Save traceback to a file."""

import marshal
import pickle
import types
from io import BytesIO
from pathlib import Path

from offline_debug._inner.models import ExceptionData, FrameData

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
            # We must verify if the value is picklable because many globals
            # (like open file handles, database connections, or modules)
            # cannot be saved to disk.
            pickle.dumps(v)
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
        exc_pickle = pickle.dumps(exc)
    except Exception:  # noqa: BLE001
        exc_pickle = pickle.dumps(
            RuntimeError(f"Unpicklable exception {type(exc).__name__}: {exc!s}")
        )

    return ExceptionData(
        exc_pickle=exc_pickle,
        tb_frames=tb_frames,
        cause=_serialize_exc_data(exc.__cause__) if exc.__cause__ else None,
        context=_serialize_exc_data(exc.__context__) if exc.__context__ else None,
    )


def save_traceback(exc: BaseException, file: Path | BytesIO | None) -> ExceptionData:
    """Serialize an exception and its traceback to a file."""
    data = _serialize_exc_data(exc)
    if file is None:
        return data

    if isinstance(file, Path):
        with file.open("wb") as f:
            pickle.dump(data, f)
    elif isinstance(file, BytesIO):
        pickle.dump(data, file)
    else:
        msg = f"Unexpected type for file {type(file).__name__}"
        raise TypeError(msg)
    return data
