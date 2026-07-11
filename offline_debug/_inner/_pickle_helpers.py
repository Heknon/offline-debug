"""
Robust pickling for exceptions that vanilla pickle cannot reconstruct.

``BaseException`` reduces to ``(type(self), self.args, state)``, so pickle
rebuilds an exception by calling ``Cls(*self.args)`` and then applying state.
Exceptions whose ``__init__`` takes required keyword-only arguments (and which
do not populate ``self.args`` via ``super().__init__``) pickle fine but fail on
load with a ``TypeError``. Since a traceback-capture library must serialize
whatever the user raised, we take over reconstruction and rebuild such
exceptions via ``__new__``, bypassing ``__init__`` entirely.
"""

import io
import pickle
from typing import IO, Any


def reconstruct_exception(
    cls: type[BaseException], args: tuple[Any, ...], state: dict[str, Any] | None
) -> BaseException:
    """Rebuild an exception without invoking its ``__init__``."""
    exc = cls.__new__(cls)
    exc.args = args
    if state:
        exc.__dict__.update(state)
    return exc


class RobustPickler(pickle.Pickler):
    """Pickler that reconstructs plain exceptions via ``__new__``."""

    # The inline suppression below is needed because this returns NotImplemented to
    # fall back to default reduction, which the stdlib stub's return type omits.
    def reducer_override(self, obj: object, /) -> object:  # ty: ignore[invalid-method-override]
        # Exclude BaseExceptionGroup: its __new__ requires (message, exceptions),
        # so the __new__ bypass fails for it, and default pickling already
        # round-trips groups correctly (they are consumed via .derive() on load).
        if isinstance(obj, BaseException) and not isinstance(obj, BaseExceptionGroup):
            state = obj.__dict__.copy() if obj.__dict__ else None
            return reconstruct_exception, (type(obj), obj.args, state)
        return NotImplemented


def robust_dumps(obj: object) -> bytes:
    """Serialize ``obj`` to bytes using :class:`RobustPickler`."""
    buf = io.BytesIO()
    RobustPickler(buf).dump(obj)
    return buf.getvalue()


def robust_dump(obj: object, file: IO[bytes]) -> None:
    """Serialize ``obj`` to an open binary ``file`` using :class:`RobustPickler`."""
    RobustPickler(file).dump(obj)
