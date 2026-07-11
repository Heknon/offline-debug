"""
Robust pickling for exceptions that vanilla pickle cannot reconstruct.

``BaseException`` reduces to ``(type(self), self.args, state)``, so pickle
rebuilds an exception by calling ``Cls(*self.args)`` and then applying state.
Exceptions whose ``__init__`` takes required keyword-only arguments (and which
do not populate ``self.args`` via ``super().__init__``) pickle fine but fail on
load with a ``TypeError``. Since a traceback-capture library must serialize
whatever the user raised, we take over reconstruction and rebuild such
exceptions via ``__new__``, bypassing ``__init__`` entirely.

Exception groups are handled separately: ``BaseExceptionGroup`` stores its
``message``/``exceptions`` through ``__new__`` (not a settable ``args``), and
subclasses may override ``__new__``/``__init__`` with required keyword-only
arguments. We reconstruct them via ``BaseExceptionGroup.__new__``, bypassing any
subclass constructor, then restore instance state.
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


def reconstruct_exception_group(
    cls: type[BaseExceptionGroup[Any]],
    message: str,
    exceptions: tuple[BaseException, ...],
    state: dict[str, Any] | None,
) -> BaseExceptionGroup[Any]:
    """
    Rebuild an exception group without invoking any subclass constructor.

    ``BaseExceptionGroup.__new__`` establishes ``message``/``exceptions`` for any
    subclass, so we call it directly and bypass a subclass ``__new__``/``__init__``
    that may demand extra required (keyword-only) arguments.
    """
    exc = BaseExceptionGroup.__new__(cls, message, exceptions)
    if state:
        exc.__dict__.update(state)
    return exc


class RobustPickler(pickle.Pickler):
    """Pickler that reconstructs exceptions (and groups) bypassing ``__init__``."""

    # The inline suppression below is needed because this returns NotImplemented to
    # fall back to default reduction, which the stdlib stub's return type omits.
    def reducer_override(self, obj: object, /) -> object:  # ty: ignore[invalid-method-override]
        # Groups must be checked first: they are also BaseExceptions, but their
        # message/exceptions live behind __new__ rather than a settable `args`.
        if isinstance(obj, BaseExceptionGroup):
            state = obj.__dict__.copy() if obj.__dict__ else None
            return reconstruct_exception_group, (
                type(obj),
                obj.message,
                obj.exceptions,
                state,
            )
        if isinstance(obj, BaseException):
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
