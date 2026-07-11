"""
Pickling for exceptions that vanilla pickle cannot reconstruct.

``BaseException`` reduces to ``(type(self), self.args, state)``, so pickle
rebuilds an exception by calling ``Cls(*self.args)`` and then applying state.
Exceptions whose ``__init__``/``__new__`` takes required keyword-only arguments
(and which do not populate ``self.args`` via ``super().__init__``) pickle fine
but fail on load with a ``TypeError``. Since a traceback-capture library must
serialize whatever the user raised, we take over reconstruction for such
exceptions.

We keep the takeover as narrow as possible so well-behaved exceptions keep
their vanilla pickle behavior:

- Classes that customize the pickle protocol (``__reduce__``, ``__setstate__``,
  ...) are left alone; their hooks know how to rebuild them (e.g. ``OSError``).
- Classes whose constructors are all C-level (every builtin exception) are left
  alone; ``Cls(*args)`` is guaranteed to work and restores C-level state such
  as ``StopIteration.value`` that lives outside ``__dict__``.
- Only classes with a Python-defined ``__init__``/``__new__`` — the only kind
  that can demand required keyword-only arguments — are reconstructed by us,
  and even then reconstruction first retries normal construction and only
  falls back to ``__new__`` (bypassing the constructor) when it fails.

Exception groups get a dedicated reconstructor because a bare
``cls.__new__(cls)`` is not enough for them: ``BaseExceptionGroup.__new__``
requires ``message``/``exceptions`` to build a valid group for any subclass.

.. warning::
    The path of this module and the names of :func:`reconstruct_exception` and
    :func:`reconstruct_exception_group` are embedded in every dump that takes
    the takeover path. Renaming or moving them makes previously saved dumps
    unloadable, so treat them as part of the on-disk format.
"""

import contextlib
import io
import pickle
import types
from typing import IO, Any

# Pickle protocol hooks that let a class control its own serialization.
# If an exception class customizes any of these, we must not override its
# reduction: the class (e.g. OSError, ImportError) knows how to rebuild itself.
_PICKLE_PROTOCOL_HOOKS = (
    "__reduce_ex__",
    "__reduce__",
    "__getstate__",
    "__setstate__",
    "__getnewargs__",
    "__getnewargs_ex__",
)


def _customizes_pickling(cls: type[BaseException]) -> bool:
    """Whether ``cls`` overrides any pickle protocol hook of ``BaseException``."""
    return any(
        getattr(cls, hook, None) is not getattr(BaseException, hook, None)
        for hook in _PICKLE_PROTOCOL_HOOKS
    )


def _has_python_constructor(cls: type[BaseException]) -> bool:
    """
    Whether ``cls`` has a Python-defined ``__init__`` or ``__new__``.

    Builtin exceptions expose C slot wrappers here, and for them default pickle
    reconstruction always works. Only a Python-level constructor can introduce
    required keyword-only arguments that break ``Cls(*args)`` at load time.
    """
    return isinstance(cls.__init__, types.FunctionType) or isinstance(
        cls.__new__, types.FunctionType
    )


def reconstruct_exception(
    cls: type[BaseException], args: tuple[Any, ...], state: dict[str, Any] | None
) -> BaseException:
    """Rebuild an exception, bypassing its constructor only if it rejects ``args``."""
    try:
        exc = cls(*args)
    except Exception:  # noqa: BLE001 - a rejecting constructor is exactly the case we handle
        exc = BaseException.__new__(cls)
        # Suppressed failures happen when e.g. `args` is shadowed by a read-only property.
        with contextlib.suppress(Exception):
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
    Rebuild an exception group, bypassing its constructor only when necessary.

    ``BaseExceptionGroup.__new__`` establishes ``message``/``exceptions`` for any
    subclass, so when normal construction fails (e.g. a subclass demanding extra
    required keyword-only arguments) we call it directly.
    """
    try:
        exc = cls(message, exceptions)
    except Exception:  # noqa: BLE001 - a rejecting constructor is exactly the case we handle
        exc = BaseExceptionGroup.__new__(cls, message, exceptions)
    if state:
        exc.__dict__.update(state)
    return exc


class CustomExceptionPickler(pickle.Pickler):
    """Pickler that takes over reconstruction of constructor-rejecting exceptions."""

    # The inline suppression below is needed because this returns NotImplemented to
    # fall back to default reduction, which the stdlib stub's return type omits.
    def reducer_override(self, obj: object, /) -> object:  # ty: ignore[invalid-method-override]
        if not isinstance(obj, BaseException):
            return NotImplemented
        cls = type(obj)
        if _customizes_pickling(cls) or not _has_python_constructor(cls):
            # Default reduction is guaranteed to round-trip; taking over would
            # lose state the class restores itself (e.g. OSError.errno).
            return NotImplemented
        state = obj.__dict__.copy() or None
        # Groups need their dedicated reconstructor: message/exceptions live
        # behind __new__ rather than a settable `args`.
        if isinstance(obj, BaseExceptionGroup):
            return reconstruct_exception_group, (cls, obj.message, obj.exceptions, state)
        return reconstruct_exception, (cls, obj.args, state)


def exception_safe_dump(obj: object, file: IO[bytes]) -> None:
    """Serialize ``obj`` to an open binary ``file`` using :class:`CustomExceptionPickler`."""
    CustomExceptionPickler(file).dump(obj)


def exception_safe_dumps(obj: object) -> bytes:
    """Serialize ``obj`` to bytes using :class:`CustomExceptionPickler`."""
    buf = io.BytesIO()
    exception_safe_dump(obj, buf)
    return buf.getvalue()
