"""
Pickling that never aborts a dump because of one object.

Objects that vanilla pickle rejects are replaced by a placeholder string, at
any depth, so a single lock, socket or connection nested inside a value costs
only that leaf and not the whole value. The failure points of pickling are
leaves: an object's ``__reduce_ex__`` raising, or a class or function that
cannot be found by its qualified name. Both are checked on the way down, once
per object, so the dump stays a single pass with native speed for containers.

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
import copyreg
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


def _takes_over(cls: type[BaseException]) -> bool:
    """
    Whether we reconstruct ``cls`` ourselves instead of leaving it to pickle.

    Default reduction is guaranteed to round-trip for the other classes, and
    taking over would lose state the class restores itself (e.g. ``OSError.errno``).
    """
    return not _customizes_pickling(cls) and _has_python_constructor(cls)


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


def placeholder(obj: object) -> str:
    """Stand-in text for an object a dump cannot carry."""
    return f"<unpicklable {type(obj).__name__}: {obj!r}>"


def _placeholder_reduce(obj: object) -> tuple[type[str], tuple[str]]:
    """Reduce ``obj`` to its placeholder, rebuilt on load as a plain string."""
    return str, (placeholder(obj),)


class ResilientPickler(pickle.Pickler):
    """Pickler that replaces what it cannot pickle with placeholders, at any depth."""

    def __init__(self, file: IO[bytes]) -> None:
        """Pickle into ``file``, remembering per dump which classes pickle by name."""
        super().__init__(file)
        self._by_reference: dict[int, bool] = {}

    def _pickles_by_reference(self, obj: object) -> bool:
        """Whether a class or function is found under its qualified name (cached per dump)."""
        by_reference = self._by_reference.get(id(obj))
        if by_reference is None:
            try:
                pickle.dumps(obj)
                by_reference = True
            except Exception:  # noqa: BLE001 - not being picklable is the case we handle
                by_reference = False
            self._by_reference[id(obj)] = by_reference
        return by_reference

    # The inline suppression below is needed because this returns NotImplemented to
    # fall back to default reduction, which the stdlib stub's return type omits.
    def reducer_override(self, obj: object, /) -> object:  # ty: ignore[invalid-method-override]
        if isinstance(obj, (type, types.FunctionType)):
            # Pickled by name; leave the pickler's own handling in place when it works.
            return NotImplemented if self._pickles_by_reference(obj) else _placeholder_reduce(obj)
        try:
            if isinstance(obj, BaseException) and _takes_over(type(obj)):
                reduced = _exception_reduce(obj)
            else:
                # Mirror the pickler's own lookup order so the reduction is identical
                # to what it would compute, without computing it twice.
                reduce = copyreg.dispatch_table.get(type(obj))
                reduced = reduce(obj) if reduce else obj.__reduce_ex__(pickle.DEFAULT_PROTOCOL)
        except Exception:  # noqa: BLE001 - not being picklable is the case we handle
            return _placeholder_reduce(obj)
        if isinstance(reduced, tuple) and not self._pickles_by_reference(_reconstructor(reduced)):
            # Loading would import the reconstructor by name and fail there.
            return _placeholder_reduce(obj)
        return reduced


def _reconstructor(reduced: tuple[Any, ...]) -> object:
    """Return the callable a load must import to rebuild an object from its reduction."""
    callable_, args = reduced[0], reduced[1]
    # pickle itself recognizes copyreg's __newobj__/__newobj_ex__ by name.
    if getattr(callable_, "__name__", None) in ("__newobj__", "__newobj_ex__"):
        return args[0]
    return callable_


def _exception_reduce(exc: BaseException) -> tuple[Any, tuple[Any, ...]]:
    """Take over reconstruction of an exception whose constructor may reject ``args``."""
    state = exc.__dict__.copy() or None
    # Groups need their dedicated reconstructor: message/exceptions live
    # behind __new__ rather than a settable `args`.
    if isinstance(exc, BaseExceptionGroup):
        return reconstruct_exception_group, (type(exc), exc.message, exc.exceptions, state)
    return reconstruct_exception, (type(exc), exc.args, state)


def resilient_dump(obj: object, file: IO[bytes]) -> None:
    """Serialize ``obj`` to an open binary ``file`` using :class:`ResilientPickler`."""
    ResilientPickler(file).dump(obj)


def resilient_dumps(obj: object) -> bytes:
    """Serialize ``obj`` to bytes using :class:`ResilientPickler`."""
    buf = io.BytesIO()
    resilient_dump(obj, buf)
    return buf.getvalue()
