"""
Object proxies are saved as placeholders without ever being touched.

A proxy such as an ``rpyc`` netref forwards every instance operation - attribute
reads, ``repr``, ``str``, ``isinstance`` through ``__class__``, pickling through
``__reduce_ex__`` - to a remote peer, and on a broken connection each one blocks
for the peer's whole timeout. The fake proxy below turns every such operation
into an exception instead, so a test can prove the save never made one.
"""

from __future__ import annotations

import pickle
from io import BytesIO
from typing import TYPE_CHECKING, Any, Never, SupportsIndex

import pytest

from offline_debug import (
    DEFAULT_PROXY_TYPES,
    load_traceback,
    parse_traceback,
    save_traceback,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from offline_debug import ExceptionData

TOUCHES: list[str] = []


class RemoteTouchedError(AssertionError):
    """Raised by the fake proxy whenever the save reaches for the peer."""


class FakeNetref:
    """Every instance operation is a remote call, like an ``rpyc`` netref."""

    __slots__ = ()

    def __getattribute__(self, name: str) -> Never:
        """Read an attribute: a remote call."""
        TOUCHES.append(name)
        raise RemoteTouchedError(name)

    def __repr__(self) -> Never:
        """Represent the object: a remote call."""
        TOUCHES.append("__repr__")
        raise RemoteTouchedError("__repr__")

    def __str__(self) -> Never:
        """Stringify the object: a remote call."""
        TOUCHES.append("__str__")
        raise RemoteTouchedError("__str__")

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> Never:
        """Pickle the object: a remote call, which is how rpyc's ``obtain`` works."""
        TOUCHES.append("__reduce_ex__")
        raise RemoteTouchedError("__reduce_ex__")

    def __hash__(self) -> Never:
        """Hash the object: a remote call."""
        TOUCHES.append("__hash__")
        raise RemoteTouchedError("__hash__")


class RemoteSensor(FakeNetref):
    """A subclass, as every concrete netref class is a subclass of the base."""

    __slots__ = ()


FAKE_NETREF_NAME = f"{FakeNetref.__module__}.{FakeNetref.__qualname__}"


class ProxyHolderError(ValueError):
    """An exception carrying a proxy in its ``args`` and as an attribute."""

    def __init__(self, proxy: object) -> None:
        """Keep ``proxy`` both as a positional argument and as an attribute."""
        super().__init__("remote step failed", proxy)
        self.remote = proxy


@pytest.fixture(autouse=True)
def _reset_touches() -> None:
    TOUCHES.clear()


def raise_with_proxies_in_scope(proxy: object) -> Never:
    """Fail with proxies in the frame's locals, nested in containers, and on the exception."""
    _client = proxy
    _pool = [proxy, proxy]
    _by_name = {"client": proxy}
    _ = locals()
    raise ProxyHolderError(proxy)


def capture(
    proxy: object, proxy_types: Iterable[type | str] = DEFAULT_PROXY_TYPES
) -> ExceptionData:
    """Save the failure through a dump and read it back, returning the parsed data."""
    try:
        raise_with_proxies_in_scope(proxy)
    except ProxyHolderError as err:
        buffer = BytesIO()
        save_traceback(err, buffer, proxy_types=proxy_types)
    buffer.seek(0)
    return parse_traceback(buffer)


def frame_locals(data: ExceptionData) -> dict[str, Any]:
    """Return the locals of the frame that raised."""
    return data.tb_frames[-1].locals


def assert_placeholder(value: object, cls: type) -> None:
    """``value`` is the placeholder the save writes for an instance of ``cls``."""
    assert isinstance(value, str), value
    assert value.startswith(f"<proxy {cls.__module__}.{cls.__qualname__} at 0x"), value


def test_registered_class_is_never_touched() -> None:
    """A proxy registered by class is saved as a placeholder with no instance access."""
    data = capture(FakeNetref(), proxy_types=(FakeNetref,))

    assert TOUCHES == []
    saved = frame_locals(data)
    assert_placeholder(saved["_client"], FakeNetref)
    # Nested proxies are replaced by the pickler, wherever they sit.
    assert_placeholder(saved["_pool"][0], FakeNetref)
    assert_placeholder(saved["_pool"][1], FakeNetref)
    assert_placeholder(saved["_by_name"]["client"], FakeNetref)

    exc = pickle.loads(data.exc_pickle)  # noqa: S301 - our own dump
    assert isinstance(exc, ProxyHolderError)
    assert_placeholder(exc.args[1], FakeNetref)
    assert_placeholder(exc.remote, FakeNetref)


def test_registered_name_matches_without_importing_the_package() -> None:
    """A fully-qualified class name matches instances of the class and its subclasses."""
    data = capture(RemoteSensor(), proxy_types=(FAKE_NETREF_NAME,))

    assert TOUCHES == []
    assert_placeholder(frame_locals(data)["_client"], RemoteSensor)


def test_registered_class_matches_subclasses() -> None:
    """Registering the base class covers every concrete proxy class derived from it."""
    data = capture(RemoteSensor(), proxy_types=(FakeNetref,))

    assert TOUCHES == []
    assert_placeholder(frame_locals(data)["_client"], RemoteSensor)


def test_rpyc_netrefs_are_recognised_by_default() -> None:
    """``rpyc.core.netref.BaseNetref`` is covered out of the box, by name."""
    assert "rpyc.core.netref.BaseNetref" in DEFAULT_PROXY_TYPES

    # Stand in for rpyc's base class without rpyc being installed.
    class BaseNetref(FakeNetref):
        __slots__ = ()

    BaseNetref.__module__ = "rpyc.core.netref"
    BaseNetref.__qualname__ = "BaseNetref"

    class RemoteList(BaseNetref):
        __slots__ = ()

    data = capture(RemoteList())

    assert TOUCHES == []
    assert_placeholder(frame_locals(data)["_client"], RemoteList)


def test_loaded_exception_carries_the_placeholders() -> None:
    """The reconstructed exception holds placeholders where the proxies were."""
    try:
        raise_with_proxies_in_scope(FakeNetref())
    except ProxyHolderError as err:
        buffer = BytesIO()
        save_traceback(err, buffer, proxy_types=(FakeNetref,))
    buffer.seek(0)

    loaded = load_traceback(buffer, should_raise=False)
    assert isinstance(loaded, ProxyHolderError)
    assert_placeholder(loaded.remote, FakeNetref)
    assert TOUCHES == []


def test_in_memory_result_holds_no_live_proxy_at_top_level() -> None:
    """With no file, the returned data already carries the placeholder for a frame value."""
    try:
        raise_with_proxies_in_scope(FakeNetref())
    except ProxyHolderError as err:
        data = save_traceback(err, None, proxy_types=(FakeNetref,))

    assert TOUCHES == []
    assert_placeholder(frame_locals(data)["_client"], FakeNetref)


def raise_value_error() -> Never:
    """Fail with nothing unusual in scope."""
    msg = "boom"
    raise ValueError(msg)


def test_invalid_entry_is_rejected() -> None:
    """An entry that is neither a class nor a name is a caller error, reported at once."""
    not_a_type_or_name: Any = (42,)
    try:
        raise_value_error()
    except ValueError as err:
        with pytest.raises(TypeError, match="classes or fully-qualified class names"):
            save_traceback(err, None, proxy_types=not_a_type_or_name)


def test_unregistered_hostile_value_does_not_abort_the_save() -> None:
    """
    A value whose pickling *and* repr fail still yields a dump.

    Before, the placeholder was built with ``repr`` inside the handler that had
    just caught the pickling failure, so a second failure escaped and the whole
    traceback was lost. Now the placeholder falls back to the bare object repr.
    """
    data = capture(FakeNetref(), proxy_types=())

    saved = frame_locals(data)["_client"]
    assert isinstance(saved, str)
    assert saved.startswith("<unpicklable FakeNetref: <")
    assert "FakeNetref object at 0x" in saved
    # The save reached for the value - which is what registering it prevents. The
    # first touch is isinstance() consulting __class__, before pickling even starts.
    assert "__class__" in TOUCHES
    assert "__repr__" in TOUCHES


def test_exception_whose_str_fails_is_still_saved() -> None:
    """The unpicklable-exception fallback must not depend on ``str(exc)`` working."""

    class HostileError(RuntimeError):
        def __init__(self) -> None:
            """Carry a local lambda, which cannot be pickled."""
            super().__init__()
            self.callback = lambda: None

        def __str__(self) -> Never:
            """Refuse to describe the failure."""
            msg = "no message for you"
            raise RemoteTouchedError(msg)

    def raise_hostile() -> Never:
        raise HostileError

    try:
        raise_hostile()
    except HostileError as err:
        buffer = BytesIO()
        save_traceback(err, buffer)
    buffer.seek(0)

    loaded = load_traceback(buffer, should_raise=False)
    assert isinstance(loaded, RuntimeError)
    message = str(loaded)
    assert message.startswith("Unpicklable exception HostileError: <")
    assert "HostileError object at 0x" in message
