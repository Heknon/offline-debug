"""Tests for the resilient pickler."""

from __future__ import annotations

import pickle
import threading
from typing import Any

from offline_debug._inner._pickle_helpers import resilient_dumps


class Job:
    """Object holding an unpicklable lock next to plain state."""

    def __init__(self) -> None:
        """Mix picklable and unpicklable attributes."""
        self.name = "nightly"
        self.lock = threading.Lock()

    def describe(self) -> str:
        """Bound methods pickle through ``getattr`` on their instance."""
        return self.name


class Unreducible:
    """Object whose own pickle hook rejects it."""

    def __reduce__(self) -> tuple:
        """Refuse to be pickled."""
        msg = "cannot pickle me"
        raise TypeError(msg)


def roundtrip(obj: object) -> Any:  # noqa: ANN401
    """Dump ``obj`` with the resilient pickler and load it back with vanilla pickle."""
    return pickle.loads(resilient_dumps(obj))  # noqa: S301


def test_nested_unpicklable_values_become_placeholders() -> None:
    """Only the unpicklable leaf is replaced, whatever its depth."""
    loaded = roundtrip({"job": Job(), "items": [1, Unreducible()]})

    assert loaded["job"].name == "nightly"
    assert loaded["job"].lock.startswith("<unpicklable lock: ")
    assert loaded["items"][0] == 1
    assert loaded["items"][1].startswith("<unpicklable Unreducible: ")


def test_unresolvable_names_become_placeholders() -> None:
    """Classes, functions and instances whose class has no importable name are replaced."""

    class Local:
        pass

    loaded = roundtrip([Local(), Local, lambda: None, Job])

    assert loaded[0].startswith("<unpicklable Local: ")
    assert loaded[1].startswith("<unpicklable type: ")
    assert loaded[2].startswith("<unpicklable function: ")
    assert loaded[3] is Job


def test_callables_pickled_through_getattr_are_kept() -> None:
    """Builtins, descriptors and bound methods have no importable type but reduce fine."""
    loaded = roundtrip([len, str.upper, Job().describe])

    assert loaded[0] is len
    assert loaded[1] is str.upper
    assert loaded[2]() == "nightly"


def test_copyreg_reducers_still_apply() -> None:
    """Types pickled through ``copyreg.dispatch_table`` keep their reducer."""
    number = 1j
    assert roundtrip(number) == number
