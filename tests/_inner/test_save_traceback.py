"""Tests for the save_traceback module."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Never

import pytest

from offline_debug import ExceptionData, load_traceback, save_traceback

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def func_to_module_string(func: Callable) -> str:
    """Convert a function's body into a module string, removing indentation."""
    import inspect
    import textwrap

    source = inspect.getsource(func)
    # Get the body of the function (skip the def line)
    lines = source.splitlines()
    body = "\n".join(lines[1:])
    return textwrap.dedent(body)


def get_frames(tb: types.TracebackType | None) -> list[types.FrameType]:
    """Extract all frames from a traceback."""
    frames = []
    curr = tb
    while curr:
        frames.append(curr.tb_frame)
        curr = curr.tb_next
    return frames


def test_unpicklable_locals_verification(tmp_path: Path) -> None:
    """Test that unpicklable local variables are handled gracefully."""
    dump_file = tmp_path / "unpicklable.dump"

    class Unpicklable:
        def __reduce__(self) -> Never:
            msg = "Cannot pickle me"
            raise TypeError(msg)

        def __repr__(self) -> str:
            return "<Unpicklable Object>"

    def fail_with_unpicklable() -> Never:
        _obj = Unpicklable()
        # Use locals() to force inclusion in the locals dictionary.
        _ = locals()["_obj"]
        msg = "Error with unpicklable"
        raise ValueError(msg)

    try:
        fail_with_unpicklable()
    except Exception as e:  # noqa: BLE001
        save_traceback(e, dump_file)

    with pytest.raises(ValueError, match="Error with unpicklable") as exc_info:
        load_traceback(dump_file)

    frames = get_frames(exc_info.tb)

    frame_names = [f.f_code.co_name for f in frames]
    assert "fail_with_unpicklable" in frame_names
    f = next(f for f in frames if f.f_code.co_name == "fail_with_unpicklable")
    # Verify that our filtering logic caught the unpicklable item.
    assert any("<unpicklable" in str(v) for v in f.f_locals.values())


GLOBAL_TEST_VAL = "I am global"
GLOBAL_VAR = "initial"


def test_global_variables_in_stack(tmp_path: Path) -> None:
    """Test that global variables are preserved in the stack."""
    dump_file = tmp_path / "globals.dump"

    def fail_with_globals() -> None:
        global GLOBAL_TEST_VAL  # noqa: PLW0602
        if GLOBAL_TEST_VAL == "I am global":
            msg = "Global test"
            raise ValueError(msg)

    try:
        fail_with_globals()
    except Exception as e:  # noqa: BLE001
        save_traceback(e, dump_file)

    with pytest.raises(ValueError, match="Global test") as exc_info:
        load_traceback(dump_file)

    frames = get_frames(exc_info.tb)

    frame_names = [f.f_code.co_name for f in frames]
    assert "fail_with_globals" in frame_names
    f = next(f for f in frames if f.f_code.co_name == "fail_with_globals")
    assert f.f_globals["GLOBAL_TEST_VAL"] == "I am global"


def test_globals_changing_between_frames(tmp_path: Path) -> None:
    """Test that globals are captured per-frame for different modules."""
    dump_file = tmp_path / "globals_changing.dump"

    # Define the second module's content inside a function
    def module2_content() -> None:
        GLOBAL_VAR = "initial"  # noqa: N806, F841

        def level_2() -> Never:
            global GLOBAL_VAR
            GLOBAL_VAR = "changed"
            msg = "Error at level 2"
            raise ValueError(msg)

    mod2_code = func_to_module_string(module2_content)
    mod2 = types.ModuleType("mod2")
    exec(mod2_code, mod2.__dict__)  # noqa: S102

    def level_1() -> None:
        try:
            mod2.level_2()
        except Exception as e:  # noqa: BLE001
            save_traceback(e, dump_file)

    level_1()

    with pytest.raises(ValueError, match="Error at level 2") as exc_info:
        load_traceback(dump_file)

    frames = get_frames(exc_info.tb)

    l1_f = next(f for f in frames if f.f_code.co_name == "level_1")
    l2_f = next(f for f in frames if f.f_code.co_name == "level_2")

    # mod2.level_2 changed its OWN GLOBAL_VAR.
    # level_1's globals (this module) should NOT have GLOBAL_VAR (or it should be different)
    assert l2_f.f_globals["GLOBAL_VAR"] == "changed"
    assert "GLOBAL_VAR" not in l1_f.f_globals or l1_f.f_globals["GLOBAL_VAR"] == "initial"


def test_unpicklable_exception_coverage(tmp_path: Path) -> None:
    """Test that unpicklable exceptions are handled by falling back to RuntimeError."""
    dump_file = tmp_path / "unpicklable_exc.dump"

    class UnpicklableError(Exception):
        def __reduce__(self) -> Never:
            msg = "Cannot pickle me"
            raise TypeError(msg)

    def raise_unpicklable() -> Never:
        msg = "Unpicklable"
        raise UnpicklableError(msg)

    try:
        raise_unpicklable()
    except Exception as e:  # noqa: BLE001
        save_traceback(e, dump_file)

    with pytest.raises(RuntimeError, match="Unpicklable exception UnpicklableError: Unpicklable"):
        load_traceback(dump_file)


def test_save_traceback_file_none() -> None:
    """Test save_traceback when file is None."""
    exc = ValueError("test")
    data = save_traceback(exc, None)
    assert isinstance(data, ExceptionData)
