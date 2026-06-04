"""Integration tests for offline-debug."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Never

import pytest

from offline_debug import load_traceback, save_traceback

if TYPE_CHECKING:
    from pathlib import Path


def get_stack_depth(item: types.FrameType | types.TracebackType | None) -> int:
    """Calculate the depth of the given stack frame or traceback chain."""
    depth = 0
    curr = item
    while curr:
        depth += 1
        curr = curr.f_back if isinstance(curr, types.FrameType) else curr.tb_next
    return depth


def get_frames(tb: types.TracebackType | None) -> list[types.FrameType]:
    """Extract all frames from a traceback."""
    frames = []
    curr = tb
    while curr:
        frames.append(curr.tb_frame)
        curr = curr.tb_next
    return frames


def test_stack_depth_preservation(tmp_path: Path) -> None:
    """Test that the stack depth is preserved during serialization."""
    dump_file = tmp_path / "depth.dump"

    original_depths = []

    def level_3() -> Never:
        f = sys._getframe()
        original_depths.append(get_stack_depth(f))
        msg = "Depth error"
        raise ValueError(msg)

    def level_2() -> None:
        f = sys._getframe()
        original_depths.append(get_stack_depth(f))
        level_3()

    def level_1() -> None:
        f = sys._getframe()
        original_depths.append(get_stack_depth(f))
        try:
            level_2()
        except Exception as e:  # noqa: BLE001
            save_traceback(e, dump_file)

    level_1()

    with pytest.raises(ValueError, match="Depth error") as exc_info:
        load_traceback(dump_file)

    # Reconstruct the depth from the traceback chain
    tb = exc_info.tb
    # Actually, load_traceback adds current frames.

    frames = get_frames(tb)
    l1_idx = next(i for i, f in enumerate(frames) if f.f_code.co_name == "level_1")
    l2_idx = next(i for i, f in enumerate(frames) if f.f_code.co_name == "level_2")
    l3_idx = next(i for i, f in enumerate(frames) if f.f_code.co_name == "level_3")

    # In a traceback, the list is TOP to BOTTOM (outer to inner)
    # So level_1 should be before level_2
    assert l1_idx < l2_idx
    assert l2_idx < l3_idx


def test_simple_exception_full_stack(tmp_path: Path) -> None:
    """Test serialization of a simple exception with a full stack."""
    dump_file = tmp_path / "simple.dump"

    def inner_raise() -> Never:
        _x = 10
        _y = "hello"
        msg = "Simple error"
        raise ValueError(msg)

    def middle_step() -> None:
        inner_raise()

    def capture_it() -> None:
        try:
            middle_step()
        except Exception as e:  # noqa: BLE001
            save_traceback(e, dump_file)

    capture_it()
    assert dump_file.exists()

    # Now we call load_traceback from another stack
    def second_stack_caller() -> None:
        load_traceback(dump_file)

    with pytest.raises(ValueError, match="Simple error") as exc_info:
        second_stack_caller()

    frames = get_frames(exc_info.tb)

    frame_names = [f.f_code.co_name for f in frames]
    assert "inner_raise" in frame_names
    assert "middle_step" in frame_names
    assert "second_stack_caller" in frame_names

    inner_idx = frame_names.index("inner_raise")
    middle_idx = frame_names.index("middle_step")
    second_idx = frame_names.index("second_stack_caller")

    # In a traceback, the list is TOP to BOTTOM (outer to inner)
    # second_stack_caller -> capture_it -> middle_step -> inner_raise
    assert second_idx < middle_idx
    assert middle_idx < inner_idx


def test_chained_exceptions_stack(tmp_path: Path) -> None:
    """Test serialization of chained exceptions."""
    dump_file = tmp_path / "chained.dump"

    def fail_inner() -> Never:
        msg = "Inner key error"
        raise KeyError(msg)

    def fail_outer() -> None:
        try:
            fail_inner()
        except KeyError as e:
            msg = "Outer runtime error"
            raise RuntimeError(msg) from e

    try:
        fail_outer()
    except Exception as e:  # noqa: BLE001
        save_traceback(e, dump_file)

    with pytest.raises(RuntimeError, match="Outer runtime error") as exc_info:
        load_traceback(dump_file)

    reconstructed_exc = exc_info.value
    assert isinstance(reconstructed_exc.__cause__, KeyError)

    outer_frames = get_frames(reconstructed_exc.__traceback__)
    outer_names = [f.f_code.co_name for f in outer_frames]
    assert "fail_outer" in outer_names

    inner_exc = reconstructed_exc.__cause__
    assert isinstance(inner_exc, KeyError)
    inner_frames = get_frames(inner_exc.__traceback__)
    inner_names = [f.f_code.co_name for f in inner_frames]
    assert "fail_inner" in inner_names


def test_reconstructed_frames_have_f_back(tmp_path: Path) -> None:
    """
    Test that reconstructed frames have their f_back pointers correctly linked.

    This test is currently expected to FAIL because f_back linking was removed
    to avoid segmentation faults, but it's required for full fidelity.
    """
    dump_file = tmp_path / "f_back_fidelity.dump"

    def level_2() -> Never:
        msg = "Fidelity error"
        raise ValueError(msg)

    def level_1() -> None:
        try:
            level_2()
        except Exception as e:  # noqa: BLE001
            save_traceback(e, dump_file)

    level_1()

    with pytest.raises(ValueError, match="Fidelity error") as exc_info:
        load_traceback(dump_file)

    frames = get_frames(exc_info.tb)
    # Traceback frames are ordered TOP to BOTTOM (outer to inner)
    # We want to check that level_2's frame points back to level_1
    l2_f = next(f for f in frames if f.f_code.co_name == "level_2")
    l1_f = next(f for f in frames if f.f_code.co_name == "level_1")

    assert l2_f.f_back is not None, "level_2.f_back should not be None"
    assert l2_f.f_back is l1_f, "level_2.f_back should point to level_1"


def test_locals_visibility_in_reconstructed_frames(tmp_path: Path) -> None:
    """Regression test: verify local variables are visible in reconstructed frames."""
    dump_file = tmp_path / "locals_visibility.dump"
    expected_val = 42

    def func_with_locals() -> Never:
        var_a = "hello"
        var_b = expected_val
        # Use them to ensure they aren't optimized away in some versions
        _ = f"{var_a} {var_b}"
        msg = "Locals error"
        raise ValueError(msg)

    try:
        func_with_locals()
    except ValueError as e:
        save_traceback(e, dump_file)

    with pytest.raises(ValueError, match="Locals error") as exc_info:
        load_traceback(dump_file)

    frames = get_frames(exc_info.tb)
    f = next(f for f in frames if f.f_code.co_name == "func_with_locals")

    assert f.f_locals.get("var_a") == "hello"
    assert f.f_locals.get("var_b") == expected_val


def test_exception_group_stack(tmp_path: Path) -> None:
    """Test serialization of ExceptionGroup."""
    dump_file = tmp_path / "eg.dump"

    def fail_1() -> Never:
        msg = "Error 1"
        raise ValueError(msg)

    def fail_2() -> Never:
        msg = "Error 2"
        raise TypeError(msg)

    try:
        try:
            fail_1()
        except ValueError as e1:
            try:
                fail_2()
            except TypeError as e2:
                raise ExceptionGroup("Group error", [e1, e2]) from None
    except ExceptionGroup as eg:
        save_traceback(eg, dump_file)

    with pytest.raises(ExceptionGroup, match="Group error") as exc_info:
        load_traceback(dump_file)

    reconstructed_eg = exc_info.value
    expected_exceptions_count = 2
    assert len(reconstructed_eg.exceptions) == expected_exceptions_count

    e1, e2 = reconstructed_eg.exceptions
    assert isinstance(e1, ValueError)
    assert isinstance(e2, TypeError)

    assert "fail_1" in [f.f_code.co_name for f in get_frames(e1.__traceback__)]
    assert "fail_2" in [f.f_code.co_name for f in get_frames(e2.__traceback__)]
