"""Intended for manual use, run the test in debug mode and view the exceptions."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from rich import traceback
from rich.console import Console

from offline_debug import load_traceback, parse_traceback, save_traceback

global_variable = 1

console = Console(force_terminal=True)
traceback.install(show_locals=True, console=console)


def failure() -> None:
    """Raise a nested exception for testing."""
    _python_version = sys.version

    def exception_raising_func() -> None:
        local = "local"
        msg = f"exception {local}"
        raise ValueError(msg)

    with tempfile.TemporaryDirectory() as tmpdir:
        dump_file = Path(tmpdir) / "exception.dump"
        try:
            exception_raising_func()
        except ValueError as e:
            exception_group = ExceptionGroup(
                "exception group", [e, RuntimeError("another exception")]
            )
            save_traceback(exception_group, dump_file)

        global global_variable
        global_variable += 1
        _exception_data = parse_traceback(dump_file)
        load_traceback(dump_file)


if __name__ == "__main__":
    failure()
