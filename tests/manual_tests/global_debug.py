"""Intended for manual use, run the test in debug mode and view the exceptions."""

import tempfile
from pathlib import Path

from offline_debug import load_traceback, save_traceback

if __name__ == "__main__":
    try:
        raise ValueError("Trigger")
    except ValueError as e:
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_file = Path(tmpdir) / "exception.dump"
            save_traceback(e, dump_file)

            load_traceback(dump_file)
