# Traceback Serializer Project (`offline-debug`)

[![PyPI version](https://img.shields.io/pypi/v/offline-debug.svg)](https://pypi.org/project/offline-debug/)
[![Tests](https://github.com/INTODAN/offline-debug/actions/workflows/ci.yml/badge.svg)](https://github.com/INTODAN/offline-debug/actions)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/INTODAN/offline-debug)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Ty checked](https://img.shields.io/badge/ty-checked-blue.svg)](https://github.com/astral-sh/ty)

## Overview

A Python package for high-fidelity serialization and deserialization of exceptions and their complete tracebacks. Unlike other
solutions, `offline-debug` reconstructs **actual** `types.FrameType` objects using the Python C API, ensuring that re-raised
exceptions look and feel genuine to debuggers and introspection tools.

## Core Functions

- `save_traceback(exc: BaseException, file: Path | BytesIO)`:
  Serializes an exception, its traceback, and all picklable local/global variables to a binary file or buffer.
- `load_traceback(file: Path | BytesIO) -> Never`:
  Loads the serialized state, reconstructs the exception and its full traceback chain (including `__cause__` and `__context__`),
  and raises it.
- `parse_traceback(file: Path | BytesIO) -> ExceptionData`:
  Loads the serialized data and returns an `ExceptionData` object. This allows for inspecting the exception, stack frames, and variables without reconstructing the full traceback or raising the exception.

## Usage Example

To get started, install with:  
`pip install offline-debug` or `uv add offline-debug`

```python
from pathlib import Path
from offline_debug import save_traceback, load_traceback, parse_traceback

# --- Saving an exception ---
try:
    some_complex_operation()
except Exception as e:
    save_traceback(e, Path("crash_report.dump"))

# --- Option 1: Re-raise the exception for debugging ---
# This will look like the original crash in your debugger
load_traceback(Path("crash_report.dump"))

# --- Option 2: Inspect data without raising ---
data = parse_traceback(Path("crash_report.dump"))
print(f"Number of frames: {len(data.tb_frames)}")
for frame in data.tb_frames:
    print(f"File: {frame.code.co_filename}, Line: {frame.lineno}")
```

### Exception Group Support

`offline-debug` has full support for `ExceptionGroup` (Python 3.11+). When you parse a saved `ExceptionGroup`, you can access its nested exceptions:

```python
from offline_debug import parse_traceback, ExceptionGroupData

data = parse_traceback(Path("exception_group.dump"))

if isinstance(data, ExceptionGroupData):
    print(f"Group contains {len(data.exceptions)} sub-exceptions")
    for sub_exc_data in data.exceptions:
        # Each sub_exc_data is itself an ExceptionData object
        print(f"Sub-exception frames: {len(sub_exc_data.tb_frames)}")
```

### Cyclic and Shared Exception Graphs

`__cause__`/`__context__` form a directed graph, not a tree. `raise X from Y` sets **both**
`X.__cause__` and `X.__context__` to `Y`, and `raise e from e` makes an exception its own cause,
so a saved graph may share nodes or contain cycles. `offline-debug` records that graph faithfully:
each exception is saved exactly once, and a cycle round-trips as a real cycle.

This means consumers must not walk `cause`/`context` naively — a recursive walk will not terminate,
and `dataclasses.asdict()` / `json.dumps()` fail outright on a cycle. For the same reason the data
nodes compare and hash by identity rather than by value (a structural `==` would recurse through
the cycle forever), so two dumps of the same exception are never equal; `id()` — or the node itself,
as a dict key or set member — is the stable handle. Use `walk_exception_data`, which visits each
node exactly once and never recurses:

```python
from offline_debug import parse_traceback, walk_exception_data

data = parse_traceback(Path("crash_report.dump"))

# Stable ids let you emit references instead of nesting - this is what makes a
# cyclic graph representable in JSON.
ids = {id(node): i for i, node in enumerate(walk_exception_data(data))}

payload = [
    {
        "id": ids[id(node)],
        "frames": len(node.tb_frames),
        "cause_id": ids[id(node.cause)] if node.cause else None,
        "context_id": ids[id(node.context)] if node.context else None,
    }
    for node in walk_exception_data(data)
]
```

### Dump Compatibility

The on-disk layout gained one optional field in 0.4.0 (`suppress_context`, defaulting to
`False`); nothing was removed or renamed. Dumps written by earlier versions load, raise and
print normally, and they gain the traceback-formatting fix, having previously failed to format
at all.

What an old dump cannot gain is anything the writer never recorded: it stored a separate copy
of every shared exception, and no `suppress_context` at all. So it still expands to more nodes
than a fresh save, and a `raise X from None` in it still prints the context its author
suppressed. Re-saving under 0.4.0 fixes both.

New dumps are readable by 0.4.0 and later. An older release can still read a new dump of an
ordinary exception — it ignores the added field, so suppression is simply not restored — but
not one whose graph genuinely contains a cycle (`raise e from e`): the pre-0.4.0 reader follows
`__cause__` forever and dies with a `RecursionError`.

## Technical Implementation

- **True Frame Reconstruction**: Uses `ctypes` to call `PyFrame_New` from the Python C API. This creates real `frame` objects
  which are required for a valid `types.TracebackType`.
- **Line-level Position Fidelity**: Reconstructed frames carry a synthetic code object (real optimized
  bytecode would segfault on `f_locals` access), so they report no bytecode offset and the stdlib resolves
  positions from the restored line numbers. Tracebacks print with correct files, lines and functions; the
  `^^^^` column markers are not available for reconstructed frames.
- **Python 3.13 Compatibility**: Leverages PEP 667 features where `f_locals` is a write-through proxy, allowing for accurate local
  variable restoration.
- **Support python 3.12 as well**
- **Resilient Serialization**:
    - `pickle` is used for exceptions and variables.
    - `marshal` is used for code objects.
    - Non-picklable items are gracefully handled by storing their `repr`.

## Development & Tooling

- **Package Manager**: `uv`
- **Minimum Python**: 3.12
- **Testing**: `pytest`
- **Commands**:
    - Add dependencies: `uv add <package>`
    - Run tests: `uv run pytest`


