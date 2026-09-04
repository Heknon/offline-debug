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

- `save_traceback(exc: BaseException, file: Path | BytesIO, proxy_types=DEFAULT_PROXY_TYPES)`:
  Serializes an exception, its traceback, and all picklable local/global variables to a binary file or buffer.
  `proxy_types` names classes whose instances must never be touched (see
  [Object proxies](#object-proxies)).
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

### Object proxies

An object proxy such as an [`rpyc`](https://rpyc.readthedocs.io/) netref forwards
*every* instance operation to a remote peer: reading an attribute, `repr`, `str`,
`isinstance` (through `__class__`) and pickling (through `__reduce_ex__`, which is
how `rpyc.classic.obtain` fetches a value). Each one is a synchronous request that,
on a broken connection, blocks for the peer's full timeout - and a proxy sitting in
a crashing frame's locals is common, since a dead peer is often *why* the test
failed. A save that pickles it to see whether it round-trips, then calls `repr` on
it to build a placeholder, waits out two timeouts per proxy and may lose the dump
entirely.

`save_traceback` therefore recognises proxies from their **type alone** and writes
a placeholder naming the class and identity, without touching the instance:

```python
save_traceback(e, dump_file, proxy_types=(*DEFAULT_PROXY_TYPES, MyProxyBase))
```

An entry is a class or a fully-qualified class name such as
`"rpyc.core.netref.BaseNetref"`; a name lets you guard against a package this code
does not import. Either form also matches subclasses, because only
`type(value).__mro__` is consulted. `DEFAULT_PROXY_TYPES` covers rpyc netrefs out of
the box. Proxies are replaced wherever they appear: frame variables, items nested
in containers, and an exception's `args` or attributes. In the dump they read as
`<proxy rpyc.core.netref.SaharaClient at 0x7f...>`.

A hostile value that is *not* registered still cannot lose the dump: the
placeholder for an unpicklable value falls back to the bare object repr when
`repr` itself fails, and the fallback for an unpicklable exception does the same
for `str`. It can still be slow, though - the save has to try the value before it
can give up on it - which is what registering the type avoids.

## Technical Implementation

- **True Frame Reconstruction**: Uses `ctypes` to call `PyFrame_New` from the Python C API. This creates real `frame` objects
  which are required for a valid `types.TracebackType`.
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


