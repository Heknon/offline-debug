"""
Build the code object a reconstructed frame runs on.

A reconstructed frame cannot carry the original code object: accessing ``f_locals``
on a frame that ``PyFrame_New`` built over optimized bytecode (any function body)
segfaults, because the frame's "fast" locals array is never initialized. So every
reconstructed frame gets a synthetic code object whose bytecode is inert, and which
keeps only the metadata a debugger or traceback printer reads: file name, function
name, and the source position of the failing instruction.

The compiler builds that code object. The synthetic code is a function whose only
statement is ``return None``, compiled from an AST whose nodes all carry the failing
position, so the compiler writes a location table that places every instruction
there. This keeps the module free of the location table's byte format, which is
private to CPython and has been rewritten more than once (3.10, 3.11): whatever
format the running interpreter uses, its own compiler produces it.

Two properties of the compiled function are then adjusted:

- Its ``CO_OPTIMIZED``/``CO_NEWLOCALS`` flags are cleared, so the frame reads its
  locals from the dictionary handed to ``PyFrame_New`` (as module-level code does)
  rather than from the fast locals array that must never be touched.
- Its ``co_name``/``co_qualname`` are replaced with the original names, which are not
  always valid identifiers (``<module>``, ``<lambda>``, ``<listcomp>``).

One consequence is that ``co_firstlineno`` is the failing line rather than the line
the original function was defined on: the compiler pins the function's ``RESUME``
instruction to its ``def`` line, and ``frame.f_lineno`` reports that instruction's line
on a frame that never ran, so the header has to sit on the failing line for
``f_lineno``, ``tb_lineno`` and every printer to agree.
"""

from __future__ import annotations

import ast
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import CodeType

# A source position as ``co_positions()`` reports it: start line, end line, and the
# start and end columns, which are ``None`` when the code carries no column information.
Position = tuple[int, int, int | None, int | None]

# Size in bytes of one bytecode instruction ("code unit"), the unit ``tb_lasti`` counts
# in and the stride between entries of ``co_positions()``. Documented by ``dis``.
CODE_UNIT_SIZE = 2

# The column offset that tells the compiler an AST node has no column information,
# which it records as ``None`` columns, exactly what ``-X no_debug_ranges`` produces.
_NO_COLUMN = -1

# The name the synthetic function is compiled under before ``co_name`` is replaced.
# The AST requires *some* name; the original names are not always valid identifiers.
_PLACEHOLDER_NAME = "reconstructed_frame"

_FLAGS_TO_CLEAR = inspect.CO_OPTIMIZED | inspect.CO_NEWLOCALS


def synthetic_code(
    filename: str, name: str, qualname: str, position: Position
) -> tuple[CodeType, int]:
    """
    Compile an inert code object whose instructions sit at ``position``.

    Returns the code object and the ``tb_lasti`` a traceback entry over it should
    carry: the offset of an instruction the compiler placed exactly at ``position``.
    (The function's ``RESUME`` instruction is pinned by the compiler to the failing
    line but column 0, which is why the first instruction is not always usable.)

    ``position`` must be one CPython itself could have recorded: a non-negative start
    line, an end line at or after it, and either both columns or neither.
    """
    lineno, end_lineno, col, end_col = position

    def located[NodeT: (ast.stmt, ast.expr)](node: NodeT) -> NodeT:
        node.lineno = lineno
        node.end_lineno = end_lineno
        node.col_offset = _NO_COLUMN if col is None else col
        node.end_col_offset = _NO_COLUMN if end_col is None else end_col
        return node

    function = located(
        ast.FunctionDef(
            name=_PLACEHOLDER_NAME,
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[located(ast.Return(value=located(ast.Constant(value=None))))],
            decorator_list=[],
            type_params=[],
        )
    )
    module = compile(ast.Module(body=[function], type_ignores=[]), filename, "exec")
    code = next(const for const in module.co_consts if inspect.iscode(const))
    code = code.replace(
        co_name=name, co_qualname=qualname, co_flags=code.co_flags & ~_FLAGS_TO_CLEAR
    )

    lasti = list(code.co_positions()).index(position) * CODE_UNIT_SIZE
    return code, lasti
