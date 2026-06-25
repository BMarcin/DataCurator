"""Config-driven record filters that select which inputs a stage processes.

A :class:`RecordFilter` inspects one input record (a plain ``dict`` read from
the dataset or a prior stage's output) and decides whether the runner should
*keep* it — i.e. feed it through the stage — or drop it. Filters are declared
per stage as runner metadata, mirroring how stage ``modifiers``/``validators``
are declared, and Hydra instantiates them from their ``_target_``:

    pipeline:
      fix_queries:
        enabled: true
        filters:
          - _target_: DataCurator.pipeline.filters.ExpressionFilter
            expression: "error == True"
        stage: { ... }

Several filters on one stage combine with **AND** (every filter must keep a
record for it to survive). The runner applies them to a stage's input *before*
the resume/limit logic, so totals, progress and the dashboard all reflect the
filtered set.

The motivating case is rerunning failed items: a previous run wrote flagged
records carrying ``error: true``; a fresh experiment reads that output and
keeps only ``error == True`` to reprocess exactly the failures. The expression
form supports full boolean logic (``and``/``or``/``not``, comparisons, ``in``),
not just a boolean field test, so richer selections — ``error == True and
error_stage == "fix_queries"``, ``score >= 0.5 or lang in ["pl", "en"]`` — are
expressible without writing code.
"""
from __future__ import annotations

import ast
import operator
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Type

from loguru import logger


class FilterExpressionError(ValueError):
    """A filter expression is malformed or uses a disallowed construct.

    Raised at filter construction (when the pipeline is built), not mid-run, so
    a typo'd or unsafe expression fails fast like a missing prompt template.
    """


class _MissingType:
    """Sentinel for a field a record does not contain.

    It is *falsy* (so a bare ``error`` over a record without that key drops it)
    and compares unequal to every real value (so ``error == True`` is simply
    ``False`` for such a record rather than raising). Ordering comparisons
    against it raise ``TypeError``, which the evaluator turns into ``False``.
    """

    __slots__ = ()

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "MISSING"


_MISSING = _MissingType()


class RecordFilter(ABC):
    """Decide whether the runner should process a given input record.

    Subclasses implement :meth:`keep`, returning ``True`` to feed the record
    through the stage and ``False`` to drop it. Config-driven via ``_target_``,
    exactly like :class:`~DataCurator.pipeline.validators.ResponseValidator`.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or type(self).__name__

    @abstractmethod
    def keep(self, record: Mapping) -> bool:
        """Return ``True`` to keep ``record``, ``False`` to drop it."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# --------------------------------------------------------------------------- #
# Safe expression evaluation
# --------------------------------------------------------------------------- #
# A tiny allow-listed evaluator over Python's own ``ast`` so an expression can
# use familiar syntax (``and``/``or``/``not``, comparisons, ``in``, arithmetic)
# without ``eval`` and without reaching any Python object: names resolve to
# record fields, and ``a.b`` / ``a[b]`` are mapping lookups, never attribute
# access. Neither ``simpleeval`` nor ``asteval`` is a project dependency, so
# this is hand-rolled like the rest of the pipeline's small helpers.

_BINOPS: Dict[Type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_CMPOPS: Dict[Type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_UNARYOPS: Dict[Type[ast.unaryop], Callable[[Any], Any]] = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
# Bare functions an expression may call (no method calls, no other names).
_ALLOWED_CALLS: Dict[str, Callable[..., Any]] = {
    "len": len,
    "abs": abs,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}
# Lowercase JSON/YAML-style literals, used only when the record has no such key.
_NAME_ALIASES: Dict[str, Any] = {"true": True, "false": False, "null": None, "none": None}

# Node types the validation walk permits; anything else is rejected up front.
_ALLOWED_NODES: tuple = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.BinOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Attribute,
    ast.Subscript,
    ast.Call,
    *_BINOPS,
    *_CMPOPS,
    *_UNARYOPS,
)


def _validate_node(node: ast.AST) -> None:
    """Reject any node outside the allow-list (and any non-allow-listed call)."""
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise FilterExpressionError(
                f"disallowed expression construct: {type(child).__name__}"
            )
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in _ALLOWED_CALLS:
                name = getattr(child.func, "id", type(child.func).__name__)
                raise FilterExpressionError(f"disallowed function call: {name!r}")
            if child.keywords:
                raise FilterExpressionError("keyword arguments are not supported in filters")


def _eval(node: ast.AST, record: Mapping) -> Any:
    """Evaluate an allow-listed AST node against ``record``."""
    if isinstance(node, ast.Expression):
        return _eval(node.body, record)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in node.values:
                result = _eval(value, record)
                if not result:
                    return result  # short-circuit, preserving the falsy value
            return result
        result = False
        for value in node.values:
            result = _eval(value, record)
            if result:
                return result
        return result
    if isinstance(node, ast.UnaryOp):
        return _UNARYOPS[type(node.op)](_eval(node.operand, record))
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_eval(node.left, record), _eval(node.right, record))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, record)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval(comparator, record)
            try:
                outcome = _CMPOPS[type(op)](left, right)
            except TypeError:
                # e.g. ordering a missing field; treat as not-satisfied, not a crash.
                outcome = False
            if not outcome:
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        if isinstance(record, Mapping) and node.id in record:
            return record[node.id]
        return _NAME_ALIASES.get(node.id, _MISSING)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(element, record) for element in node.elts]
    if isinstance(node, ast.Set):
        return {_eval(element, record) for element in node.elts}
    if isinstance(node, ast.Attribute):
        value = _eval(node.value, record)
        if isinstance(value, Mapping) and node.attr in value:
            return value[node.attr]
        return _MISSING
    if isinstance(node, ast.Subscript):
        value = _eval(node.value, record)
        key = _eval(node.slice, record)
        try:
            return value[key]
        except (KeyError, IndexError, TypeError):
            return _MISSING
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)  # guaranteed by _validate_node
        try:
            return _ALLOWED_CALLS[node.func.id](*(_eval(arg, record) for arg in node.args))
        except (TypeError, ValueError):
            return _MISSING
    raise FilterExpressionError(f"unsupported expression node: {type(node).__name__}")


class ExpressionFilter(RecordFilter):
    """Keep a record when a boolean ``expression`` evaluates truthy over it.

    The expression is ordinary Python-flavoured syntax evaluated against the
    record's fields: bare names are field lookups (``error``, ``error_stage``),
    ``a.b`` / ``a["b"]`` index nested mappings, and ``and``/``or``/``not``,
    comparisons (``== != < <= > >= in not in``) and arithmetic are supported,
    along with the helpers ``len``/``abs``/``str``/``int``/``float``/``bool``.
    It is parsed and allow-list-validated at construction (so a bad expression
    fails when the pipeline is built), then evaluated per record.

    A field absent from the record reads as a falsy sentinel that compares
    unequal to every value, so ``error == True`` simply drops records without
    an ``error`` key instead of raising. ``true``/``false``/``null`` are
    accepted as aliases for ``True``/``False``/``None`` (only when no field of
    that name exists).
    """

    def __init__(self, expression: str, *, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.expression = expression
        try:
            self._tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise FilterExpressionError(f"could not parse filter expression {expression!r}: {exc}") from exc
        _validate_node(self._tree)

    def keep(self, record: Mapping) -> bool:
        outcome = bool(_eval(self._tree, record))
        logger.debug(f"{self.name}: {self.expression!r} -> {outcome}")
        return outcome

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, expression={self.expression!r})"
