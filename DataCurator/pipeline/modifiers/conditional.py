"""Conditional constant-injection stage modifier.

The conditional sibling of
:class:`~DataCurator.pipeline.modifiers.constants.ConstantsModifier`: where
that one always writes its literals, this one writes them only when a
``when`` expression — the same safe language stage filters speak (see
:class:`~DataCurator.pipeline.filters.ExpressionFilter`) — evaluates truthy
over the context. Values are config literals, never computed: anything that
derives a value belongs in a dedicated modifier (like
:class:`~DataCurator.pipeline.modifiers.verdict.DeriveVerdictModifier`);
this one only routes decisions already visible in the context's fields.

Set-paths may be dotted: ``gate_answer.verdict`` rewrites the ``verdict``
key inside the ``gate_answer`` mapping (copy-then-set, missing intermediate
mappings are created). Because the condition pins the prior state, an
override marker needs no machinery — write it as another literal:

    when: 'metricx_draft_score > 10 and gate_answer.verdict == "publishable"'
    set:
      gate_answer.metricx_override: publishable   # marker: pre-override verdict
      gate_answer.verdict: needs_edit

A field absent from the context reads as the filters' missing sentinel —
equality is false and ordering comparisons evaluate false — so a condition
over a field an upstream modifier declined to write (its own `when` was
falsy, or its inputs were missing) simply does not fire. Absence never means
"the service was down": a modifier whose backing service fails raises.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from loguru import logger
from omegaconf import OmegaConf

from DataCurator.pipeline.filters import compile_expression
from DataCurator.pipeline.stage import ModifierPhase, StageContext, StageModifier


def _set_path(context: StageContext, path: str, value: Any) -> None:
    """Write ``value`` at a dotted ``path``, copying nested mappings on the way.

    The first segment is a context field; deeper segments walk mappings,
    which are shallow-copied level by level (the
    :class:`~DataCurator.pipeline.modifiers.verdict.DeriveVerdictModifier`
    copy-then-set idiom) so shared references never mutate in place. Missing
    intermediates are created as empty dicts.
    """
    first, _, rest = path.partition(".")
    if not rest:
        context.set(first, value)
        return
    root = dict(context[first]) if first in context and isinstance(context[first], Mapping) else {}
    node = root
    keys = rest.split(".")
    for key in keys[:-1]:
        child = node.get(key)
        node[key] = dict(child) if isinstance(child, Mapping) else {}
        node = node[key]
    node[keys[-1]] = value
    context.set(first, root)


class SetWhenModifier(StageModifier):
    """Write literal ``set`` values to dotted paths when ``when`` holds."""

    def __init__(
        self,
        when: str,
        set: Mapping[str, Any],
        *,
        phase: ModifierPhase = ModifierPhase.AFTER,
        name: Optional[str] = None,
    ) -> None:
        """Compile the condition and capture the literals to inject.

        ``when`` is validated here, so a malformed expression fails at
        pipeline build like a typo'd filter. ``set`` maps dotted context
        paths to the literal values written when the condition is truthy.
        """
        super().__init__(name=name)
        self.when = when
        self._evaluate = compile_expression(when)
        if OmegaConf.is_config(set):
            set = OmegaConf.to_container(set, resolve=True)  # type: ignore[assignment]
        if not set:
            raise ValueError(f"{self.name}: 'set' needs at least one path -> value entry")
        self.values = dict(set)
        self.phases = (phase,)

    async def modify(self, context: StageContext, phase: ModifierPhase) -> None:
        """Evaluate the condition; on truth, apply every ``set`` entry."""
        if not self._evaluate(context):
            return
        logger.debug(f"{self.name}: {self.when!r} holds; setting {list(self.values)}")
        for path, value in self.values.items():
            _set_path(context, path, value)
