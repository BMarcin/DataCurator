"""MetricX score stage modifier — signal acquisition only.

Scores one ``(source_field, hypothesis_field)`` pair through the hosted
MetricX-QE service (0–25, lower is better) and writes the float to
``target``. That is the whole job: it takes no decisions. Pair it with a
:class:`~DataCurator.pipeline.modifiers.conditional.SetWhenModifier` whose
expression reads the score to act on it (the gate override), or use it
alone as per-record telemetry — the score lands as a column in the output
JSONL.

An optional ``when`` expression — the same safe language stage filters and
:class:`~DataCurator.pipeline.modifiers.conditional.SetWhenModifier` speak —
gates the scoring: when it evaluates falsy the modifier writes **nothing**,
so the paid MetricX call is spent only on the records that need it (e.g.
``when: 'gate_answer.verdict == "publishable"'`` skips drafts already routed
to fix, whose score no downstream override would read). An absent field
reads as the filters' missing sentinel, so the condition simply does not
fire rather than raising.

A declared modifier always runs: when the service cannot score (unbound
endpoint, outage) :class:`~DataCurator.pipeline.metricx.MetricXUnavailable`
propagates as a connection-class error rather than leaving the target quietly
absent. Silence would be indistinguishable from the ``when``-skip above, and
every ordering comparison against the missing field downstream would evaluate
false — the gate override would disable itself precisely when nobody was
watching.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from DataCurator.pipeline.filters import compile_expression
from DataCurator.pipeline.metricx import MetricXClient
from DataCurator.pipeline.stage import ModifierPhase, StageContext, StageModifier


class MetricXScoreModifier(StageModifier):
    """Write MetricX-QE(``source_field``, ``hypothesis_field``) to ``target``."""

    def __init__(
        self,
        *,
        source_field: str = "source_text",
        hypothesis_field: str = "target_text",
        target: str = "metricx_score",
        remote_server: Optional[str] = None,
        api_token: Optional[str] = None,
        when: Optional[str] = None,
        timeout: float = 180.0,
        retries: int = 3,
        phase: ModifierPhase = ModifierPhase.AFTER,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.source_field = source_field
        self.hypothesis_field = hypothesis_field
        self.target = target
        # Compiled here so a malformed gate expression fails at pipeline
        # build, like a typo'd filter, not on the first record.
        self.when = when
        self._when = compile_expression(when) if when else None
        self.phases = (phase,)
        self._client = MetricXClient(
            remote_server, api_token=api_token, timeout=timeout, retries=retries
        )

    async def modify(self, context: StageContext, phase: ModifierPhase) -> None:
        if self._when is not None and not self._when(context):
            logger.debug(f"{self.name}: {self.when!r} is falsy; skipping scoring")
            return
        source = str(context.get(self.source_field) or "")
        hypothesis = str(context.get(self.hypothesis_field) or "")
        if not source or not hypothesis:
            logger.debug(
                f"{self.name}: {self.source_field!r}/{self.hypothesis_field!r} "
                f"missing or empty; nothing to score"
            )
            return
        # MetricXUnavailable propagates: an unscored record must not look like
        # a deliberately skipped one.
        scores = await self._client.score_pairs([(source, hypothesis)])
        context.set(self.target, scores[0])
