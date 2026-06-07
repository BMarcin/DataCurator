"""Generic pipeline-stage and stage-modifier abstractions.

A :class:`Stage` is one self-contained processing step — running an LLM
prompt, applying regex fixes, calling a translation API, etc. Its public
entry point is :meth:`Stage.run`, which subclasses drive by implementing
:meth:`Stage.process`.

A :class:`StageModifier` hooks into a stage's execution and mutates the
variables flowing through it *on the fly*. The motivating example is a
LanguageTool modifier attached to an LLM stage: it normalises a field
``before`` the prompt is built so the model sees cleaned text, without
the LLM stage knowing anything about LanguageTool. Modifiers never
re-implement a stage — they adjust its inputs (``before`` phase) and/or
its outputs (``after`` phase).

The variables both sides operate on live in a :class:`StageContext`, a
small mutable mapping with explicit :meth:`~StageContext.add` /
:meth:`~StageContext.replace` / :meth:`~StageContext.set` helpers so the
intent of every modification is obvious at the call site.

The API is async-first because the primary stages (LLM, translation) are
IO-bound. Purely synchronous work — regex fixes, LanguageTool — slots in
just as easily: implement the body without awaiting anything.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, MutableMapping
from enum import Enum
from typing import Any, Callable, Dict, Optional, Sequence

from loguru import logger


# Sentinel distinguishing "argument not supplied" from an explicit ``None``.
_MISSING: Any = object()


class StageContext(MutableMapping):
    """A mutable bag of named variables flowing through a stage.

    Behaves like a ``dict`` (supports ``ctx["x"]``, ``in``, iteration,
    ``len`` ...) but adds intent-revealing helpers:

    * :meth:`add` — introduce a *new* variable; errors if it exists.
    * :meth:`replace` — overwrite an *existing* variable; errors if absent.
    * :meth:`set` — add-or-overwrite, no questions asked.

    Modifiers use these to make clear whether they are enriching the
    context with something new or rewriting a value the stage will read.
    """

    def __init__(self, data: Optional[Mapping[str, Any]] = None, **variables: Any) -> None:
        self._data: Dict[str, Any] = dict(data or {})
        if variables:
            self._data.update(variables)

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    # -- Intent-revealing mutators -----------------------------------------
    def add(self, name: str, value: Any) -> None:
        """Add a brand-new variable. Raise ``KeyError`` if ``name`` exists."""
        if name in self._data:
            raise KeyError(f"variable {name!r} already exists; use replace()/set()")
        self._data[name] = value

    def replace(self, name: str, value: Any) -> Any:
        """Overwrite an existing variable, returning the previous value.

        Raise ``KeyError`` if ``name`` is not already present — use this
        when a modifier is meant to *transform* a value the stage produces
        or consumes, so a typo'd field name fails loudly.
        """
        if name not in self._data:
            raise KeyError(f"variable {name!r} does not exist; use add()/set()")
        old = self._data[name]
        self._data[name] = value
        return old

    def set(self, name: str, value: Any) -> None:
        """Add the variable if new, overwrite it otherwise."""
        self._data[name] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the underlying variables."""
        return dict(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"


class ModifierPhase(str, Enum):
    """When a modifier runs relative to the stage's core processing."""

    BEFORE = "before"  # adjust the stage's inputs
    AFTER = "after"  # adjust the stage's outputs


class StageModifier(ABC):
    """Mutates the variables a stage operates on, on the fly.

    Subclasses implement :meth:`modify`, reading/writing the shared
    :class:`StageContext`. :attr:`phases` declares whether the modifier
    runs ``before`` processing (to adjust inputs), ``after`` (to adjust
    outputs), or both; :meth:`Stage.run` only calls a modifier for the
    phases it declares.

    For the common "rewrite one field" case prefer :class:`FieldModifier`,
    which handles the add/replace bookkeeping for you.
    """

    #: Phases this modifier participates in. Subclasses typically narrow this.
    phases: Sequence[ModifierPhase] = (ModifierPhase.BEFORE,)

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or type(self).__name__

    @abstractmethod
    async def modify(self, context: StageContext, phase: ModifierPhase) -> None:
        """Mutate ``context`` for the given ``phase``.

        Use :meth:`StageContext.add` to introduce a new variable and
        :meth:`StageContext.replace` / :meth:`StageContext.set` to
        overwrite an existing one.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, phases={tuple(self.phases)!r})"


class FieldModifier(StageModifier):
    """Transform a single named field in the context.

    Subclasses implement :meth:`transform`; the result is written back to
    ``source`` (an in-place rewrite) or, when ``target`` is given, to that
    field instead — added if new, replaced if it already exists. This is
    exactly the shape a LanguageTool/regex pre-pass needs: read one field,
    clean it, put it back.

    If ``source`` is absent from the context the modifier is a no-op,
    unless ``required=True``, in which case a ``KeyError`` is raised.
    """

    def __init__(
        self,
        source: str,
        target: Optional[str] = None,
        *,
        phase: ModifierPhase = ModifierPhase.BEFORE,
        required: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.source = source
        self.target = target or source
        self.required = required
        self.phases = (phase,)

    @abstractmethod
    async def transform(self, value: Any, context: StageContext) -> Any:
        """Return the new value for the field given its current ``value``."""

    async def modify(self, context: StageContext, phase: ModifierPhase) -> None:
        if self.source not in context:
            if self.required:
                raise KeyError(f"{self.name}: required field {self.source!r} missing from context")
            return
        new_value = await self.transform(context[self.source], context)
        context.set(self.target, new_value)


class Stage(ABC):
    """One self-contained processing step in a pipeline.

    Subclasses implement :meth:`process`, the core work, which reads its
    inputs from the :class:`StageContext` and writes its outputs back into
    it. Callers invoke :meth:`run`, which wraps ``process`` with the
    attached modifiers: every ``before`` modifier fires first (adjusting
    inputs), then ``process`` runs, then every ``after`` modifier fires
    (adjusting outputs). The fully-populated context is returned.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        modifiers: Optional[Sequence[StageModifier]] = None,
    ) -> None:
        self.name = name or type(self).__name__
        self.modifiers: list[StageModifier] = list(modifiers or [])
        # Optional observer, called as ``cb(modifier_index, phase)`` after each
        # modifier runs on a record. Lets a runner drive a progress bar without
        # the stage knowing anything about it. ``None`` means "no reporting".
        self._progress_callback: Optional[Callable[[int, "ModifierPhase"], None]] = None

    def set_progress_callback(
        self, callback: Optional[Callable[[int, "ModifierPhase"], None]]
    ) -> None:
        """Attach (or, with ``None``, detach) the per-modifier progress observer."""
        self._progress_callback = callback

    def add_modifier(self, modifier: StageModifier) -> "Stage":
        """Attach a modifier; returns ``self`` so calls can be chained."""
        self.modifiers.append(modifier)
        return self

    @abstractmethod
    async def process(self, context: StageContext) -> None:
        """Perform the stage's core work, mutating ``context`` in place.

        Read input variables via ``context[...]`` and publish results with
        ``context.set(...)`` / ``context.add(...)``.
        """

    async def run(self, data: Any = None, **variables: Any) -> StageContext:
        """Run the stage end to end and return the resulting context.

        ``data`` may be an existing :class:`StageContext`, any mapping (its
        items seed a fresh context), or ``None``. Extra keyword arguments
        are merged in as additional variables, overriding ``data``.
        """
        context = self._as_context(data, variables)
        await self._apply_modifiers(context, ModifierPhase.BEFORE)
        await self.process(context)
        await self._apply_modifiers(context, ModifierPhase.AFTER)
        return context

    async def _apply_modifiers(self, context: StageContext, phase: ModifierPhase) -> None:
        """Run every attached modifier registered for ``phase``, in order."""
        for index, modifier in enumerate(self.modifiers):
            if phase not in modifier.phases:
                continue
            logger.debug(f"{self.name}: applying modifier {modifier.name} ({phase.value})")
            await modifier.modify(context, phase)
            if self._progress_callback is not None:
                self._progress_callback(index, phase)

    @staticmethod
    def _as_context(data: Any, variables: Mapping[str, Any]) -> StageContext:
        """Coerce ``data`` into a :class:`StageContext`, merging ``variables``."""
        if isinstance(data, StageContext):
            context = data
        elif data is None:
            context = StageContext()
        elif isinstance(data, Mapping):
            context = StageContext(data)
        else:
            raise TypeError(
                f"Stage.run expects a StageContext, mapping or None, got {type(data).__name__}"
            )
        if variables:
            context.update(variables)
        return context

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, modifiers={self.modifiers!r})"
