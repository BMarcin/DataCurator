# Fix-stage improvements: flag spike + invented words

This page plans the next round of quality fixes for the gated translation
pipeline, based on the `gate_queries_trivia_no_thinking_batching3` run — the
first with the hardened prompts and the batch item-id echo. Those guards
worked (batch swaps gone, kiwi-class answer leaks gone, metricX ≥10 tail
36→20), and the remaining losses now concentrate in two places: a validator
misconfiguration that flags correct fixes, and fix rewrites that invent
Polish non-words.

## What the batching3 run showed

The monitor raised a flagged-rate spike at `fix_document`. Grouping all 292
flagged records from the finalize output:

- **~205 flags: `CharacterCountValidator` on `'?'`.** The messy trivia
  sources often lack terminal punctuation ("What was the nickname of Captain
  Pierce in the TV series MASH"); the Google draft adds the `?` a Polish
  question needs, the fix keeps it — and the validator demands an *exact*
  `?` count against the English source. A correct fix fails deterministically
  and burns every retry. Mismatches are almost all off-by-one (137× "1 vs 0",
  40× "2 vs 1", 11× "1 vs 2", 10× "0 vs 1").
- **~20 flags: `SimilarityValidator` on `improved_polished`.** Legitimate
  restructured rewrites measure 0.19–0.30 char-similarity to the draft
  (Polish reordering re-inflects most words), below the 0.3 floor. Sampled
  rejects were *better* translations than the drafts. The one
  `improved_primary` flag (0.29 < 0.45) was genuine garble — that floor works.

Separately, the new worst tail is dominated by **invented Polish non-words**
from fix rewrites — "nowog", "dziewięn", "głade oczko", "zwyzwyczajny" —
plus merged-word edit residue ("którejktórego") and doubled words
("posiada ma dudkę"). LanguageTool's Polish spellcheck catches this family
mechanically, but today LT only checks the Google draft at the gate, never
the fix outputs.

## Change 1 — tolerate sentence-final `?` / `!`

Split the fix-stage `CharacterCountValidator` in two:

```yaml
- _target_: DataCurator.pipeline.validators.CharacterCountValidator
  fields: [improved_primary, improved_polished]
  reference: source_text
  characters: ['\', '[', ']', '{', '}', '|', '@']   # structural: exact
- _target_: DataCurator.pipeline.validators.CharacterCountValidator
  fields: [improved_primary, improved_polished]
  reference: source_text
  characters: ['!', '?']
  max_diff: 1                                        # sentence-final drift
```

The strict entry keeps the original target (backslash-escape artifacts);
the tolerant entry admits exactly the observed legitimate divergence while
still failing pathological cases (a "1 vs 3" flood stays caught). Applies
to every gated config carrying the fix validator block —
`gate_queries_trivia_no_thinking_batching3.yaml` and its batching/batching2/
documents mirrors (grep for the block).

## Change 2 — lower the polished similarity floor

`improved_polished` `min_similarity`: **0.3 → 0.15**. `improved_primary`
stays 0.45 (a minimal edit must stay draft-close) and the gate stays 0.5.
Rationale: legit rewrites measured ≥0.19, and the fix stage is unbatched, so
there is no cross-record swap to detect — the floor's only job there is
rejecting nonsense (unrelated text, near-empty output), which sits well
below 0.15.

## Change 3 — `LanguageToolValidator`, detection-only

A new `ResponseValidator` in `DataCurator/pipeline/validators.py` runs a
**narrow allow-list** of LT rules (Polish spellcheck + word repetition —
not the full inventory) over `improved_primary` / `improved_polished`.

- **Detection only.** It calls `LanguageToolChecker.get_issues`
  (`DataCurator/LanguageTool/LanguageTool.py`) — never the fixing methods.
  A surviving finding raises `ResponseValidationError`, so the LLM
  regenerates the rewrite; exhausted retries flag the record. LT never
  edits the text, so a wrong LT suggestion can never enter the dataset.
- **`required: true` by default.** An unreachable LT server fails
  validation and flags records, rather than silently disabling the guard.
- **False-positive filters, applied before rejection:**
  - `suppress_source_field: source_text` drops findings whose span appears
    verbatim (case-insensitively) in the source. Note the reference is a
    *different language* than the validated fields — English vs Polish —
    so the only spans that can match are terms copied through verbatim
    ("mangel worzel", brand names), which the prompts mandate keeping.
    Reuse `_suppress_verbatim` from
    `DataCurator/pipeline/modifiers/languagetool.py`, adapted to match
    case-insensitively so a sentence-case copy still suppresses.
  - `skip_capitalized` drops capitalized unknowns (proper nouns).
  - Rule ids are **probed, not assumed**: run the known bad outputs
    ("nowog", "głade oczko", "posiada ma dudkę", "którejktórego") and a
    clean control set against the LT server to fix the `allowed_fixes`
    list — the same probing that built the gate deny-list.
- **Prerequisites.** `ResponseValidator.validate` becomes async (call
  sites: `pipeline/stages/llm.py` `_validate` and the per-item validation
  in `distributed/batch.py`), so the LT HTTP call can run via
  `asyncio.to_thread` like the gate modifier does; and the worker's
  `inject_bindings` (`distributed/worker.py`) must patch `remote_server`
  on validator specs too — today it only patches modifiers, so under the
  controller's `bindings: none` a validator's
  `${oc.select:_bindings.languagetool.remote_server,null}` resolves to null.

Config sketch for the fix stage (fix-only for now; the gate rewrite is a
minimal edit of an already-LT-checked draft):

```yaml
- _target_: DataCurator.pipeline.validators.LanguageToolValidator
  fields: [improved_primary, improved_polished]
  language: pl-PL
  allowed_fixes: [<probed spellcheck + repetition rule ids>]
  suppress_source_field: source_text
  remote_server: ${oc.select:_bindings.languagetool.remote_server,null}
```

## Change 4 — `LaBSEValidator`, cross-lingual adequacy gate

Every fix-stage guard today is **meaning-blind**: `LengthRatioValidator` and
the two `SimilarityValidator`s measure the rewrite against the Polish *draft*
(length, edit-distance), `CharacterCountValidator` counts invariant marks, and
`SimilarityValidator` explicitly requires a *same-language* reference
(`validators.py`) — so none can see a fluent rewrite that silently drops a
clause or mistranslates a term. That adequacy drift is the residue MetricX
still flags in the worst tail, and nothing rejects it inside the loop. Change 3
catches invented non-words and mechanics; this catches meaning.

LaBSE (`sentence-transformers/LaBSE`) embeds both languages into one space, so
`cosine(source_text, improved_*)` is a cheap source↔translation adequacy
score. A new `LaBSEValidator` reuses `SimilarityValidator`'s shape (`fields`,
`reference`, `min_similarity`, dotted paths, skip on missing/empty reference)
but scores **across languages against the source** — the exact reference
`SimilarityValidator` forbids.

- **Hosted like LanguageTool.** LaBSE runs as a separate HTTP microservice
  (`POST /similarity {source, hypotheses[]} -> {scores[]}`, server-side
  batching); the validator calls it over async httpx, one request per record.
  The endpoint is a worker-env binding
  (`remote_server: ${oc.select:_bindings.labse.remote_server,null}`), so it
  never enters the stage sig. LaBSE is ~470M params — CPU-fine, no VRAM
  contention with vLLM; the pipeline stays model-free, every heavy call still
  an outbound HTTP request like vLLM and LT.
- **Fail-open on outage.** Unlike the LT validator's `required: true`, a
  connection error **skips** the check (after a short bounded retry, like the
  LT checker's) rather than raising — an advisory adequacy signal is not worth
  burning LLM resamples during a service blip, and a real outage still surfaces
  through the worker's outage breaker. Detect it on httpx's own transport
  errors (`httpx.TransportError`) and 5xx responses, **not** `LanguageTool.py`'s
  `requests`-specific `_is_connection_error` (it matches neither). `required:
  true` is available when a hard gate is wanted.
- **Threshold is probed, not assumed.** A faithful EN↔PL pair scores ~0.7–0.85
  cosine; set `min_similarity` from LaBSE scores on the known-bad outputs plus
  a clean control set — the same discipline that built the gate deny-list and
  the Change 3 rule allow-list. The `0.6` below is a placeholder.
- **Prerequisites.** The same two as Change 3 — async
  `ResponseValidator.validate` (a **breaking signature change**: every call
  site *and* every existing validator's unit test moves to `await`), and
  `inject_bindings` (`DataCurator/distributed/worker.py`) patching
  `remote_server` on validator specs, matched by the `LaBSEValidator`
  `_target_`, from a **new `labse:` section** added to the bindings files
  (`config/bindings/*.yaml`, `bindings.example.yaml` — the LT section already
  exists, this one is net-new). No `wire.py` change: `remote_server` is already
  a `wire.BINDING_KEYS` member, and `strip_null_bindings` drops the
  null-resolved key before the behaviour/bindings leak check. No new capability
  tag either — reachability is a binding concern like LT, so the stages'
  `requires` is unchanged.

Config sketch for the fix stage (unbatched — the validator runs via
`llm.py`'s `_validate`):

```yaml
- _target_: DataCurator.pipeline.validators.LaBSEValidator
  fields: [improved_primary, improved_polished]
  reference: source_text            # English — the RenameModifier already sets it
  min_similarity: 0.6               # placeholder — calibrate, do not assume
  required: false                   # fail-open on service outage
  remote_server: ${oc.select:_bindings.labse.remote_server,null}
```

**Insertion points.** Add the same block to the **gate stage's
`improved_translation`** — the minimal-fix path that reaches finalize without
passing through fix — with `skip_empty_fields: true`, since `""` is its no-fix
sentinel (that stage is batched, so the validator runs per item via
`batch.py`). The **pick stage needs no gate**: its two candidates are the
already-gated fix rewrites, and its output is only an id — `SelectByIdModifier`
resolves the chosen text in an *after* modifier that runs *after* `validate`,
so there is no translation text in scope for the validator to score. Roll the
block out to the same config set as Changes 1–3 (batching3 and its
batching/batching2/documents mirrors).

**Signal caveat.** LaBSE is sentence-level, so on short trivia *queries* the
cosine is coarser than on full sentences — calibrate the floor on the actual
query-length distribution, and expect the gate to bite harder on the documents
run (the same trivia-vs-documents split MetricX already notes).

## Verification & rollout

Verification stays unit-level: the LT rule-id probe script; validator tests
with a stubbed checker (findings raise with rule ids in the message,
source-verbatim suppression, capitalized skip, connection failure with
`required: true`); for the LaBSE gate, validator tests with a stubbed httpx
client (a faithful pair passes, a below-threshold field raises, a connection
error skips when `required: false`), a service sanity check (a garbled EN↔PL
pair scores worse than a good one), and the score-calibration probe that sets
`min_similarity`; the full pytest suite. The async-`validate` refactor must
leave the three existing validators behaviour-identical (they gain `async`,
await nothing). Rollout follows the usual caveat:
code and config hash into the stage sigs, so the changes take effect on the
next commit + worker restart, and the fix stage recomputes on the next
seeded job.
