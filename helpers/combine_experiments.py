"""Combine the outputs of several experiments by record ``id``, healing errors.

The pipeline is fault-tolerant: a record that keeps failing a stage is flagged
with ``error: true`` (plus ``error_message``/``error_type``/``error_stage``) and
its pristine original is passed through instead of aborting the run, so a flagged
record lacks the stage's real output. The standard recovery is a *rerun*
experiment (``config/experiment/*_rerun.yaml``) that reprocesses exactly those
failures; its records keep the same ``id`` as the base run, so the fixes can be
merged back by ``id``.

This script does that merge. Pass the base run first, then one or more reruns;
for each ``id`` it keeps the best record across all inputs:

  * a successful record always beats an errored one (a rerun's fix replaces the
    base run's failure);
  * a successful base record is never regressed by a later errored one;
  * when every input still errors an ``id``, the latest attempt's error is kept.

Each positional argument is a stage *directory* (its ``step-*.jsonl`` shards are
read in order, like the runner), a glob of ``.jsonl`` files, or a single file.

Examples:
    # merge a base run with its rerun (final stage of each)
    uv run python helpers/combine_experiments.py \\
        --out data/queries_fiqa_combined.jsonl \\
        dataset/fix_queries_fiqa_no_thinking/01_pick_the_best_translation \\
        dataset/fix_queries_fiqa_no_thinking_rerun/01_pick_the_best_translation

    # same, but drop records still errored after every rerun
    uv run python helpers/combine_experiments.py --drop-still-errored \\
        --out data/queries_fiqa_clean.jsonl \\
        dataset/fix_queries_fiqa_no_thinking/01_pick_the_best_translation \\
        dataset/fix_queries_fiqa_no_thinking_rerun/01_pick_the_best_translation
"""

import argparse
import glob
import sys
from pathlib import Path
from typing import Any

import orjson


def load_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file at ``path`` into a list of dicts, skipping blank lines."""
    items: list[dict] = []
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(orjson.loads(line))
    return items


def load_input(arg: str) -> list[dict]:
    """Resolve one positional argument into an ordered list of records.

    A directory is read as a stage output: its ``step-*.jsonl`` shards are
    concatenated in filename order (matching the runner's shard sequence). A
    pattern containing a glob character is expanded and its files concatenated in
    sorted order. Anything else is treated as a single JSONL file.
    """
    path = Path(arg)
    if path.is_dir():
        files = sorted(path.glob("step-*.jsonl"))
        if not files:
            raise FileNotFoundError(f"{arg!r} is a directory but holds no step-*.jsonl shards")
    elif any(ch in arg for ch in "*?["):
        files = [Path(p) for p in sorted(glob.glob(arg))]
        if not files:
            raise FileNotFoundError(f"glob {arg!r} matched no files")
    else:
        if not path.is_file():
            raise FileNotFoundError(f"{arg!r} is not a file, directory or glob")
        files = [path]
    rows: list[dict] = []
    for file in files:
        rows.extend(load_jsonl(file))
    return rows


def record_key(rec: dict, id_fields: list[str]) -> tuple:
    """Build the merge key for ``rec`` from ``id_fields`` (a composite tuple)."""
    key = []
    for field in id_fields:
        if field not in rec:
            raise KeyError(f"record is missing id field {field!r}: {rec!r}")
        key.append(rec[field])
    return tuple(key)


def is_errored(rec: dict, error_field: str) -> bool:
    """Whether ``rec`` is a flagged failure (truthy ``error_field``)."""
    return bool(rec.get(error_field))


def merge(
    inputs: list[tuple[str, list[dict]]],
    id_fields: list[str],
    error_field: str,
) -> tuple[dict[tuple, dict], dict[str, int]]:
    """Merge records from every input by id, healing errors with later fixes.

    ``inputs`` is an ordered ``(label, records)`` list (base first). A successful
    record always wins over an errored one; among equally-ranked records a later
    input wins only when both are errored (keep the latest attempt) — a
    successful base record is left untouched by a later run. Returns the merged
    ``{key: record}`` map and a stats dict.
    """
    merged: dict[tuple, dict] = {}
    healed: set[tuple] = set()  # keys that went errored -> success during the merge
    for label, records in inputs:
        for rec in records:
            key = record_key(rec, id_fields)
            incoming_err = is_errored(rec, error_field)
            current = merged.get(key)
            if current is None:
                merged[key] = rec
                continue
            current_err = is_errored(current, error_field)
            # Replace on an upgrade (errored -> success) or another errored
            # attempt (errored -> errored, keep the latest). Never regress a
            # success, and keep the first success on a success/success tie.
            if current_err and not incoming_err:
                merged[key] = rec
                healed.add(key)
            elif current_err and incoming_err:
                merged[key] = rec
            else:
                # current is a success: keep it, but warn on an in-run dup clash.
                if not incoming_err:
                    print(
                        f"warning: duplicate id {key} seen again in {label!r}; "
                        f"keeping the first successful record",
                        file=sys.stderr,
                    )
    residual = sum(1 for rec in merged.values() if is_errored(rec, error_field))
    return merged, {"merged": len(merged), "healed": len(healed), "residual": residual}


def _sort_key(rec: dict, id_fields: list[str]) -> Any:
    """Sort key for deterministic output: by id fields, ints before other types."""
    return tuple((isinstance(rec.get(f), str), rec.get(f)) for f in id_fields)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine experiment outputs by id, replacing errored records "
        "with fixed versions from later reruns.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="stage directory, glob or .jsonl file per experiment, base run first",
    )
    parser.add_argument("--out", required=True, type=Path, help="combined JSONL output path")
    parser.add_argument(
        "--id-field",
        default="id",
        help="record id field to merge on; comma-separated for a composite key (default: id)",
    )
    parser.add_argument(
        "--error-field",
        default="error",
        help="field marking a flagged failure (default: error)",
    )
    parser.add_argument(
        "--drop-still-errored",
        action="store_true",
        help="omit records still errored after the merge (clean, successful-only dataset)",
    )
    args = parser.parse_args()

    id_fields = [f.strip() for f in args.id_field.split(",") if f.strip()]

    inputs: list[tuple[str, list[dict]]] = []
    for arg in args.inputs:
        records = load_input(arg)
        errored = sum(1 for r in records if is_errored(r, args.error_field))
        print(f"loaded {len(records):>7} records ({errored} errored) from {arg}", file=sys.stderr)
        inputs.append((arg, records))

    merged, stats = merge(inputs, id_fields, args.error_field)

    out_records = sorted(merged.values(), key=lambda r: _sort_key(r, id_fields))
    if args.drop_still_errored:
        out_records = [r for r in out_records if not is_errored(r, args.error_field)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        for rec in out_records:
            f.write(orjson.dumps(rec, option=orjson.OPT_APPEND_NEWLINE))

    print(
        f"merged {stats['merged']} unique ids: {stats['healed']} healed (errored->fixed), "
        f"{stats['residual']} still errored; wrote {len(out_records)} records to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
