"""Re-attach translated JSONL back onto a mined-negatives HF dataset.

This is the inverse of ``HF_mined_ds_to_jsonl.py``. That script explodes a
mined-negatives dataset (e.g. ``mining-negatives/squadv2-mined-negatives``) into
two flat ``{id, source, text}`` files — one row per query, one per unique
document — which the pipeline then translates, writing a ``text_gtranslate``
field alongside each ``text``.

Here we go the other way: load the *original* dataset and add the translations
back as new columns, joining by id, so every original column is preserved and
the result can be uploaded to HuggingFace. Each row carries a scalar
``query_id`` plus parallel ``pos``/``pos_id`` and ``neg``/``neg_id`` lists, so
translations rejoin losslessly:

    <query-column> (scalar) <- translated query,   keyed by query_id
    <pos-column>   (list)   <- translated positives, keyed elementwise by pos_id
    <neg-column>   (list)   <- translated negatives, keyed elementwise by neg_id

An id with no translation (e.g. a record the translate stage flagged) becomes
``None`` in its column; the counts are reported at the end.

Run:
    uv run python helpers/jsonl_to_HF_mined_ds.py
    # smoke test on the first few rows
    uv run python helpers/jsonl_to_HF_mined_ds.py --limit 5 --out /tmp/smoke.jsonl
"""

import argparse
from pathlib import Path

import orjson
from datasets import load_dataset

# load_input resolves a directory (-> ordered step-*.jsonl shards), a glob, or a
# single file into a list of records — the same convention the runner uses.
from combine_experiments import load_input

OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def build_map(arg: str, field: str) -> dict:
    """Map each processed record's ``id`` to its ``field`` (its translation).

    ``r.get(field)`` (not ``r[field]``) tolerates flagged/errored records that
    never got a translation — their id maps to ``None`` rather than raising.
    """
    return {r["id"]: r.get(field) for r in load_input(arg)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="mining-negatives/squadv2-mined-negatives",
        help="Original HuggingFace dataset repo id "
        "(default: mining-negatives/squadv2-mined-negatives).",
    )
    parser.add_argument("--split", default="train", help="Dataset split (default: train).")
    parser.add_argument(
        "--queries",
        default="dataset/translate_squadv2_queries_pl/00_translate",
        help="Translated queries: a stage dir, glob, or single JSONL file.",
    )
    parser.add_argument(
        "--documents",
        default="dataset/translate_squadv2_documents_pl/00_translate",
        help="Translated documents: a stage dir, glob, or single JSONL file.",
    )
    parser.add_argument(
        "--field",
        default="text_gtranslate",
        help="Field read from each processed record as the translation "
        "(default: text_gtranslate).",
    )
    parser.add_argument(
        "--query-column",
        default="query_pl",
        help="Name of the new translated-query column (default: query_pl).",
    )
    parser.add_argument(
        "--pos-column",
        default="pos_pl",
        help="Name of the new translated-positives column (default: pos_pl).",
    )
    parser.add_argument(
        "--neg-column",
        default="neg_pl",
        help="Name of the new translated-negatives column (default: neg_pl).",
    )
    parser.add_argument(
        "--out",
        help="Output JSONL path. Default: data/<dataset-basename>_translated.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N rows (handy for smoke runs).",
    )
    args = parser.parse_args()

    qmap = build_map(args.queries, args.field)
    dmap = build_map(args.documents, args.field)

    dataset = load_dataset(args.dataset, split=args.split)

    if args.out:
        out_path = Path(args.out)
    else:
        basename = args.dataset.split("/")[-1].replace("-", "_")
        out_path = OUT_DIR / f"{basename}_translated.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    missing_query = 0
    missing_pos = 0
    missing_neg = 0

    def translate_docs(ids: list, missing_counter: str) -> list:
        nonlocal missing_pos, missing_neg
        out = []
        for doc_id in ids or []:
            text = dmap.get(doc_id)
            if text is None:
                if missing_counter == "pos":
                    missing_pos += 1
                else:
                    missing_neg += 1
            out.append(text)
        return out

    with out_path.open("wb") as f:
        for row in dataset:
            if args.limit is not None and n_rows >= args.limit:
                break

            # dict(row) keeps the original column order; new columns are appended.
            record = dict(row)
            query_text = qmap.get(row["query_id"])
            if query_text is None:
                missing_query += 1
            record[args.query_column] = query_text
            record[args.pos_column] = translate_docs(row["pos_id"], "pos")
            record[args.neg_column] = translate_docs(row["neg_id"], "neg")

            f.write(orjson.dumps(record))
            f.write(b"\n")
            n_rows += 1

    print(f"Wrote {n_rows} rows to {out_path}")
    print(f"  missing query translations:     {missing_query}")
    print(f"  missing positive translations:  {missing_pos}")
    print(f"  missing negative translations:  {missing_neg}")


if __name__ == "__main__":
    main()
