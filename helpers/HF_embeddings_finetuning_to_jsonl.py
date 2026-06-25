"""Convert the lightonai/embeddings-fine-tuning dataset to JSONL.

The dataset ships multiple configs ("subsets"); this script supports the two
text subsets, ``documents`` and ``queries``. Each is split by source corpus
(fiqa, nq, hotpotqa, msmarco, fever, squadv2, trivia). The ``document_id`` /
``query_id`` values restart at 0 within every split, so we keep the original id
and also store the ``source`` split to disambiguate collisions across splits.

Pick what to download with ``--subset`` and ``--split``:

    # one subset, one corpus -> data/embeddings_finetuning_queries_fiqa.jsonl
    python HF_embeddings_finetuning_to_jsonl.py --subset queries --split fiqa

    # repeat --subset for several; omit --split to merge all corpora
    python HF_embeddings_finetuning_to_jsonl.py --subset documents --subset queries

By default (no flags) every subset and every corpus is exported, one merged
file per subset:
    data/embeddings_finetuning_documents.jsonl
    data/embeddings_finetuning_queries.jsonl

When ``--split`` is given, the corpus name becomes part of the output filename,
e.g. ``--subset documents --split fiqa`` writes:
    data/embeddings_finetuning_documents_fiqa.jsonl
"""

import argparse
from pathlib import Path

import orjson
from datasets import load_dataset

REPO = "lightonai/embeddings-fine-tuning"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"

# subset (HF config) -> the columns to read for the record id and text.
# Add a new subset here to support it (the ``scores`` config has a different
# shape and is intentionally not included).
SUBSETS = {
    "documents": {"id_field": "document_id", "text_field": "document"},
    "queries": {"id_field": "query_id", "text_field": "query"},
}


def load_splits(subset: str, split: str | None):
    """Return a list of ``(split_name, dataset)`` pairs for the subset.

    When ``split`` is given only that corpus is downloaded; otherwise the whole
    config (all corpora) is loaded and returned split by split.
    """
    if split is not None:
        try:
            dataset = load_dataset(REPO, subset, split=split)
        except ValueError as exc:
            raise SystemExit(f"Unknown split '{split}' for subset '{subset}'. ({exc})")
        return [(split, dataset)]
    return list(load_dataset(REPO, subset).items())


def dump(items, id_field: str, text_field: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_path.open("wb") as f:
        for split_name, split in items:
            for row in split:
                record = {
                    "id": row[id_field],
                    "source": split_name,
                    "text": row[text_field],
                }
                f.write(orjson.dumps(record))
                f.write(b"\n")
                rows += 1
    print(f"Wrote {rows} rows to {out_path}")


def out_name(subset: str, split: str | None) -> Path:
    suffix = f"_{split}" if split else ""
    return OUT_DIR / f"embeddings_finetuning_{subset}{suffix}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        choices=list(SUBSETS),
        action="append",
        help="Subset (HF config) to export; repeatable. Default: all subsets.",
    )
    parser.add_argument(
        "--split",
        "--source",
        dest="split",
        help="Export only this source corpus (e.g. fiqa). Default: all corpora merged.",
    )
    args = parser.parse_args()

    subsets = args.subset or list(SUBSETS)
    for subset in subsets:
        fields = SUBSETS[subset]
        items = load_splits(subset, args.split)
        dump(items, fields["id_field"], fields["text_field"], out_name(subset, args.split))


if __name__ == "__main__":
    main()
