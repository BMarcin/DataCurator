"""Convert the lightonai/embeddings-fine-tuning dataset to JSONL.

The dataset ships two configs, ``documents`` and ``queries``, each split by
source corpus (fiqa, nq, hotpotqa, msmarco, fever, squadv2, trivia). The
``document_id`` / ``query_id`` values restart at 0 within every split, so we
keep the original id and also store the ``source`` split to disambiguate
collisions across splits.

By default every source corpus is merged into one file per config:
    data/embeddings_finetuning_documents.jsonl
    data/embeddings_finetuning_queries.jsonl

Pass ``--source <corpus>`` to export only one corpus; its name is then part of
the output filename, e.g. ``--source fiqa`` writes:
    data/embeddings_finetuning_documents_fiqa.jsonl
    data/embeddings_finetuning_queries_fiqa.jsonl
"""

import argparse
from pathlib import Path

import orjson
from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def dump(dataset_dict, id_field: str, text_field: str, out_path: Path, source: str | None) -> None:
    if source is not None:
        if source not in dataset_dict:
            raise SystemExit(
                f"Unknown source '{source}'. Available: {', '.join(dataset_dict.keys())}"
            )
        items = [(source, dataset_dict[source])]
    else:
        items = list(dataset_dict.items())

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


def out_name(kind: str, source: str | None) -> Path:
    suffix = f"_{source}" if source else ""
    return OUT_DIR / f"embeddings_finetuning_{kind}{suffix}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        help="Export only this source corpus (e.g. fiqa). Default: all corpora merged.",
    )
    args = parser.parse_args()

    ds_documents = load_dataset("lightonai/embeddings-fine-tuning", "documents")
    ds_queries = load_dataset("lightonai/embeddings-fine-tuning", "queries")

    dump(ds_documents, "document_id", "document", out_name("documents", args.source), args.source)
    dump(ds_queries, "query_id", "query", out_name("queries", args.source), args.source)


if __name__ == "__main__":
    main()
