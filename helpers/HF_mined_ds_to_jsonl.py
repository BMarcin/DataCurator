"""Convert the jansowa/trivia-mined-negatives dataset to JSONL.

Each row holds one ``query`` (with a ``query_id``) plus mined ``pos`` / ``neg``
texts and their parallel ``pos_id`` / ``neg_id`` lists. We split this into two
files in the same ``{id, source, text}`` shape the pipeline consumes:

    data/trivia_mined_negatives_queries.jsonl    -> one row per query
    data/trivia_mined_negatives_documents.jsonl  -> one row per unique document

The same document id can appear in many rows (as a positive for one query and a
negative for another), so documents are de-duplicated by id in a single pass —
each text is emitted (and later translated) only once.
"""

import argparse
from pathlib import Path

import orjson
from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="jansowa/trivia-mined-negatives",
        help="HuggingFace dataset repo id (default: jansowa/trivia-mined-negatives).",
    )
    parser.add_argument("--split", default="train", help="Dataset split (default: train).")
    parser.add_argument(
        "--source",
        default="trivia",
        help="Value written to the `source` field of every record (default: trivia).",
    )
    parser.add_argument(
        "--out-prefix",
        help="Output filename prefix. Default: derived from the dataset repo name.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)

    out_prefix = args.out_prefix or args.dataset.split("/")[-1].replace("-", "_")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queries_path = OUT_DIR / f"{out_prefix}_queries.jsonl"
    documents_path = OUT_DIR / f"{out_prefix}_documents.jsonl"

    seen_queries: set[str] = set()
    seen_documents: set[str] = set()
    n_queries = 0
    n_documents = 0

    def write(f, record_id: str, text: str) -> None:
        f.write(orjson.dumps({"id": record_id, "source": args.source, "text": text}))
        f.write(b"\n")

    with queries_path.open("wb") as qf, documents_path.open("wb") as df:
        for row in dataset:
            query_id = row["query_id"]
            if query_id not in seen_queries:
                seen_queries.add(query_id)
                write(qf, query_id, row["query"])
                n_queries += 1

            # Positives and negatives are both just documents; the same id may
            # recur across rows, so emit each only once.
            for ids, texts in ((row["pos_id"], row["pos"]), (row["neg_id"], row["neg"])):
                for doc_id, text in zip(ids or [], texts or []):
                    if doc_id not in seen_documents:
                        seen_documents.add(doc_id)
                        write(df, doc_id, text)
                        n_documents += 1

    print(f"Wrote {n_queries} rows to {queries_path}")
    print(f"Wrote {n_documents} rows to {documents_path}")


if __name__ == "__main__":
    main()
