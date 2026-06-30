"""Convert a mined-negatives dataset to JSONL, flagging role and processed state.

This is an enriched variant of ``HF_mined_ds_to_jsonl.py``. It produces the same
two ``{id, source, text}`` files the pipeline consumes — one row per query, one
row per unique document — but adds two kinds of flags:

    data/<prefix>_queries.jsonl    -> {id, source, text[, processed]}
    data/<prefix>_documents.jsonl  -> {id, source, text, is_positive, is_negative[, processed]}

* ``is_positive`` / ``is_negative``: a document's mined role. The same document
  id can be a positive for one query and a negative for another, so both may be
  true on a single (de-duplicated) row — no information is lost.

* ``processed``: optional. Point ``--documents-processed-dir`` /
  ``--queries-processed-dir`` at an experiment stage output (a directory of
  ``step-*.jsonl`` shards, a glob, or a single file) and each output row is
  flagged ``true`` when its id appears there, ``false`` otherwise. Presence is
  what counts — a record flagged ``error`` still counts as processed. The two
  dirs are independent and each optional; when one is omitted, that output gets
  no ``processed`` field at all. They are kept separate because query and
  document ids both restart at 0 and would collide in a shared set.
"""

import argparse
from pathlib import Path

import orjson
from datasets import load_dataset

# load_input resolves a directory (-> ordered step-*.jsonl shards), a glob, or a
# single file into a list of records — the same convention the runner uses.
from combine_experiments import load_input

OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def load_processed_ids(arg: str | None) -> set | None:
    """Return the set of ids present in a stage output, or None if ``arg`` is unset."""
    if arg is None:
        return None
    return {rec["id"] for rec in load_input(arg)}


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
    parser.add_argument(
        "--documents-processed-dir",
        help="Stage dir/glob/file whose ids mark documents as already processed.",
    )
    parser.add_argument(
        "--queries-processed-dir",
        help="Stage dir/glob/file whose ids mark queries as already processed.",
    )
    args = parser.parse_args()

    docs_processed = load_processed_ids(args.documents_processed_dir)
    queries_processed = load_processed_ids(args.queries_processed_dir)

    dataset = load_dataset(args.dataset, split=args.split)

    out_prefix = args.out_prefix or args.dataset.split("/")[-1].replace("-", "_")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queries_path = OUT_DIR / f"{out_prefix}_queries.jsonl"
    documents_path = OUT_DIR / f"{out_prefix}_documents.jsonl"

    # Documents are accumulated (not streamed) because a doc's role can be
    # completed by a later row: positive for one query, negative for another.
    documents: dict[object, dict] = {}
    seen_queries: set = set()

    def query_record(query_id, text: str) -> dict:
        rec = {"id": query_id, "source": args.source, "text": text}
        if queries_processed is not None:
            rec["processed"] = query_id in queries_processed
        return rec

    with queries_path.open("wb") as qf:
        for row in dataset:
            query_id = row["query_id"]
            if query_id not in seen_queries:
                seen_queries.add(query_id)
                qf.write(orjson.dumps(query_record(query_id, row["query"])))
                qf.write(b"\n")

            for ids, texts, role in (
                (row["pos_id"], row["pos"], "is_positive"),
                (row["neg_id"], row["neg"], "is_negative"),
            ):
                for doc_id, text in zip(ids or [], texts or []):
                    doc = documents.get(doc_id)
                    if doc is None:
                        doc = {"text": text, "is_positive": False, "is_negative": False}
                        documents[doc_id] = doc
                    doc[role] = True

    n_pos = n_neg = n_both = n_doc_processed = 0
    with documents_path.open("wb") as df:
        for doc_id, doc in documents.items():
            rec = {
                "id": doc_id,
                "source": args.source,
                "text": doc["text"],
                "is_positive": doc["is_positive"],
                "is_negative": doc["is_negative"],
            }
            if docs_processed is not None:
                rec["processed"] = doc_id in docs_processed
                n_doc_processed += rec["processed"]
            df.write(orjson.dumps(rec))
            df.write(b"\n")
            n_pos += doc["is_positive"]
            n_neg += doc["is_negative"]
            n_both += doc["is_positive"] and doc["is_negative"]

    n_query_processed = (
        sum(qid in queries_processed for qid in seen_queries)
        if queries_processed is not None
        else None
    )

    print(f"Wrote {len(seen_queries)} rows to {queries_path}")
    if n_query_processed is not None:
        print(f"  queries processed: {n_query_processed}/{len(seen_queries)}")
    print(f"Wrote {len(documents)} rows to {documents_path}")
    print(f"  documents positive/negative/both: {n_pos}/{n_neg}/{n_both}")
    if docs_processed is not None:
        print(f"  documents processed: {n_doc_processed}/{len(documents)}")


if __name__ == "__main__":
    main()
