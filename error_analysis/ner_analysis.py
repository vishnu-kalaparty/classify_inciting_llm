"""
Stanford NER analysis on key phrases from misclassified samples (Inciting dataset).
Uses Stanza (Stanford NLP) to extract named entities from key phrases
in binary combined outputs for False Negatives and False Positives.

FP = Gold: None, Pred: Inciting (Identity / Imputed Misdeeds / Exhortation)
FN = Gold: Inciting, Pred: None
"""

import argparse
import json
import csv
from pathlib import Path
from collections import Counter

import stanza

_DEFAULT_COMBINED = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "zero_shot"
    / "inciting_combined.jsonl"
)
_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "ner"

INCITING_LABELS = {"Identity", "Imputed Misdeeds", "Exhortation"}


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file that may contain pretty-printed (multi-line) JSON objects."""
    records = []
    with open(path, "r") as f:
        content = f.read()
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        while idx < len(content) and content[idx] in " \t\n\r":
            idx += 1
        if idx >= len(content):
            break
        obj, end_idx = decoder.raw_decode(content, idx)
        records.append(obj)
        idx = end_idx
    return records


def derive_pred_label(rec: dict) -> str:
    """Derive a single predicted label from binary-style flags.

    Priority order: Identity > Imputed Misdeeds > Exhortation.
    If none of the three are positive, return \"None\".
    """
    if rec.get("pred_identity") == "Identity":
        return "Identity"
    if rec.get("pred_imputed_misdeeds") == "Imputed Misdeeds":
        return "Imputed Misdeeds"
    if rec.get("pred_exhortation") == "Exhortation":
        return "Exhortation"
    return "None"


def run_ner(texts: list[str], nlp) -> list[dict]:
    """Run Stanford NER on a list of texts."""
    entities = []
    for text in texts:
        if not text.strip():
            continue
        doc = nlp(text)
        for sent in doc.sentences:
            for ent in sent.ents:
                entities.append({
                    "text": ent.text,
                    "type": ent.type,
                    "source_phrase": text,
                })
    return entities


def summarise_entities(entities: list[dict]) -> Counter:
    return Counter((e["text"], e["type"]) for e in entities)


def save_entity_results(entities: list[dict], summary: Counter, tag: str, output_dir: Path):
    detail_path = output_dir / f"ner_{tag}_entities.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "type", "source_phrase"])
        writer.writeheader()
        writer.writerows(entities)
    print(f"  Entity details saved: {detail_path}")

    summary_path = output_dir / f"ner_{tag}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entity", "type", "count"])
        for (ent_text, ent_type), count in summary.most_common():
            writer.writerow([ent_text, ent_type, count])
    print(f"  Entity summary saved: {summary_path}")

    type_counts = Counter(e["type"] for e in entities)
    type_path = output_dir / f"ner_{tag}_type_distribution.csv"
    with open(type_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_type", "count"])
        for etype, count in type_counts.most_common():
            writer.writerow([etype, count])
    print(f"  Type distribution saved: {type_path}")


def main():
    parser = argparse.ArgumentParser(description="NER on FN/FP key phrases from inciting_combined.jsonl")
    parser.add_argument(
        "--combined-jsonl",
        type=Path,
        default=_DEFAULT_COMBINED,
        help="Path to inciting_combined.jsonl (binary combined format)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="default",
        help="Subfolder under error_analysis/outputs/ner/ for CSV outputs (e.g. zero_shot)",
    )
    args = parser.parse_args()
    jsonl_path = args.combined_jsonl.resolve()
    output_dir = _OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading/loading Stanza English NER model ...")
    stanza.download("en", processors="tokenize,ner", verbose=False)
    nlp = stanza.Pipeline("en", processors="tokenize,ner", verbose=False)

    print(f"Loading JSONL: {jsonl_path}")
    records = load_jsonl(jsonl_path)
    print(f"  Total records: {len(records)}")

    fn_phrases = []
    fp_phrases = []

    for rec in records:
        gold = rec.get("gold_label", "")
        pred = derive_pred_label(rec)
        phrases = rec.get("text", "")

        if not phrases or not phrases.strip():
            continue

        if gold in INCITING_LABELS and pred == "None":
            fn_phrases.append(phrases)
        elif gold == "None" and pred in INCITING_LABELS:
            fp_phrases.append(phrases)

    print(f"  False Negatives (Inciting->None): {len(fn_phrases)} key-word texts")
    print(f"  False Positives (None->Inciting): {len(fp_phrases)} key-word texts")

    print("\n--- False Negatives: Stanford NER on Key Words ---")
    fn_entities = run_ner(fn_phrases, nlp)
    fn_summary = summarise_entities(fn_entities)
    print(f"  Found {len(fn_entities)} entity mentions ({len(fn_summary)} unique)")
    save_entity_results(fn_entities, fn_summary, "fn_inciting_pred_none", output_dir)

    print("\n--- False Positives: Stanford NER on Key Words ---")
    fp_entities = run_ner(fp_phrases, nlp)
    fp_summary = summarise_entities(fp_entities)
    print(f"  Found {len(fp_entities)} entity mentions ({len(fp_summary)} unique)")
    save_entity_results(fp_entities, fp_summary, "fp_none_pred_inciting", output_dir)

    print("\n--- Top Entities ---")
    for label, summary in [("FN", fn_summary), ("FP", fp_summary)]:
        print(f"\n  {label} -- Top 20 entities:")
        for (ent_text, ent_type), count in summary.most_common(20):
            print(f"    {ent_text:30s} [{ent_type:10s}]  x{count}")

    print("\nDone.")


if __name__ == "__main__":
    main()
