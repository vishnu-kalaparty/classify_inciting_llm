"""
Compute and save classification metrics for INCITE binary classifiers.

Provides:
1. Per-binary-classifier metrics (4 gold rows x 2 pred columns: <Category> / Not)
2. Holistic metrics: each positive label F1 from its binary classifier + None F1 from consensus + macro F1

Run standalone on a combined output JSONL:
  python calculate_metrics.py path/to/inciting_combined.jsonl
"""
import argparse
import json
import sys
from typing import Any, Dict, List

# Labels when run as standalone (avoids importing constants / API key)
LABELS_ORDER = ["Identity", "Imputed Misdeeds", "Exhortation", "None"]
BINARY_CATEGORIES = ["Identity", "Imputed Misdeeds", "Exhortation"]


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_binary_metrics(
    results: List[Dict[str, Any]],
    category: str,
    output_path: str,
) -> Dict[str, Any]:
    """
    Compute and print metrics for one binary classifier.
    Table: 4 original gold labels as rows x 2 pred columns (<category> / Not).
    """
    not_label = f"Not {category}"
    gold_labels_original = LABELS_ORDER
    pred_labels_binary = [category, not_label]

    conf = {g: {p: 0 for p in pred_labels_binary} for g in gold_labels_original}
    skipped = 0
    for r in results:
        gold = r.get("gold_label", "")
        pred = r.get("pred_label", "")
        if gold not in gold_labels_original:
            skipped += 1
            continue
        if pred == category:
            conf[gold][category] += 1
        elif pred == not_label:
            conf[gold][not_label] += 1
        else:
            skipped += 1

    total = sum(conf[g][p] for g in gold_labels_original for p in pred_labels_binary)

    binary_gold_positive = category
    tp = conf[binary_gold_positive][category]
    fp = sum(conf[g][category] for g in gold_labels_original if g != binary_gold_positive)
    fn = conf[binary_gold_positive][not_label]
    prec, rec, f1 = _precision_recall_f1(tp, fp, fn)

    tp_not = sum(conf[g][not_label] for g in gold_labels_original if g != binary_gold_positive)
    fp_not = conf[binary_gold_positive][not_label]
    fn_not = sum(conf[g][category] for g in gold_labels_original if g != binary_gold_positive)
    prec_not, rec_not, f1_not = _precision_recall_f1(tp_not, fp_not, fn_not)

    accuracy = (tp + tp_not) / total if total > 0 else 0.0
    macro_f1 = (f1 + f1_not) / 2

    print(f"\n{'='*70}")
    print(f"Binary Classifier Metrics: {category} vs {not_label}")
    print(f"{'='*70}")
    print(f"Total: {total}  Skipped (Unknown): {skipped}  Accuracy: {accuracy:.4f}")

    cat_short = category[:11]
    not_short = not_label[:15]
    header = f"{'Gold Label':<22} {cat_short:>12} {not_short:>16}   {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}"
    print(f"\n{header}")
    print("-" * len(header))
    for g in gold_labels_original:
        c_count = conf[g][category]
        n_count = conf[g][not_label]
        support = c_count + n_count
        if g == binary_gold_positive:
            row_prec, row_rec, row_f1 = prec, rec, f1
        else:
            row_prec = row_rec = row_f1 = None
        if row_prec is not None:
            row = f"{g:<22} {c_count:>12} {n_count:>16}   {row_prec:>8.4f} {row_rec:>8.4f} {row_f1:>8.4f} {support:>8}"
        else:
            row = f"{g:<22} {c_count:>12} {n_count:>16}   {'':>8} {'':>8} {'':>8} {support:>8}"
        print(row)

    print(f"\n{category} — Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"{not_label} — Precision: {prec_not:.4f}  Recall: {rec_not:.4f}  F1: {f1_not:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    metrics = {
        "category": category,
        "total": total,
        "skipped": skipped,
        "accuracy": accuracy,
        "positive": {"precision": prec, "recall": rec, "f1": f1},
        f"not_{category.lower().replace(' ', '_')}": {"precision": prec_not, "recall": rec_not, "f1": f1_not},
        "macro_f1": macro_f1,
        "confusion_matrix": conf,
    }
    metrics_path = output_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {metrics_path}")
    return metrics


def compute_holistic_metrics(
    all_results: Dict[str, List[Dict[str, Any]]],
    labels: List[str],
    output_path: str,
) -> Dict[str, Any]:
    """
    Compute holistic 4-class metrics:
    - Identity/Imputed Misdeeds/Exhortation F1 from their respective binary classifiers
    - None F1 from consensus (all three say "Not")
    - Macro F1 across all 4 labels
    """
    first_cat = BINARY_CATEGORIES[0]
    n_rows = len(all_results[first_cat])

    per_class: Dict[str, Dict[str, int]] = {}
    for lbl in labels:
        per_class[lbl] = {"tp": 0, "fp": 0, "fn": 0}

    for i in range(n_rows):
        gold = all_results[first_cat][i]["gold_label"]
        if gold not in labels:
            continue

        for cat in BINARY_CATEGORIES:
            not_label = f"Not {cat}"
            pred = all_results[cat][i]["pred_label"]
            if pred == cat:
                if gold == cat:
                    per_class[cat]["tp"] += 1
                else:
                    per_class[cat]["fp"] += 1
            elif pred == not_label:
                if gold == cat:
                    per_class[cat]["fn"] += 1

        all_not = all(
            all_results[cat][i]["pred_label"] == f"Not {cat}" for cat in BINARY_CATEGORIES
        )
        if all_not:
            if gold == "None":
                per_class["None"]["tp"] += 1
            else:
                per_class["None"]["fp"] += 1
        else:
            if gold == "None":
                per_class["None"]["fn"] += 1

    print(f"\n{'='*70}")
    print("Holistic Metrics (Binary Classifiers -> 4-Class)")
    print(f"{'='*70}")

    header = f"{'Label':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    class_metrics = {}
    for lbl in labels:
        tp = per_class[lbl]["tp"]
        fp = per_class[lbl]["fp"]
        fn = per_class[lbl]["fn"]
        prec, rec, f1 = _precision_recall_f1(tp, fp, fn)
        support = tp + fn
        class_metrics[lbl] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        print(f"{lbl:<22} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10}")

    macro_prec = sum(class_metrics[l]["precision"] for l in labels) / len(labels)
    macro_rec = sum(class_metrics[l]["recall"] for l in labels) / len(labels)
    macro_f1 = sum(class_metrics[l]["f1"] for l in labels) / len(labels)
    total_support = sum(class_metrics[l]["support"] for l in labels)
    if total_support > 0:
        weighted_f1 = sum(
            class_metrics[l]["f1"] * class_metrics[l]["support"] for l in labels
        ) / total_support
    else:
        weighted_f1 = 0.0

    print(f"\n{'Macro Avg':<22} {macro_prec:>10.4f} {macro_rec:>10.4f} {macro_f1:>10.4f} {total_support:>10}")
    print(f"{'Weighted Avg':<22} {'':>10} {'':>10} {weighted_f1:>10.4f} {total_support:>10}")

    metrics = {
        "per_class": {
            lbl: {k: v for k, v in class_metrics[lbl].items()}
            for lbl in labels
        },
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_support": total_support,
    }

    metrics_path = output_path.replace(".jsonl", "_holistic_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved holistic metrics to {metrics_path}")
    return metrics


def _read_multi_line_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL with multi-line indented JSON records (brace-counting)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    records = []
    in_string = False
    escape = False
    depth = 0
    start = 0
    for j, c in enumerate(content):
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if not in_string:
            if c == "{":
                if depth == 0:
                    start = j
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    records.append(json.loads(content[start : j + 1]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute metrics from INCITE binary classification output files.\n"
            "Pass individual binary JSONL files OR the combined JSONL."
        )
    )
    parser.add_argument(
        "jsonl_paths",
        nargs="+",
        help=(
            "Paths to .jsonl files. For individual binary metrics pass each "
            "binary output. For holistic metrics pass all three binary outputs."
        ),
    )
    args = parser.parse_args()

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for path in args.jsonl_paths:
        records = _read_multi_line_jsonl(path)
        if not records:
            print(f"No records found in {path}", file=sys.stderr)
            continue

        category = records[0].get("category")
        if category and category in BINARY_CATEGORIES:
            valid = [r for r in records if r.get("gold_label") in LABELS_ORDER]
            print(f"\nLoaded {len(records)} records from {path}; {len(valid)} valid for {category}")
            compute_binary_metrics(valid, category, path)
            all_results[category] = valid
        else:
            print(f"Skipping {path}: no recognized category field in records.", file=sys.stderr)

    if len(all_results) == len(BINARY_CATEGORIES):
        lengths = [len(v) for v in all_results.values()]
        if len(set(lengths)) == 1:
            combined_path = args.jsonl_paths[0].replace(
                BINARY_CATEGORIES[0].lower().replace(" ", "_"),
                "combined",
            )
            compute_holistic_metrics(all_results, LABELS_ORDER, combined_path)
        else:
            print(
                "Warning: binary result lists have different lengths; skipping holistic metrics.",
                file=sys.stderr,
            )
    elif len(all_results) > 0:
        print(
            f"\nOnly {len(all_results)}/{len(BINARY_CATEGORIES)} binary classifiers found. "
            "Pass all 3 binary output files for holistic metrics.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
