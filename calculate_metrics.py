"""
Compute and save classification metrics for INCITE (confusion matrix, per-class, target table).

Run standalone on a classification output JSONL:
  python calculate_metrics.py path/to/inciting_gpt_5_mini_few_shot.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List

# Labels when run as standalone (avoids importing constants / API key)
LABELS_ORDER = ["Identity", "Imputed Misdeeds", "Exhortation", "None"]
VALID_LABELS = {"Identity", "Imputed Misdeeds", "Exhortation", "None"}


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _compute_metrics_from_confusion(
    conf: Dict[str, Dict[str, int]], labels: List[str]
) -> Dict[str, Any]:
    """Compute overall metrics from confusion matrix (like HateXplain calculate_metrics)."""
    total = sum(conf[g][p] for g in labels for p in labels)
    correct = sum(conf[g][g] for g in labels)
    metrics: Dict[str, Any] = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
    }
    per_class = {}
    for lbl in labels:
        tp = conf[lbl][lbl]
        fp = sum(conf[g][lbl] for g in labels if g != lbl)
        fn = sum(conf[lbl][p] for p in labels if p != lbl)
        prec, rec, f1 = _precision_recall_f1(tp, fp, fn)
        per_class[lbl] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": tp + fn,
        }
    metrics["per_class"] = per_class
    n = len(labels)
    metrics["macro_precision"] = sum(per_class[l]["precision"] for l in labels) / n
    metrics["macro_recall"] = sum(per_class[l]["recall"] for l in labels) / n
    metrics["macro_f1"] = sum(per_class[l]["f1"] for l in labels) / n
    total_support = sum(per_class[l]["support"] for l in labels)
    if total_support > 0:
        metrics["weighted_precision"] = sum(per_class[l]["precision"] * per_class[l]["support"] for l in labels) / total_support
        metrics["weighted_recall"] = sum(per_class[l]["recall"] * per_class[l]["support"] for l in labels) / total_support
        metrics["weighted_f1"] = sum(per_class[l]["f1"] * per_class[l]["support"] for l in labels) / total_support
    else:
        metrics["weighted_precision"] = metrics["weighted_recall"] = metrics["weighted_f1"] = 0.0
    return metrics


def _compute_target_metrics(
    gold_labels: List[str], pred_labels: List[str], labels: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute metrics per label (treating each label as a "target" like HateXplain's target table).
    Returns dict: label -> {count, accuracy, macro_f1, weighted_f1, per_class, confusion_matrix}.
    """
    target_pairs: Dict[str, List[tuple[str, str]]] = defaultdict(list)
    for g, p in zip(gold_labels, pred_labels):
        if g in labels and p in labels:
            target_pairs[g].append((g, p))

    target_metrics = {}
    for label in labels:
        pairs = target_pairs.get(label, [])
        if not pairs:
            target_metrics[label] = {
                "count": 0,
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "per_class": {lbl: {"support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0} for lbl in labels},
                "confusion_matrix": {g: {p: 0 for p in labels} for g in labels},
            }
            continue
        g_list = [x[0] for x in pairs]
        p_list = [x[1] for x in pairs]
        conf = {g: {p: 0 for p in labels} for g in labels}
        for gi, pi in zip(g_list, p_list):
            conf[gi][pi] += 1
        total = len(pairs)
        correct = sum(conf[g][g] for g in labels)
        per_class = {}
        for lbl in labels:
            tp = conf[lbl][lbl]
            fp = sum(conf[g][lbl] for g in labels if g != lbl)
            fn = sum(conf[lbl][p] for p in labels if p != lbl)
            prec, rec, f1 = _precision_recall_f1(tp, fp, fn)
            per_class[lbl] = {"support": tp + fn, "precision": prec, "recall": rec, "f1": f1}
        total_support = sum(per_class[l]["support"] for l in labels)
        macro_f1 = sum(per_class[l]["f1"] for l in labels) / len(labels)
        weighted_f1 = (
            sum(per_class[l]["f1"] * per_class[l]["support"] for l in labels) / total_support
            if total_support > 0 else 0.0
        )
        target_metrics[label] = {
            "count": total,
            "accuracy": correct / total if total > 0 else 0.0,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class": per_class,
            "confusion_matrix": conf,
        }
    return target_metrics


def _print_target_table(
    target_metrics: Dict[str, Dict[str, Any]],
    labels: List[str],
    overall_per_class: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    """Print target table: pred counts + per-class metrics. Use overall_per_class so values match global Per-Class Metrics."""
    print("\n--- Per-Class Metrics (target table) ---")
    short = ["Identity", "Imputed_Mis", "Exhortation", "None"]
    header = f"{'Target':<22} " + " ".join(f"{s:>10}" for s in short)
    header += f" {'Prec':>5} {'Recall':>8} {'F1':>5} {'Support':>14}"
    print(header)
    print("-" * len(header))
    for target in labels:
        tmetrics = target_metrics.get(target, {})
        conf = tmetrics.get("confusion_matrix", {})
        pred_counts = [conf.get(target, {}).get(p, 0) for p in labels]
        if overall_per_class and target in overall_per_class:
            pc = overall_per_class[target]
            prec = pc.get("precision", 0.0)
            rec = pc.get("recall", 0.0)
            f1 = pc.get("f1", 0.0)
            support = pc.get("support", 0)
        else:
            pc = tmetrics.get("per_class", {}).get(target, {})
            prec = pc.get("precision", 0.0)
            rec = pc.get("recall", 0.0)
            f1 = pc.get("f1", 0.0)
            support = pc.get("support", 0)
        row = f"{target:<22} " + " ".join(f"{c:>10}" for c in pred_counts)
        row += f" {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {support:>8}"
        print(row)


def compute_and_save_metrics(
    gold_labels: List[str],
    pred_labels: List[str],
    labels: List[str],
    output_path: str,
) -> Dict[str, Any]:
    """
    Compute metrics from gold/pred labels, print summary and target table, save to output_path (jsonl -> _metrics.json).
    Returns the full metrics dict.
    """
    target_metrics = _compute_target_metrics(gold_labels, pred_labels, labels)
    conf = {g: {p: 0 for p in labels} for g in labels}
    for g in labels:
        for p in labels:
            conf[g][p] = target_metrics[g]["confusion_matrix"].get(g, {}).get(p, 0)

    overall = _compute_metrics_from_confusion(conf, labels)
    total = len(gold_labels)
    correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)

    print(f"\nEvaluation on {total} examples:")
    print(f"Accuracy: {correct}/{total} = {overall['accuracy']:.4f}")
    print("\n--- Overall Metrics ---")
    print(f"Total: {overall['total']}  Correct: {overall['correct']}  Accuracy: {overall['accuracy']:.4f}")
    print("\n--- Macro Averages ---")
    print(f"Macro Precision: {overall['macro_precision']:.4f}  Macro Recall: {overall['macro_recall']:.4f}  Macro F1: {overall['macro_f1']:.4f}")
    print("\n--- Weighted Averages ---")
    print(f"Weighted Precision: {overall['weighted_precision']:.4f}  Weighted Recall: {overall['weighted_recall']:.4f}  Weighted F1: {overall['weighted_f1']:.4f}")

    _print_target_table(target_metrics, labels, overall["per_class"])

    metrics_path = output_path.replace(".jsonl", "_metrics.json")
    metrics = {
        "accuracy": overall["accuracy"],
        "correct": overall["correct"],
        "total": overall["total"],
        "macro_precision": overall["macro_precision"],
        "macro_recall": overall["macro_recall"],
        "macro_f1": overall["macro_f1"],
        "weighted_precision": overall["weighted_precision"],
        "weighted_recall": overall["weighted_recall"],
        "weighted_f1": overall["weighted_f1"],
        "per_class": overall["per_class"],
        "confusion_matrix": conf,
        "labels": labels,
        "target_metrics": target_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics to {metrics_path}")
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
        description="Compute metrics from an INCITE classification output JSONL (gold_label, pred_label)."
    )
    parser.add_argument(
        "jsonl_path",
        help="Path to the .jsonl file (e.g. inciting_gpt_5_mini_few_shot.jsonl)",
    )
    args = parser.parse_args()
    path = args.jsonl_path
    if not path.endswith(".jsonl"):
        print("Warning: path should usually be a .jsonl file; metrics will be written to <path>_metrics.json", file=sys.stderr)
    records = _read_multi_line_jsonl(path)
    gold_labels = []
    pred_labels = []
    for r in records:
        g = (r.get("gold_label") or "").strip()
        p = (r.get("pred_label") or "").strip()
        if g in VALID_LABELS and p in VALID_LABELS:
            gold_labels.append(g)
            pred_labels.append(p)
    if not gold_labels:
        print("No valid (gold_label, pred_label) pairs found in the JSONL. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(records)} records; {len(gold_labels)} valid label pairs for metrics.")
    compute_and_save_metrics(gold_labels, pred_labels, LABELS_ORDER, path)


if __name__ == "__main__":
    main()
