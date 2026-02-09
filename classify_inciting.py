"""
Classify INCITE dataset (inciting speech) using GPT 5 mini.
Uses INCITE-Dataset.xlsx: columns sentence before, main sentence, sentence after, label.
Labels: Identity, Imputed Misdeeds, Exhortation, None.
"""
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import requests

from constants import (
    API_KEY,
    API_URL,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_NUM_THREADS,
    DEFAULT_OUTPUT,
    DEFAULT_SLEEP_SECONDS,
    ENABLE_FEW_SHOT,
    INCITE_XLSX,
    LABEL_MAP_STR,
    LABELS_ORDER,
    MAX_TOKENS,
    MODEL_NAME,
    RUN_ERRORS_FILE,
    START_INDEX,
    VALID_LABELS,
)
from data import load_few_shot_examples, load_incite_xlsx
from prompts import build_prompt


def call_gpt_completion(prompt: str) -> str:
    """Call API for GPT 5 mini. Returns choices[0].message.content."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": MAX_TOKENS,
        "seed": 42,
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        try:
            err = resp.json()
            msg = err.get("error", {}).get("message", str(err))
            typ = err.get("error", {}).get("type", "unknown")
            raise Exception(f"API Error {resp.status_code} ({typ}): {msg}")
        except (ValueError, KeyError):
            raise Exception(f"API Error {resp.status_code}: {resp.text[:500]}")
    return resp.json()["choices"][0]["message"]["content"]


def normalize_gold_label(label_val: str) -> str:
    if not label_val:
        return "Unknown"
    return LABEL_MAP_STR.get(label_val.strip().lower(), "Unknown")


def extract_model_classification(response_text: str) -> str:
    """
    Extract classification from model response.
    Expects "1. Classification: [Identity / Imputed Misdeeds / Exhortation / None]"
    """
    text = (response_text or "").strip()
    if "<|message|>" in text:
        idx = text.rfind("<|message|>")
        text = text[idx + len("<|message|>"):]
    c_idx = text.lower().find("classification:")
    if c_idx == -1:
        return "Unknown"
    start = c_idx + len("classification:")
    end = text.find("\n", start)
    label = (text[start:end] if end != -1 else text[start:]).strip()
    label_lower = label.lower()
    if "imputed misdeeds" in label_lower or "imputed misdeed" in label_lower:
        return "Imputed Misdeeds"
    if "identity" in label_lower:
        return "Identity"
    if "exhortation" in label_lower:
        return "Exhortation"
    if "none" in label_lower and "misdeeds" not in label_lower:
        return "None"
    return "Unknown"


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
    # Micro precision/recall/F1 equal accuracy in multi-class; omit to avoid redundancy
    return metrics


def _compute_target_metrics(
    gold_labels: List[str], pred_labels: List[str], labels: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute metrics per label (treating each label as a "target" like HateXplain's target table).
    Returns dict: label -> {count, accuracy, macro_f1, weighted_f1, per_class, confusion_matrix}.
    """
    # Group (gold, pred) pairs by gold label (each gold label = one "target" row)
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


def _error_record_to_example(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record.get("id"),
        "text": record.get("text", ""),
        "label": record.get("gold_label", ""),
    }


def retry_errors(
    errors_path: str,
    main_output_path: str,
    num_threads: int = 5,
    few_shot_examples: List[dict] | None = None,
) -> None:
    if not os.path.isfile(errors_path):
        print(f"Errors file not found: {errors_path}")
        return
    if not os.path.isfile(main_output_path):
        print(f"Main output not found: {main_output_path}")
        return
    records = _read_multi_line_jsonl(errors_path)
    n = len(records)
    print(f"Loaded {n} records from {errors_path}. Re-running API.", flush=True)
    examples = [_error_record_to_example(r) for r in records]
    indices = [r["index"] for r in records]
    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_i = {}
        for i, (ex, idx) in enumerate(zip(examples, indices)):
            future = executor.submit(_process_single, ex, idx, n, few_shot_examples)
            future_to_i[future] = i
        for future in as_completed(future_to_i):
            i = future_to_i[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append({
                    "index": indices[i],
                    "id": records[i].get("id"),
                    "text": records[i].get("text", ""),
                    "gold_label": records[i].get("gold_label", "Unknown"),
                    "pred_label": "Unknown",
                    "model_response": f"THREAD_ERROR: {e}",
                    "few_shot_enabled": few_shot_examples is not None,
                })
            completed += 1
            print(f"[{completed}/{n}] retry (index {indices[i]})", flush=True)
    results.sort(key=lambda r: r["index"])
    retry_by_index = {r["index"]: r for r in results}
    all_records = _read_multi_line_jsonl(main_output_path)
    by_index = {r["index"]: r for r in all_records}
    for idx, rec in retry_by_index.items():
        by_index[idx] = rec
    sorted_records = sorted(by_index.values(), key=lambda r: r["index"])
    with open(main_output_path, "w", encoding="utf-8") as fout:
        for r in sorted_records:
            fout.write(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
    still_unknown = [r for r in results if r.get("pred_label") == "Unknown"]
    with open(errors_path, "w", encoding="utf-8") as ferr:
        for r in still_unknown:
            ferr.write(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
    print(f"Merged {main_output_path}. {len(still_unknown)} still unknown -> {errors_path}")


def _process_single(
    example: Dict[str, Any],
    idx: int,
    total: int,
    few_shot_examples: List[dict] | None,
) -> Dict[str, Any]:
    text = example.get("text", "").strip()
    gold_label = normalize_gold_label(example.get("label", ""))
    prompt = build_prompt(text, few_shot_examples)
    try:
        response_text = call_gpt_completion(prompt)
    except Exception as e:
        response_text = f"ERROR: {e}"
    pred_label = extract_model_classification(response_text)
    return {
        "index": idx,
        "id": example.get("id"),
        "text": text,
        "gold_label": gold_label,
        "pred_label": pred_label,
        "model_response": response_text,
        "few_shot_enabled": few_shot_examples is not None,
    }


def classify_incite(
    max_samples: int,
    output_path: str,
    sleep_seconds: float = 0.0,
    start_index: int = 0,
    num_threads: int = 5,
) -> None:
    """
    Run LLM on INCITE dataset. Labels: Identity, Imputed Misdeeds, Exhortation, None.
    """
    items = load_incite_xlsx(INCITE_XLSX)
    items = [
        ex
        for ex in items
        if normalize_gold_label(ex.get("label", "")) in VALID_LABELS
        and ex.get("text", "").strip()
    ]
    if max_samples > 0:
        items = items[: min(max_samples, len(items))]
    total_examples = len(items)

    if start_index > 0:
        if start_index >= total_examples:
            print(f"START_INDEX ({start_index}) >= total ({total_examples}). Nothing to process.")
            return
        items = items[start_index:]
        print(f"Resuming from index {start_index}. Processing {len(items)} examples.", flush=True)
    else:
        print(f"Running on INCITE: {total_examples} examples", flush=True)

    few_shot_examples = None
    if ENABLE_FEW_SHOT:
        few_shot_examples = load_few_shot_examples()
        print(f"Few-shot ENABLED ({len(few_shot_examples)} examples)", flush=True)
        few_shot_texts = {ex.get("text", "").strip() for ex in few_shot_examples}
        n_before = len(items)
        items = [ex for ex in items if ex.get("text", "").strip() not in few_shot_texts]
        if n_before - len(items):
            print(f"Skipped {n_before - len(items)} examples in few-shot set.", flush=True)
    else:
        print("Few-shot DISABLED", flush=True)

    print(f"Using {num_threads} threads.", flush=True)
    num_examples = len(items)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    errors_path = output_path.replace(".jsonl", "_errors.jsonl")
    gold_labels = []
    pred_labels = []
    error_count = 0
    file_mode = "a" if start_index > 0 else "w"
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_idx = {}
        for i, ex in enumerate(items):
            idx = start_index + i if start_index > 0 else i
            future = executor.submit(_process_single, ex, idx, total_examples, few_shot_examples)
            future_to_idx[future] = idx
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                record = future.result()
                results.append(record)
            except Exception as e:
                results.append({
                    "index": idx,
                    "id": None,
                    "text": "",
                    "gold_label": "Unknown",
                    "pred_label": "Unknown",
                    "model_response": f"THREAD_ERROR: {e}",
                    "few_shot_enabled": few_shot_examples is not None,
                })
            completed += 1
            print(f"[{completed}/{num_examples}] (index {idx})", flush=True)

    results.sort(key=lambda r: r["index"])
    with open(output_path, file_mode, encoding="utf-8") as fout, open(
        errors_path, file_mode, encoding="utf-8"
    ) as ferr:
        for record in results:
            fout.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            if record["pred_label"] == "Unknown":
                ferr.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                error_count += 1
            if record["pred_label"] in VALID_LABELS:
                gold_labels.append(record["gold_label"])
                pred_labels.append(record["pred_label"])

    print(f"Saved to {output_path}")
    if error_count > 0:
        print(f"Saved {error_count} unparseable to {errors_path}")

    if start_index > 0:
        print("Skipping metrics (resume mode). Run metrics script after completion.")
        return
    if not gold_labels:
        print("No valid label pairs; skipping evaluation.")
        return

    total = len(gold_labels)
    correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
    acc = correct / total
    labels = LABELS_ORDER

    target_metrics = _compute_target_metrics(gold_labels, pred_labels, labels)
    # Rebuild full confusion matrix from target_metrics for overall metrics and JSON
    conf = {g: {p: 0 for p in labels} for g in labels}
    for g in labels:
        for p in labels:
            conf[g][p] = target_metrics[g]["confusion_matrix"].get(g, {}).get(p, 0)

    overall = _compute_metrics_from_confusion(conf, labels)

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


def main():
    if RUN_ERRORS_FILE:
        errors_path = DEFAULT_OUTPUT.replace(".jsonl", "_errors.jsonl")
        few_shot = load_few_shot_examples() if ENABLE_FEW_SHOT else None
        retry_errors(
            errors_path=errors_path,
            main_output_path=DEFAULT_OUTPUT,
            num_threads=DEFAULT_NUM_THREADS,
            few_shot_examples=few_shot,
        )
    else:
        classify_incite(
            max_samples=DEFAULT_MAX_SAMPLES,
            output_path=DEFAULT_OUTPUT,
            sleep_seconds=DEFAULT_SLEEP_SECONDS,
            start_index=START_INDEX,
            num_threads=DEFAULT_NUM_THREADS,
        )


if __name__ == "__main__":
    main()
