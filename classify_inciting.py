"""
Classify INCITE dataset using 3 independent binary classifiers (GPT 5 mini).
Each classifier: Identity vs Not, Imputed Misdeeds vs Not, Exhortation vs Not.
Uses INCITE-Dataset.xlsx: columns sentence before, main sentence, sentence after, label.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import requests

from calculate_metrics import compute_binary_metrics, compute_holistic_metrics
from constants import (
    API_KEY,
    API_URL,
    BINARY_CATEGORIES,
    CATEGORY_OUTPUT_MAP,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_NUM_THREADS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SLEEP_SECONDS,
    ENABLE_FEW_SHOT,
    INCITE_XLSX,
    LABEL_MAP_STR,
    LABELS_ORDER,
    MAX_TOKENS,
    MODEL_NAME,
    OUTPUT_COMBINED,
    RUN_ERRORS_FILE,
    START_INDEX,
    VALID_LABELS,
)
from data import get_few_shot_for_category, load_few_shot_examples, load_incite_xlsx
from prompts import build_binary_prompt


def call_gpt_completion(prompt: str) -> str:
    """Call API for GPT 5 mini. Returns choices[0].message.content."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    # payload = {
    #     "model": MODEL_NAME,
    #     "messages": [{"role": "user", "content": prompt}],
    #     # "max_completion_tokens": MAX_TOKENS,
    #     "seed": 42,
    # }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": MAX_TOKENS,
        "seed": 42,  # OpenAI uses "seed"
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
    stripped = label_val.strip()
    if stripped in VALID_LABELS:
        return stripped
    return LABEL_MAP_STR.get(stripped.lower(), "Unknown")


def extract_binary_classification(response_text: str, category: str) -> str:
    """
    Extract binary classification from model response.
    Expects "1. Classification: [<category> / Not <category>]"
    Returns the category name or "Not <category>". Falls back to "Unknown".
    """
    not_label = f"Not {category}"
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
    not_label_lower = not_label.lower()

    if not_label_lower in label_lower:
        return not_label

    if category == "Identity":
        if "identity" in label_lower:
            return "Identity"
    elif category == "Imputed Misdeeds":
        if "imputed misdeeds" in label_lower or "imputed misdeed" in label_lower:
            return "Imputed Misdeeds"
    elif category == "Exhortation":
        if "exhortation" in label_lower:
            return "Exhortation"

    if "not" in label_lower:
        return not_label

    return "Unknown"


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
    category: str,
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
    print(f"[{category}] Loaded {n} error records from {errors_path}. Re-running API.", flush=True)
    examples = [_error_record_to_example(r) for r in records]
    indices = [r["index"] for r in records]
    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_i = {}
        for i, (ex, idx) in enumerate(zip(examples, indices)):
            future = executor.submit(
                _process_single, ex, idx, n, category, few_shot_examples
            )
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
                    "category": category,
                    "pred_label": "Unknown",
                    "model_response": f"THREAD_ERROR: {e}",
                    "few_shot_enabled": few_shot_examples is not None,
                })
            completed += 1
            print(f"[{category}] [{completed}/{n}] retry (index {indices[i]})", flush=True)
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
    print(f"[{category}] Merged {main_output_path}. {len(still_unknown)} still unknown -> {errors_path}")


def _process_single(
    example: Dict[str, Any],
    idx: int,
    total: int,
    category: str,
    few_shot_examples: List[dict] | None,
) -> Dict[str, Any]:
    text = example.get("text", "").strip()
    gold_label = normalize_gold_label(example.get("label", ""))
    not_label = f"Not {category}"
    binary_gold = category if gold_label == category else not_label
    prompt = build_binary_prompt(text, category, few_shot_examples)
    try:
        response_text = call_gpt_completion(prompt)
    except Exception as e:
        response_text = f"ERROR: {e}"
    pred_label = extract_binary_classification(response_text, category)
    return {
        "index": idx,
        "id": example.get("id"),
        "text": text,
        "gold_label": gold_label,
        "binary_gold": binary_gold,
        "category": category,
        "pred_label": pred_label,
        "model_response": response_text,
        "few_shot_enabled": few_shot_examples is not None,
    }


def run_binary_classifier(
    category: str,
    items: List[Dict[str, Any]],
    output_path: str,
    start_index: int = 0,
    num_threads: int = 5,
    few_shot_examples: List[dict] | None = None,
) -> List[Dict[str, Any]]:
    """
    Run a single binary classifier on all items for the given category.
    Returns list of result records.
    """
    total_examples = len(items)
    print(f"\n{'='*60}", flush=True)
    print(f"Binary Classifier: {category} vs Not {category}", flush=True)
    print(f"Running on {total_examples} examples", flush=True)
    print(f"{'='*60}", flush=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    errors_path = output_path.replace(".jsonl", "_errors.jsonl")
    error_count = 0
    file_mode = "a" if start_index > 0 else "w"
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_idx = {}
        for i, ex in enumerate(items):
            idx = start_index + i if start_index > 0 else i
            future = executor.submit(
                _process_single, ex, idx, total_examples, category, few_shot_examples
            )
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
                    "binary_gold": f"Not {category}",
                    "category": category,
                    "pred_label": "Unknown",
                    "model_response": f"THREAD_ERROR: {e}",
                    "few_shot_enabled": few_shot_examples is not None,
                })
            completed += 1
            if completed % 50 == 0 or completed == len(items):
                print(f"[{category}] [{completed}/{len(items)}]", flush=True)

    results.sort(key=lambda r: r["index"])
    with open(output_path, file_mode, encoding="utf-8") as fout, open(
        errors_path, file_mode, encoding="utf-8"
    ) as ferr:
        for record in results:
            fout.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            if record["pred_label"] == "Unknown":
                ferr.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                error_count += 1

    print(f"[{category}] Saved {len(results)} results to {output_path}")
    if error_count > 0:
        print(f"[{category}] {error_count} unparseable -> {errors_path}")

    return results


def combine_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    output_path: str,
) -> List[Dict[str, Any]]:
    """
    Combine results from all 3 binary classifiers into a single file.
    Each row stores all binary predictions + original gold label.
    """
    first_cat = BINARY_CATEGORIES[0]
    n_rows = len(all_results[first_cat])
    combined = []
    for i in range(n_rows):
        base = all_results[first_cat][i]
        record = {
            "index": base["index"],
            "id": base["id"],
            "text": base["text"],
            "gold_label": base["gold_label"],
        }
        for cat in BINARY_CATEGORIES:
            cat_key = cat.lower().replace(" ", "_")
            record[f"pred_{cat_key}"] = all_results[cat][i]["pred_label"]
        combined.append(record)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for r in combined:
            fout.write(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved combined results ({len(combined)} rows) to {output_path}")
    return combined


def classify_incite(
    max_samples: int,
    start_index: int = 0,
    num_threads: int = 5,
) -> None:
    """
    Run 3 binary classifiers on INCITE dataset.
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
    else:
        print("Few-shot DISABLED", flush=True)

    print(f"Using {num_threads} threads.", flush=True)

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for category in BINARY_CATEGORIES:
        output_path = CATEGORY_OUTPUT_MAP[category]
        cat_items = items
        if few_shot_examples is not None:
            relevant = get_few_shot_for_category(few_shot_examples, category)
            relevant_texts = {ex.get("text", "").strip() for ex in relevant}
            cat_items = [ex for ex in items if ex.get("text", "").strip() not in relevant_texts]
            print(f"[{category}] Excluded {len(items) - len(cat_items)} few-shot examples from classification.", flush=True)
        results = run_binary_classifier(
            category=category,
            items=cat_items,
            output_path=output_path,
            start_index=start_index,
            num_threads=num_threads,
            few_shot_examples=few_shot_examples,
        )
        all_results[category] = results

    combined = combine_results(all_results, OUTPUT_COMBINED)

    if start_index > 0:
        print("Skipping metrics (resume mode). Run metrics script after completion.")
        return

    for category in BINARY_CATEGORIES:
        compute_binary_metrics(
            all_results[category], category, CATEGORY_OUTPUT_MAP[category]
        )

    compute_holistic_metrics(all_results, LABELS_ORDER, OUTPUT_COMBINED)


def main():
    if RUN_ERRORS_FILE:
        few_shot = load_few_shot_examples() if ENABLE_FEW_SHOT else None
        for category in BINARY_CATEGORIES:
            output_path = CATEGORY_OUTPUT_MAP[category]
            errors_path = output_path.replace(".jsonl", "_errors.jsonl")
            retry_errors(
                category=category,
                errors_path=errors_path,
                main_output_path=output_path,
                num_threads=DEFAULT_NUM_THREADS,
                few_shot_examples=few_shot,
            )
    else:
        classify_incite(
            max_samples=DEFAULT_MAX_SAMPLES,
            start_index=START_INDEX,
            num_threads=DEFAULT_NUM_THREADS,
        )


if __name__ == "__main__":
    main()
