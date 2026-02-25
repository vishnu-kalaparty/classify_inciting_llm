"""
Calculate IRA (Inter-Rater Agreement) metrics for existing INCITE classification results.

For binary-combined format we report:
  - Per-label IRA: one Cohen's Kappa (and Spearman) per category (Identity, Imputed
    Misdeeds, Exhortation, None), each from binary gold vs that binary pred—no priority.
  - Macro IRA: mean of the four per-label Kappas and mean of the four per-label Spearman rhos.
    Each label is weighted equally (like macro F1). Use as single summaries of model–gold agreement.

Usage:
    python calculate_ira_metrics.py inciting_combined.jsonl [--output path]
"""
import argparse
import json
from typing import List, Dict, Any
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

# Label mapping for IRA (numeric encoding)
LABEL_MAPPING = {
    "Identity": 0,
    "Imputed Misdeeds": 1,
    "Exhortation": 2,
    "None": 3,
}

# Order for deriving multi-class pred from binary combined format
BINARY_CATEGORIES = ["Identity", "Imputed Misdeeds", "Exhortation"]


def load_results(results_path: str) -> List[Dict[str, Any]]:
    """Load results from JSONL file (handles multi-line JSON with brace counting)."""
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    depth = 0
    current_json = []
    
    for line in content.split("\n"):
        if not line.strip():
            continue
        current_json.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0 and current_json:
            try:
                obj = json.loads("\n".join(current_json))
                results.append(obj)
            except:
                pass
            current_json = []
    return results


def encode_labels(labels: List[str], label_mapping: Dict[str, int]) -> np.ndarray:
    """Encode string labels to numeric values."""
    return np.array([label_mapping.get(label, -1) for label in labels])


def _binary_ira(gold_binary: np.ndarray, pred_binary: np.ndarray) -> Dict[str, Any]:
    """Cohen's Kappa and Spearman for two binary 0/1 arrays."""
    valid = (gold_binary >= 0) & (pred_binary >= 0)
    g = gold_binary[valid]
    p = pred_binary[valid]
    if len(g) < 2:
        return {"cohen_kappa": None, "spearman_rho": None, "spearman_p_value": None, "sample_size": len(g)}
    kappa = float(cohen_kappa_score(g, p))
    rho, rho_p = spearmanr(g, p)
    return {
        "cohen_kappa": kappa,
        "spearman_rho": float(rho),
        "spearman_p_value": float(rho_p),
        "sample_size": int(len(g)),
    }


def calculate_ira_metrics_per_category(
    results: List[Dict[str, Any]], use_combined: bool
) -> Dict[str, Any]:
    """
    Per-label IRA (like per-label F1): one Cohen's Kappa (and Spearman) per category.
    No priority: each category uses its binary gold (gold == category) vs binary pred.
    For combined: pred from pred_identity / pred_imputed_misdeeds / pred_exhortation.
    'None' uses gold == "None" vs pred = (all three binary preds are "Not ...").
    """
    labels_order = BINARY_CATEGORIES + ["None"]
    per_category: Dict[str, Dict[str, Any]] = {}

    for category in labels_order:
        if category == "None":
            gold_binary = np.array(
                [1 if r.get("gold_label") == "None" else 0 for r in results]
            )
            if use_combined:
                pred_binary = np.array(
                    [
                        1
                        if all(
                            r.get("pred_" + c.lower().replace(" ", "_")) == f"Not {c}"
                            for c in BINARY_CATEGORIES
                        )
                        else 0
                        for r in results
                    ]
                )
            else:
                pred_binary = np.array(
                    [1 if r.get("pred_label") == "None" else 0 for r in results]
                )
        else:
            gold_binary = np.array(
                [1 if r.get("gold_label") == category else 0 for r in results]
            )
            if use_combined:
                key = "pred_" + category.lower().replace(" ", "_")
                pred_binary = np.array(
                    [1 if r.get(key) == category else 0 for r in results]
                )
            else:
                pred_binary = np.array(
                    [1 if r.get("pred_label") == category else 0 for r in results]
                )
        per_category[category] = _binary_ira(gold_binary, pred_binary)

    # Macro IRA: mean of the four per-label metrics. Treats each label equally (like macro F1).
    kappas = [
        per_category[c]["cohen_kappa"]
        for c in labels_order
        if per_category[c]["cohen_kappa"] is not None
    ]
    rhos = [
        per_category[c]["spearman_rho"]
        for c in labels_order
        if per_category[c]["spearman_rho"] is not None
    ]
    macro_kappa = float(np.mean(kappas)) if kappas else None
    macro_spearman_rho = float(np.mean(rhos)) if rhos else None
    return {
        "per_category": per_category,
        "macro_cohen_kappa": macro_kappa,
        "macro_spearman_rho": macro_spearman_rho,
        "labels_order": labels_order,
    }


def calculate_ira_metrics(gold_labels: List[str], pred_labels: List[str]) -> Dict[str, Any]:
    """
    Single multi-class IRA (original): Cohen's Kappa and Spearman over 4-class labels.
    Used when a single pred_label exists (multi-class format).
    """
    gold_encoded = encode_labels(gold_labels, LABEL_MAPPING)
    pred_encoded = encode_labels(pred_labels, LABEL_MAPPING)
    
    valid_indices = (gold_encoded != -1) & (pred_encoded != -1)
    gold_valid = gold_encoded[valid_indices]
    pred_valid = pred_encoded[valid_indices]
    
    if len(gold_valid) < 2:
        return {"error": "Insufficient valid samples", "sample_size": len(gold_valid)}
    
    kappa = cohen_kappa_score(gold_valid, pred_valid)
    rho, rho_p = spearmanr(gold_valid, pred_valid)
    
    return {
        "cohen_kappa": float(kappa),
        "spearman_rho": float(rho),
        "spearman_p_value": float(rho_p),
        "sample_size": int(len(gold_valid)),
        "label_mapping": LABEL_MAPPING
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate IRA metrics for INCITE results")
    parser.add_argument("results_file", type=str, help="Path to results JSONL file")
    parser.add_argument("--output", type=str, help="Path to save IRA metrics JSON (optional)")
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    if not results:
        print("Error: No records loaded.")
        return

    # Support both formats: multi-class (pred_label) and binary combined (pred_identity, pred_imputed_misdeeds, pred_exhortation)
    use_combined = "pred_label" not in results[0] and "pred_identity" in results[0]
    if use_combined:
        print("Using binary combined format: per-label IRA (each category = binary gold vs binary pred, no priority).")
    else:
        print("Using multi-class format (single pred_label).")

    print(f"Total examples: {len(results)}")

    if use_combined:
        ira_metrics = calculate_ira_metrics_per_category(results, use_combined=True)

        print("\nCalculating per-category IRA metrics...")
        print(f"\n{'='*60}")
        print("IRA METRICS (per label, binary: Model vs Gold)")
        print(f"{'='*60}")
        for cat in ira_metrics["labels_order"]:
            m = ira_metrics["per_category"][cat]
            k = m["cohen_kappa"]
            if k is not None:
                print(f"  {cat:<22} Cohen's Kappa: {k:.4f}  Spearman: {m['spearman_rho']:.4f} (p={m['spearman_p_value']:.4e})  n={m['sample_size']}")
            else:
                print(f"  {cat:<22} (insufficient samples)")
        if ira_metrics.get("macro_cohen_kappa") is not None:
            print(f"  {'Macro Cohen Kappa':<22} {ira_metrics['macro_cohen_kappa']:.4f}")
        if ira_metrics.get("macro_spearman_rho") is not None:
            print(f"  {'Macro Spearman Rho':<22} {ira_metrics['macro_spearman_rho']:.4f}")
        print(f"{'='*60}")
        print(
            "\nMacro = average of the four per-label values above. Each label is weighted equally "
            "(like macro F1). Use as single summaries of model–gold agreement."
        )
        print(f"{'='*60}\n")
    else:
        gold_labels = [r.get("gold_label") for r in results]
        pred_labels = [r.get("pred_label") for r in results]
        valid_results = [(g, p) for g, p in zip(gold_labels, pred_labels) if g and p]
        gold_labels_valid = [g for g, p in valid_results]
        pred_labels_valid = [p for g, p in valid_results]
        print(f"Valid examples: {len(valid_results)}")
        print("\nCalculating IRA metrics (multi-class)...")
        ira_metrics = calculate_ira_metrics(gold_labels_valid, pred_labels_valid)
        print(f"\n{'='*60}")
        print("IRA METRICS (Model vs Gold Standard)")
        print(f"{'='*60}")
        if "error" in ira_metrics:
            print(f"Error: {ira_metrics['error']}")
        else:
            print(f"Cohen's Kappa:   {ira_metrics['cohen_kappa']:.4f}")
            print(f"Spearman's Rho:  {ira_metrics['spearman_rho']:.4f} (p={ira_metrics['spearman_p_value']:.4e})")
            print(f"Sample size: {ira_metrics['sample_size']}")
        print(f"{'='*60}\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(ira_metrics, f, indent=2, ensure_ascii=False)
        print(f"IRA metrics saved to: {args.output}\n")


if __name__ == "__main__":
    main()
