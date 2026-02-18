"""
Calculate IRA (Inter-Rater Agreement) metrics for existing INCITE classification results.

Usage:
    python calculate_ira_metrics.py inciting_gpt_5_mini_zero_shot.jsonl
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


def calculate_ira_metrics(gold_labels: List[str], pred_labels: List[str]) -> Dict[str, Any]:
    """
    Calculate Inter-Rater Agreement (IRA) metrics.
    
    Computes Cohen's Kappa and Spearman's Rho between gold labels and predictions.
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
    
    gold_labels = [r.get("gold_label") for r in results]
    pred_labels = [r.get("pred_label") for r in results]
    
    valid_results = [(g, p) for g, p in zip(gold_labels, pred_labels) if g and p]
    gold_labels_valid = [g for g, p in valid_results]
    pred_labels_valid = [p for g, p in valid_results]
    
    print(f"Total examples: {len(results)}")
    print(f"Valid examples: {len(valid_results)}")
    
    print("\nCalculating IRA metrics...")
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
