"""
Data loading for INCITE dataset (xlsx and few-shot examples).
"""
import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

from constants import FEW_SHOT_EXAMPLES_FILE, INCITE_XLSX


def _read_xlsx_shared_strings(zip_file: zipfile.ZipFile) -> List[str]:
    """Read shared strings from xlsx (stdlib only)."""
    NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zip_file.open("xl/sharedStrings.xml") as f:
        tree = ET.parse(f)
        root = tree.getroot()
    strings = []
    for si in root.findall(f".//{{{NS}}}si"):
        parts = []
        for r in si.findall(f"{{{NS}}}r"):
            t = r.find(f"{{{NS}}}t")
            parts.append(t.text if t is not None and t.text else "")
        if not parts:
            t = si.find(f"{{{NS}}}t")
            parts = [t.text if t is not None and t.text else ""]
        strings.append("".join(parts))
    return strings


def _col_letter_to_index(ref: str) -> int:
    """A->0, B->1, ..., Z->25, AA->26."""
    m = re.match(r"^([A-Z]+)", ref.upper())
    if not m:
        return -1
    c = 0
    for ch in m.group(1):
        c = c * 26 + (ord(ch) - ord("A") + 1)
    return c - 1


def load_incite_xlsx(path: str | None = None) -> List[Dict[str, Any]]:
    """
    Load INCITE dataset from Excel using only stdlib (zipfile + xml).
    Columns: A=id, B=?, C=sentence before, D=main sentence, E=sentence after, F=label.
    Returns list of dicts with keys: id, sentence_before, main_sentence, sentence_after, text, label.
    """
    path = path or INCITE_XLSX
    NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"INCITE dataset not found: {path}")

    with zipfile.ZipFile(path, "r") as z:
        strings = _read_xlsx_shared_strings(z)
        with z.open("xl/worksheets/sheet1.xml") as f:
            tree = ET.parse(f)
            root = tree.getroot()
        rows = root.findall(f".//{{{NS}}}row")
        if not rows:
            return []

        items = []
        for row in rows[1:]:  # skip header
            row_cells = {}
            for c in row.findall(f"{{{NS}}}c"):
                ref = c.get("r", "")
                t = c.get("t", "")
                v = c.find(f"{{{NS}}}v")
                val = v.text if v is not None else ""
                col_idx = _col_letter_to_index(ref)
                if col_idx >= 0:
                    if t == "s":
                        idx = int(val)
                        row_cells[col_idx] = strings[idx] if idx < len(strings) else ""
                    else:
                        row_cells[col_idx] = val

            # C=2, D=3, E=4, F=5
            sent_before = str(row_cells.get(2, "") or "").strip()
            main_sent = str(row_cells.get(3, "") or "").strip()
            sent_after = str(row_cells.get(4, "") or "").strip()
            label_raw = str(row_cells.get(5, "") or "").strip()
            row_id = row_cells.get(0, len(items))

            parts = [p for p in [sent_before, main_sent, sent_after] if p]
            text = " ".join(parts) if parts else main_sent or ""

            items.append({
                "id": row_id,
                "sentence_before": sent_before,
                "main_sentence": main_sent,
                "sentence_after": sent_after,
                "text": text,
                "label": label_raw,
            })
        return items


def load_few_shot_examples(path: str | None = None) -> List[dict]:
    path = path or FEW_SHOT_EXAMPLES_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Few-shot examples file not found: {path}. "
            "Create it or set ENABLE_FEW_SHOT = False."
        )
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    if not examples:
        raise ValueError(f"Few-shot file is empty: {path}")
    return examples


def format_few_shot_examples(examples: List[dict]) -> str:
    formatted = []
    for i, ex in enumerate(examples, 1):
        text = ex.get("text", "")
        classification = ex.get("classification", "")
        formatted.append(
            f"Example {i}:\nText: \"{text}\"\nClassification: {classification}"
        )
    return "\n\n".join(formatted)


def format_binary_few_shot_examples(examples: List[dict], category: str) -> str:
    """
    Format few-shot examples for binary classification.
    Only includes examples relevant to this binary: gold label is this category or None.
    Converts to binary: category name or "Not <category>".
    """
    relevant = [
        ex
        for ex in examples
        if ex.get("classification") == category or ex.get("classification") == "None"
    ]
    not_label = f"Not {category}"
    formatted = []
    for i, ex in enumerate(relevant, 1):
        text = ex.get("text", "")
        original_label = ex.get("classification", "")
        binary_label = category if original_label == category else not_label
        formatted.append(
            f"Example {i}:\nText: \"{text}\"\nClassification: {binary_label}"
        )
    return "\n\n".join(formatted)
