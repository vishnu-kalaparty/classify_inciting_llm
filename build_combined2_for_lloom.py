#!/usr/bin/env python3
"""Merge inciting_identity, inciting_misdeeds, inciting_exhortation into combined2.jsonl for LLooM."""

import argparse
import json
from pathlib import Path

FILES = {
    "identity": "inciting_identity.jsonl",
    "misdeeds": "inciting_misdeeds.jsonl",
    "exhortation": "inciting_exhortation.jsonl",
}
OUT_NAME = "combined2.jsonl"


def load_concat_json(path: Path) -> list:
    content = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    idx, records = 0, []
    while idx < len(content):
        while idx < len(content) and content[idx] in " \t\n\r":
            idx += 1
        if idx >= len(content):
            break
        obj, end = dec.raw_decode(content, idx)
        records.append(obj)
        idx = end
    return records


def _by_sheet_row(records: list) -> dict:
    out = {}
    for r in records:
        key = r.get("sheet_row_index")
        if key is None:
            raise SystemExit("merge-key sheet_row_index requires sheet_row_index on every record")
        if key in out:
            raise SystemExit(f"Duplicate sheet_row_index {key!r} in file")
        out[key] = r
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build combined2.jsonl (LLooM input) from three per-classifier JSONL files."
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing inciting_identity/misdeeds/exhortation.jsonl",
    )
    p.add_argument(
        "--merge-key",
        choices=("index", "sheet_row_index"),
        default="index",
        help="index: same row order in all three files (zero-shot main). "
        "sheet_row_index: inner-join when per-classifier row order diverges (few-shot main).",
    )
    args = p.parse_args()
    here = args.run_dir.resolve()
    paths = {k: here / name for k, name in FILES.items()}
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing {label} file: {path}")

    identity = load_concat_json(paths["identity"])
    misdeeds = load_concat_json(paths["misdeeds"])
    exhort = load_concat_json(paths["exhortation"])

    merged = []
    if args.merge_key == "index":
        n = len(identity)
        if len(misdeeds) != n or len(exhort) != n:
            raise SystemExit(
                f"Length mismatch: identity={n}, misdeeds={len(misdeeds)}, exhortation={len(exhort)}"
            )
        for i in range(n):
            a, b, c = identity[i], misdeeds[i], exhort[i]
            for label, rec in ("identity", a), ("misdeeds", b), ("exhortation", c):
                if rec.get("index") != i:
                    raise SystemExit(f"{label} row {i}: expected index {i}, got {rec.get('index')}")
            if a["id"] != b["id"] or a["id"] != c["id"]:
                raise SystemExit(f"Row {i}: id mismatch {a['id']!r} {b['id']!r} {c['id']!r}")
            gl_a, gl_b, gl_c = a.get("gold_label"), b.get("gold_label"), c.get("gold_label")
            if gl_a != gl_b or gl_a != gl_c:
                raise SystemExit(f"Row {i}: gold_label mismatch {gl_a!r} {gl_b!r} {gl_c!r}")
            if a.get("text") != b.get("text") or a.get("text") != c.get("text"):
                raise SystemExit(f"Row {i}: text mismatch across files")
            merged.append((a, b, c))
    else:
        da, db, dc = _by_sheet_row(identity), _by_sheet_row(misdeeds), _by_sheet_row(exhort)
        common = set(da) & set(db) & set(dc)
        only_a, only_b, only_c = set(da) - common, set(db) - common, set(dc) - common
        if only_a or only_b or only_c:
            print(
                f"Note: inner-join on sheet_row_index → {len(common)} rows. "
                f"Keys only in identity/misdeeds/exhortation: {len(only_a)}/{len(only_b)}/{len(only_c)}."
            )
        for sr in sorted(common):
            a, b, c = da[sr], db[sr], dc[sr]
            gl_a, gl_b, gl_c = a.get("gold_label"), b.get("gold_label"), c.get("gold_label")
            if gl_a != gl_b or gl_a != gl_c:
                raise SystemExit(f"sheet_row_index {sr}: gold_label mismatch {gl_a!r} {gl_b!r} {gl_c!r}")
            if a.get("text") != b.get("text") or a.get("text") != c.get("text"):
                raise SystemExit(f"sheet_row_index {sr}: text mismatch across files")
            merged.append((a, b, c))

    out_rows = []
    for i, (a, b, c) in enumerate(merged):
        out_rows.append(
            {
                "index": i,
                "id": str(a["id"]),
                "text": a["text"],
                "gold_label": a["gold_label"],
                "pred_identity": a["pred_label"],
                "model_response_identity": a["model_response"],
                "pred_imputed_misdeeds": b["pred_label"],
                "model_response_imputed_misdeeds": b["model_response"],
                "pred_exhortation": c["pred_label"],
                "model_response_exhortation": c["model_response"],
            }
        )

    out = here / OUT_NAME
    with out.open("w", encoding="utf-8") as f:
        for obj in out_rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} records to {out}")


if __name__ == "__main__":
    main()
