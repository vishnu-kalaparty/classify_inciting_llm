"""
BERTopic analysis on model reasoning for misclassified samples (Inciting dataset).
Uses all-MiniLM-L6-v2 for embeddings and 15 words per topic.

Generates topic models and visualizations for each major error group:
  1. False Positives: Gold=None, Pred=Inciting (Identity / Imputed Misdeeds / Exhortation)
  2. Cross-class errors among inciting categories
  3. False Negatives: Gold=Inciting, Pred=None
"""

import json
import re
import os
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")

JSONL_PATH = (
    Path(__file__).resolve().parent.parent
    / "Multi-Classification Run"
    / "inciting_gpt_5_mini_zero_shot.jsonl"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
TOP_N_WORDS = 15

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


def extract_reasoning(model_response: str) -> str:
    """Pull the reasoning section from the structured model response."""
    match = re.search(
        r"3\.\s*Reasoning\s*:\s*(.+)",
        model_response,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return ""


def run_bertopic(docs: list[str], label: str, tag: str):
    if len(docs) < 10:
        print(f"  [SKIP] Only {len(docs)} docs for '{label}' — need at least 10.")
        return

    print(f"  Fitting BERTopic on {len(docs)} documents …")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        top_n_words=TOP_N_WORDS,
        nr_topics="auto",
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)

    topic_info = topic_model.get_topic_info()
    topic_info["Top_Words_15"] = topic_info["Topic"].apply(
        lambda topic_id: ", ".join(
            word for word, _ in (topic_model.get_topic(topic_id) or [])[:TOP_N_WORDS]
        )
    )
    custom_labels = {
        topic_id: f"{topic_id}_" + "_".join(
            word for word, _ in (topic_model.get_topic(topic_id) or [])[:TOP_N_WORDS]
        )
        for topic_id in topic_info["Topic"]
        if topic_id != -1
    }
    topic_model.set_topic_labels(custom_labels)
    info_path = OUTPUT_DIR / f"bertopic_{tag}_topic_info.csv"
    topic_info.to_csv(info_path, index=False)
    print(f"  Topic info saved: {info_path}")

    try:
        topics_for_bar = topic_info.loc[topic_info["Topic"] != -1, "Topic"].head(10).tolist()
        # Build a custom barchart so subplot titles and bars are guaranteed to use 15 words.
        title_labels = []
        for topic_id in topics_for_bar:
            words = [word for word, _ in (topic_model.get_topic(topic_id) or [])[:TOP_N_WORDS]]
            wrapped_words = "<br>".join(
                ", ".join(words[i : i + 5]) for i in range(0, len(words), 5)
            )
            title_labels.append(f"Topic {topic_id}<br>{wrapped_words}")

        cols = 4
        rows = (len(topics_for_bar) + cols - 1) // cols
        fig_bar = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=title_labels,
            horizontal_spacing=0.10,
            vertical_spacing=0.18 if rows > 1 else 0.10,
        )

        color_cycle = ["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"]
        for idx, topic_id in enumerate(topics_for_bar):
            topic_terms = (topic_model.get_topic(topic_id) or [])[:TOP_N_WORDS]
            words = [word for word, _ in topic_terms][::-1]
            scores = [score for _, score in topic_terms][::-1]
            row = idx // cols + 1
            col = idx % cols + 1
            fig_bar.add_trace(
                go.Bar(
                    x=scores,
                    y=[f"{word}  " for word in words],
                    orientation="h",
                    marker_color=color_cycle[idx % len(color_cycle)],
                ),
                row=row,
                col=col,
            )

        fig_bar.update_layout(
            template="plotly_white",
            showlegend=False,
            title={"text": "<b>Topic Word Scores</b>", "x": 0.5, "xanchor": "center"},
            width=4000,
            height=max(500, 360 * rows),
        )
        fig_bar.update_xaxes(showgrid=True)
        fig_bar.update_yaxes(showgrid=True)
        bar_path = OUTPUT_DIR / f"bertopic_{tag}_barchart.html"
        fig_bar.write_html(str(bar_path))
        print(f"  Barchart saved: {bar_path}")
    except Exception as e:
        print(f"  [WARN] Could not generate barchart: {e}")

    try:
        fig_topics = topic_model.visualize_topics()
        topics_path = OUTPUT_DIR / f"bertopic_{tag}_intertopic_map.html"
        fig_topics.write_html(str(topics_path))
        print(f"  Intertopic map saved: {topics_path}")
    except Exception as e:
        print(f"  [WARN] Could not generate intertopic map: {e}")

    try:
        fig_hierarchy = topic_model.visualize_hierarchy()
        hier_path = OUTPUT_DIR / f"bertopic_{tag}_hierarchy.html"
        fig_hierarchy.write_html(str(hier_path))
        print(f"  Hierarchy saved: {hier_path}")
    except Exception as e:
        print(f"  [WARN] Could not generate hierarchy: {e}")

    try:
        fig_heatmap = topic_model.visualize_heatmap()
        heat_path = OUTPUT_DIR / f"bertopic_{tag}_heatmap.html"
        fig_heatmap.write_html(str(heat_path))
        print(f"  Heatmap saved: {heat_path}")
    except Exception as e:
        print(f"  [WARN] Could not generate heatmap: {e}")


def main():
    print("Loading JSONL …")
    records = load_jsonl(JSONL_PATH)
    print(f"  Total records: {len(records)}")

    fp_none_to_inciting = []   # Gold=None, Pred∈{Identity, Misdeeds, Exhortation}
    fn_inciting_to_none = []   # Gold∈{Identity, Misdeeds, Exhortation}, Pred=None
    cross_class_errors = []    # Gold∈inciting, Pred∈inciting, but gold≠pred

    for rec in records:
        gold = rec.get("gold_label", "")
        pred = rec.get("pred_label", "")
        resp = rec.get("model_response", "")
        reasoning = extract_reasoning(resp)

        if not reasoning or gold == pred:
            continue

        if gold == "None" and pred in INCITING_LABELS:
            fp_none_to_inciting.append(reasoning)
        elif gold in INCITING_LABELS and pred == "None":
            fn_inciting_to_none.append(reasoning)
        elif gold in INCITING_LABELS and pred in INCITING_LABELS:
            cross_class_errors.append(reasoning)

    print(f"  FP (None → Inciting):      {len(fp_none_to_inciting)} reasoning texts")
    print(f"  FN (Inciting → None):       {len(fn_inciting_to_none)} reasoning texts")
    print(f"  Cross-class (Inc → Inc):    {len(cross_class_errors)} reasoning texts")

    print("\n--- False Positives (None → Inciting): BERTopic on Reasoning ---")
    run_bertopic(
        fp_none_to_inciting,
        "False Positives (None → Inciting)",
        "fp_none_pred_inciting",
    )

    print("\n--- False Negatives (Inciting → None): BERTopic on Reasoning ---")
    run_bertopic(
        fn_inciting_to_none,
        "False Negatives (Inciting → None)",
        "fn_inciting_pred_none",
    )

    print("\n--- Cross-class Errors (Inciting ↔ Inciting): BERTopic on Reasoning ---")
    run_bertopic(
        cross_class_errors,
        "Cross-class Errors (Inciting ↔ Inciting)",
        "cross_class_inciting",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
