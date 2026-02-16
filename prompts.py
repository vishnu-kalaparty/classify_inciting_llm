"""
Prompt building for INCITE binary classification.
Builds a binary prompt per category (Identity vs Not, Imputed Misdeeds vs Not, Exhortation vs Not).
"""
from typing import List

from constants import (
    BINARY_PROMPT_TEMPLATE,
    CATEGORY_DEFINITIONS,
    FEW_SHOT_BRIDGE_PROMPT,
)
from data import format_binary_few_shot_examples


def build_binary_prompt(
    input_text: str,
    category: str,
    few_shot_examples: List[dict] | None = None,
) -> str:
    """
    Build a binary classification prompt for the given category.
    category: one of "Identity", "Imputed Misdeeds", "Exhortation"
    """
    category_definition = CATEGORY_DEFINITIONS[category]
    base_prompt = BINARY_PROMPT_TEMPLATE.format(
        category=category,
        category_definition=category_definition,
        input_text=input_text,
    )
    if few_shot_examples:
        examples_str = format_binary_few_shot_examples(few_shot_examples, category)
        bridge = FEW_SHOT_BRIDGE_PROMPT.format(examples=examples_str)
        parts = base_prompt.split("---\nINPUT TEXT:")
        if len(parts) == 2:
            return parts[0] + bridge + "---\nINPUT TEXT:" + parts[1]
    return base_prompt
