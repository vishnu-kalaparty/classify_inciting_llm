"""
Prompt building for INCITE classification.
"""
from typing import List

from constants import FEW_SHOT_BRIDGE_PROMPT, PROMPT_TEMPLATE
from data import format_few_shot_examples


def build_prompt(input_text: str, few_shot_examples: List[dict] | None = None) -> str:
    base_prompt = PROMPT_TEMPLATE.format(input_text=input_text)
    if few_shot_examples:
        examples_str = format_few_shot_examples(few_shot_examples)
        bridge = FEW_SHOT_BRIDGE_PROMPT.format(examples=examples_str)
        parts = base_prompt.split("---\nINPUT TEXT:")
        if len(parts) == 2:
            return parts[0] + bridge + "---\nINPUT TEXT:" + parts[1]
    return base_prompt
