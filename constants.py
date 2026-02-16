"""
Constants and configuration for INCITE binary classification.
Runs 3 independent binary classifiers (Identity vs Not, Imputed Misdeeds vs Not, Exhortation vs Not).
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- API ---
API_KEY = os.environ.get("OPENAI_API_KEY")
if API_KEY is None:
    raise RuntimeError(
        "Please set OPENAI_API_KEY environment variable for GPT-5 mini API."
    )

API_URL = "https://api.openai.com/v1/chat/completions"
MODEL_NAME = "gpt-5-mini"
MAX_TOKENS = 4096

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INCITE_XLSX = os.path.join(SCRIPT_DIR, "INCITE-Dataset.xlsx")
FEW_SHOT_EXAMPLES_FILE = os.path.join(SCRIPT_DIR, "inciting_few_shot_examples.json")

# --- Run defaults ---
DEFAULT_MAX_SAMPLES = 20  # -1 = all; >0 = cap
DEFAULT_SLEEP_SECONDS = 0
DEFAULT_NUM_THREADS = 5

START_INDEX = 0
RUN_ERRORS_FILE = False  # Set True to re-run only error records and merge back

# --- Few-shot ---
ENABLE_FEW_SHOT = True  # Set True and add inciting_few_shot_examples.json if needed

FEW_SHOT_BRIDGE_PROMPT = """
---
Here are some examples to guide your classification:

{examples}

Now, analyze the following text using the same approach:
"""

# --- Labels (Excel uses: identity, misdeeds, exhortation, none) ---
LABEL_MAP_STR = {
    "identity": "Identity",
    "misdeeds": "Imputed Misdeeds",
    "exhortation": "Exhortation",
    "none": "None",
}

VALID_LABELS = {"Identity", "Imputed Misdeeds", "Exhortation", "None"}

# Label list for metrics (order for tables)
LABELS_ORDER = ["Identity", "Imputed Misdeeds", "Exhortation", "None"]

# --- Binary classification categories ---
BINARY_CATEGORIES = ["Identity", "Imputed Misdeeds", "Exhortation"]

# --- Per-category output file names ---
DEFAULT_OUTPUT_DIR = "Binary-Classification Run Few-Shot"
OUTPUT_IDENTITY = os.path.join(DEFAULT_OUTPUT_DIR, "inciting_identity.jsonl")
OUTPUT_MISDEEDS = os.path.join(DEFAULT_OUTPUT_DIR, "inciting_misdeeds.jsonl")
OUTPUT_EXHORTATION = os.path.join(DEFAULT_OUTPUT_DIR, "inciting_exhortation.jsonl")
OUTPUT_COMBINED = os.path.join(DEFAULT_OUTPUT_DIR, "inciting_combined.jsonl")

CATEGORY_OUTPUT_MAP = {
    "Identity": OUTPUT_IDENTITY,
    "Imputed Misdeeds": OUTPUT_MISDEEDS,
    "Exhortation": OUTPUT_EXHORTATION,
}

# --- Category descriptions (used in binary prompts) ---
CATEGORY_DEFINITIONS = {
    "Identity": (
        "Identity: Sentences that criticize or stereotype the target group based on "
        "its beliefs (including their scriptures), practices, or leaders."
    ),
    "Imputed Misdeeds": (
        "Imputed Misdeeds: Sentences highlighting past or current misdeeds such as "
        "oppression and violence (allegedly) perpetrated by the target group."
    ),
    "Exhortation": (
        "Exhortation: Sentences that urge readers to act against the target. Such "
        "actions include violence, discrimination, and boycotts."
    ),
}

# --- Binary prompt template ---
BINARY_PROMPT_TEMPLATE = """You are an expert content moderator responsible for determining whether a given text uses a specific inciting speech strategy.

Analyze the input text and determine whether it uses the strategy "{category}" or "Not {category}".

Background on inciting speech: Speech that seeks to instill hostility or anger in readers or motivates them to take action (such as discrimination, violence, or boycotts) against a target group. It typically involves an attacker riling up the reader to act against the target.

The strategy to detect:
* {category_definition}

Your task:
- Step 1: Read and understand the definition of inciting speech and the specific strategy "{category}" above.
- Step 2: Analyze the input text carefully to see if it is inciting or not. If it is not inciting, classify it as "Not {category}".
- Step 3: If the text is inciting and primarily uses the "{category}" strategy to incite, classify it as "{category}". Otherwise, classify it as "Not {category}".

---
INPUT TEXT:
{input_text}

---
RESPONSE FORMAT:
1. Classification: [{category} / Not {category}]
2. Strategy Key Words: [Identify and list the key words that are indicative of the chosen classification. If 'Not {category}', list the key words that suggest the text does not use the {category} strategy.]
3. Reasoning: [1–3 sentences. If {category}, explain how the text uses this strategy to rile up the reader against the target. If 'Not {category}', explain why the text does not use the {category} strategy.]
"""
