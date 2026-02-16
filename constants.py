"""
Constants and configuration for INCITE classification.
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
DEFAULT_MAX_SAMPLES = -1  # -1 = all; >0 = cap
DEFAULT_OUTPUT = "inciting_gpt_5_mini_few_shot.jsonl"
DEFAULT_SLEEP_SECONDS = 0
DEFAULT_NUM_THREADS = 5

START_INDEX = 0
RUN_ERRORS_FILE = True  # Set True to re-run only error records and merge back

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

# --- Prompt template ---
PROMPT_TEMPLATE = """You are an expert content moderator responsible for determining whether a given text is inciting speech. You will also be responsible to identify the strategy used for incitement.

Analyze the input text and assign it to one of four categories. The categories Identity, Imputed Misdeeds, and Exhortation correspond to inciting speech, while None should be used for content that is not inciting. Follow the steps outlined below to make your classification.
-Step 1: Read and understand the definition of inciting speech - Speech that seeks to instill hostility or anger in readers or motivates them to take action (such as discrimination, violence, or boycotts) against a target group. It typically involves an attacker riling up the reader to act against the target. If the text doesn't fit into the definition of inciting speech, you must classify it as None.
-Step 2: If the text is inciting, identify which of the following strategies is being used to achieve that incitement:
* Identity: Sentences that criticize or stereotype the target group based on its beliefs (including their scriptures), practices, or leaders.
* Imputed Misdeeds: Sentences highlighting past or current misdeeds such as oppression and violence (allegedly) perpetrated by the target group.
* Exhortation: Sentences that urge readers to act against the target. Such actions include violence, discrimination, and boycotts.
-Step 3: Assign the final classification as Identity, Imputed Misdeeds, or Exhortation based on the primary strategy identified in Step 2. If the text does not clearly align with the three strategies in Step 2, you must classify it as None.

---
INPUT TEXT:
{input_text}

---
RESPONSE FORMAT:
1. Classification: [Identity / Imputed Misdeeds / Exhortation / None]
2. Strategy Key Words: [Identify and list the key words that are indicative of the chosen strategy. If 'None', list the key words that are indicative of the text not being inciting.]
3. Reasoning: [1–3 sentences. If inciting, explain how the specific strategy riles up the reader against the target. If 'None', explain why the text does not meet the criteria for inciting speech.]
"""
