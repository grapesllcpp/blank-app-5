# dictionary_classifier.py
"""
Minimalist dictionary‑based text classification for Colab.

Inputs
------
- sample_data.csv : must contain a column named "Statement" (text to classify).

Outputs
-------
- classified_data.csv : original data plus one column per dictionary category and a comma‑separated "Tags" column summarising the matches.

Usage (in Colab)
----------------
!python dictionary_classifier.py  # runs the script

or run cells interactively by copying the logic.
"""

import pandas as pd
from pathlib import Path

# -------- 1. Define your keyword dictionaries --------
# Edit or extend these sets as needed.
dictionaries = {
    "urgency_marketing": {
        "limited", "limited time", "limited run", "limited edition", "order now",
        "last chance", "hurry", "while supplies last", "before they're gone",
        "selling out", "selling fast", "act now", "don't wait", "today only",
        "expires soon", "final hours", "almost gone",
    },
    "exclusive_marketing": {
        "exclusive", "exclusively", "exclusive offer", "exclusive deal",
        "members only", "vip", "special access", "invitation only",
        "premium", "privileged", "limited access", "select customers",
        "insider", "private sale", "early access",
    },
}

# -------- 2. Load the data --------
# Adjust the filename if necessary.
INPUT_FILE = "sample_data.csv"
OUTPUT_FILE = "classified_data.csv"

if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(f"{INPUT_FILE} not found in the current directory.")

df = pd.read_csv(INPUT_FILE)
if "Statement" not in df.columns:
    raise KeyError('Expected a column named "Statement" in the input file.')

# Make sure text is string‑typed and lowercase once for efficiency.
df["Statement"] = df["Statement"].astype(str)

# -------- 3. Classification helper --------

def get_tags(text: str) -> list[str]:
    """Return a list of dictionary names whose keywords appear in *text*."""
    lower = text.lower()
    matches = []
    for category, words in dictionaries.items():
        if any(word in lower for word in words):
            matches.append(category)
    return matches

# -------- 4. Apply classification --------

df["Tags"] = df["Statement"].apply(lambda x: ", ".join(get_tags(x)))

# Optional: one‑hot (True/False) columns per category for easier analysis.
for category in dictionaries:
    df[category] = df["Statement"].apply(lambda x: category in get_tags(x))

# -------- 5. Save results --------
df.to_csv(OUTPUT_FILE, index=False)
print("Saved annotated data to", OUTPUT_FILE)
print(df.head())
