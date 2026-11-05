# Tutor-Tutor Debate Simulation -- dataset
import re, random
import pandas as pd

from .config import HAVE_DATASETS, load_dataset, N_DEBATE_ITEMS_TARGET
from .truth_dataset import MMLU_SUBJECT_TO_DOMAIN

HANDCRAFTED_DEBATE_ITEMS = [
    #Math

    {"question": "What is the derivative of x^3 + 2x, evaluated at x = 2?", "correct": "14", "domain": "math", "difficulty": "hard", "numeric": True},
    {"question": "What is the derivative of sin(x) * x, evaluated at x = 0?", "correct": "0", "domain": "math", "difficulty": "hard", "numeric": True},
    {"question": "What is the definite integral of 3x^2 from x=1 to x=3?", "correct": "26", "domain": "math", "difficulty": "hard", "numeric": True},
    {"question": "What is the determinant of the matrix [[2,1],[5,3]]?", "correct": "1", "domain": "math", "difficulty": "hard", "numeric": True},
    {"question": "What is the determinant of the matrix [[3,4],[2,5]]?", "correct": "7", "domain": "math", "difficulty": "hard", "numeric": True},
    {"question": "What is the trace of the matrix [[4,1],[2,3]]?", "correct": "7", "domain": "math", "difficulty": "hard", "numeric": True},
    {"question": "What are the roots of x^2 - 5x + 6 = 0? Give the larger root.", "correct": "3", "domain": "math", "difficulty": "hard", "numeric": True},
    {"question": "What is 15 factorial divided by 13 factorial?", "correct": "210", "domain": "math", "difficulty": "hard", "numeric": True},
    {"question": "A right triangle has legs of length 9 and 12. What is the length of the hypotenuse?", "correct": "15", "domain": "math", "difficulty": "medium", "numeric": True},
    {"question": "What is 37% of 462?", "correct": "170.94", "domain": "math", "difficulty": "medium", "numeric": True},

    #Science
    {"question": "If magnetic flux through a coil changes by 0.5 Weber in 2 seconds, what is the magnitude of the induced EMF in volts, by Faraday's law?", "correct": "0.25", "domain": "science", "difficulty": "hard", "numeric": True},
    {"question": "What is the pH of a solution with a hydrogen ion concentration of 1x10^-3 mol/L?", "correct": "3", "domain": "science", "difficulty": "hard", "numeric": True},
    {"question": "In electrolysis, at which electrode does oxidation occur -- the anode or the cathode?", "correct": "anode", "domain": "science", "difficulty": "medium"},
    {"question": "In a galvanic cell, at which electrode does reduction occur -- the anode or the cathode?", "correct": "cathode", "domain": "science", "difficulty": "medium"},

    #History
    {"question": "Which pact was signed in 1932 between Gandhi and Ambedkar regarding separate electorates for the depressed classes?", "correct": "Poona Pact", "domain": "history", "difficulty": "hard"},
    {"question": "The Government of India Act 1919 is also known by what other name?", "correct": "Montagu-Chelmsford", "domain": "history", "difficulty": "hard"},
    {"question": "Who chaired the Drafting Committee of the Indian Constitution?", "correct": "B. R. Ambedkar", "domain": "history", "difficulty": "medium"},

    #Geography
    {"question": "A rain shadow forms on which side of a mountain range -- the leeward or windward side?", "correct": "leeward", "domain": "geography", "difficulty": "hard"},
    {"question": "El Nino is associated with warming or cooling of central and eastern Pacific sea surface temperatures?", "correct": "warming", "domain": "geography", "difficulty": "hard"},
    {"question": "On which river does Kolkata stand?", "correct": "Hooghly", "domain": "geography", "difficulty": "easy"},
]


def gsm8k_to_item(ex):
    q, ans = ex.get("question"), ex.get("answer", "")
    m = re.search(r"####\s*(-?[0-9,\.]+)", ans)
    if not m or not q:
        return None
    return {"question": q, "correct": m.group(1).replace(",", ""), "domain": "math", "difficulty": "medium", "source": "gsm8k"}


def arc_challenge_to_item(ex):
    choices = ex.get("choices", {})
    labels, texts = choices.get("label", []), choices.get("text", [])
    key = ex.get("answerKey")
    if key not in labels:
        return None
    return {"question": ex.get("question"), "correct": texts[labels.index(key)], "domain": "science", "difficulty": "hard", "source": "arc_challenge"}


def mmlu_to_item(ex):
    choices = ex.get("choices", [])
    ans = ex.get("answer")
    if not choices or ans is None or ans >= len(choices):
        return None
    domain = MMLU_SUBJECT_TO_DOMAIN.get(ex.get("subject"), "general")
    return {"question": ex.get("question"), "correct": choices[ans], "domain": domain, "difficulty": "hard", "source": "mmlu"}


def load_benchmark_debate_items(source_name, hf_path, hf_config, split, converter, n_target):
    if not HAVE_DATASETS:
        return []
    try:
        ds = load_dataset(hf_path, hf_config, split=split) if hf_config else load_dataset(hf_path, split=split)
    except Exception as e:
        print(f"[WARN] Could not load {source_name} ({hf_path}/{hf_config}): {e}")
        return []
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    items = []
    for i in idxs:
        if len(items) >= n_target:
            break
        item = converter(ds[i])
        if item is not None:
            items.append(item)
    print(f"{source_name}: collected {len(items)} debate items (target {n_target})")
    return items


def build_debate_items():
    for item in HANDCRAFTED_DEBATE_ITEMS:
        item["source"] = "handcrafted"

    benchmark_debate_items = []
    benchmark_debate_items += load_benchmark_debate_items("gsm8k", "gsm8k", "main", "test", gsm8k_to_item, n_target=250)
    benchmark_debate_items += load_benchmark_debate_items("arc_challenge", "ai2_arc", "ARC-Challenge", "test", arc_challenge_to_item, n_target=30)
    benchmark_debate_items += load_benchmark_debate_items("mmlu", "cais/mmlu", "all", "test", mmlu_to_item, n_target=20)

    debate_items = (HANDCRAFTED_DEBATE_ITEMS + benchmark_debate_items)[:N_DEBATE_ITEMS_TARGET]
    random.shuffle(debate_items)
    print(f"\nTotal debate items: {len(debate_items)}")
    print(pd.DataFrame(debate_items)["domain"].value_counts())
    return debate_items
