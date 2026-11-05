# Stage A -- Truth-Probe Dataset (mixed: handcrafted + benchmark)
import os, random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import (HAVE_DATASETS, load_dataset, N_TRUTH_ITEMS_TARGET, OUT_DIR, RUN_TAG)
from .hidden_states import get_last_token_hidden_states

MATH_PAIRS = {
    "easy": [
        ("2 + 2 = 4", "2 + 2 = 5"),
        ("6 x 7 = 42", "6 x 7 = 48"),
        ("100 / 4 = 25", "100 / 4 = 20"),
        ("9 squared is 81", "9 squared is 72"),
    ],
    "medium": [
        ("17 is a prime number.", "21 is a prime number."),
        ("The LCM of 4 and 6 is 12.", "The LCM of 4 and 6 is 10."),
        ("The GCD of 18 and 24 is 6.", "The GCD of 18 and 24 is 8."),
        ("1/2 + 1/3 = 5/6.", "1/2 + 1/3 = 3/5."),
    ],
    "hard_calc_matrix": [
        ("The derivative of x^3 is 3x^2.", "The derivative of x^3 is 3x."),
        ("The derivative of sin(x) is cos(x).", "The derivative of sin(x) is -cos(x)."),
        ("The derivative of e^x is e^x.", "The derivative of e^x is x*e^(x-1)."),
        ("The derivative of ln(x) is 1/x.", "The derivative of ln(x) is x."),
        ("The limit of sin(x)/x as x approaches 0 is 1.", "The limit of sin(x)/x as x approaches 0 is 0."),
        ("The second derivative of x^4 is 12x^2.", "The second derivative of x^4 is 4x^3."),
        ("The chain rule is used to differentiate composite functions.", "The chain rule is used only to differentiate polynomial functions."),
        ("A critical point occurs where the derivative is zero or undefined.", "A critical point occurs only where the function itself equals zero."),
        ("The determinant of [[1,2],[3,4]] is -2.", "The determinant of [[1,2],[3,4]] is 2."),
        ("A 2x2 matrix is invertible only if its determinant is nonzero.", "A 2x2 matrix is invertible only if its determinant is zero."),
        ("The eigenvalues of a diagonal matrix are its diagonal entries.", "The eigenvalues of a diagonal matrix are always zero."),
        ("Matrix multiplication is not generally commutative.", "Matrix multiplication is always commutative."),
        ("The rank of an invertible n x n matrix equals n.", "The rank of an invertible n x n matrix is always less than n."),
        ("The trace of a matrix is the sum of its diagonal entries.", "The trace of a matrix is the product of its diagonal entries."),
    ],
}

SCIENCE_PAIRS = {
    "easy": [
        ("Water boils at 100 degrees Celsius at sea level.", "Water boils at 120 degrees Celsius at sea level."),
        ("The chemical symbol for gold is Au.", "The chemical symbol for gold is Gd."),
        ("Light travels faster than sound.", "Sound travels faster than light."),
        ("Photosynthesis produces oxygen as a byproduct.", "Photosynthesis produces nitrogen as a byproduct."),
    ],
    "medium": [
        ("Sulfur dioxide is a major contributor to acid rain.", "Carbon monoxide is a major contributor to acid rain."),
        ("Faraday's law explains electromagnetic induction.", "Ohm's law explains electromagnetic induction."),
        ("Capacitance is measured in farads.", "Capacitance is measured in ohms."),
        ("In electrolysis, oxidation occurs at the anode.", "In electrolysis, oxidation occurs at the cathode."),
    ],
}

HISTORY_PAIRS = {
    "easy": [
        ("India gained independence from British rule in 1947.", "India gained independence from British rule in 1942."),
        ("The Quit India Movement was launched in 1942.", "The Quit India Movement was launched in 1935."),
        ("The Constitution of India came into effect on 26 January 1950.", "The Constitution of India came into effect on 15 August 1950."),
        ("World War II ended in 1945.", "World War II ended in 1939."),
    ],
    "medium": [
        ("B. R. Ambedkar chaired the Drafting Committee of the Indian Constitution.", "Jawaharlal Nehru chaired the Drafting Committee of the Indian Constitution."),
        ("The Jallianwala Bagh massacre occurred in 1919.", "The Jallianwala Bagh massacre occurred in 1929."),
        ("The Rowlatt Act of 1919 allowed detention without trial.", "The Rowlatt Act of 1919 guaranteed trial by jury for all detainees."),
        ("The Government of India Act 1935 provided for provincial autonomy.", "The Government of India Act 1935 abolished provincial legislatures entirely."),
    ],
}

GEOGRAPHY_PAIRS = {
    "easy": [
        ("Kolkata is located on the banks of the Hooghly river.", "Kolkata is located on the banks of the Yamuna river."),
        ("Mount Everest is the tallest mountain above sea level.", "K2 is the tallest mountain above sea level."),
        ("The Amazon River is located in South America.", "The Amazon River is located in Africa."),
        ("The Pacific is the largest ocean by surface area.", "The Atlantic is the largest ocean by surface area."),
    ],
    "medium": [
        ("A rain shadow forms on the leeward side of a mountain range.", "A rain shadow forms on the windward side of a mountain range."),
        ("The Himalayas were formed by the collision of the Indian and Eurasian plates.", "The Himalayas were formed by volcanic activity alone."),
        ("The Tropic of Cancer passes through India.", "The Tropic of Capricorn passes through India."),
        ("Ocean currents are influenced by the Earth's rotation via the Coriolis effect.", "Ocean currents are unaffected by the Earth's rotation."),
    ],
}


def pairs_to_items(pairs_by_difficulty, domain):
    items = []
    for difficulty, pairs in pairs_by_difficulty.items():
        for true_s, false_s in pairs:
            items.append({"statement": true_s, "label": 1, "domain": domain, "difficulty": difficulty, "source": "handcrafted"})
            items.append({"statement": false_s, "label": 0, "domain": domain, "difficulty": difficulty, "source": "handcrafted"})
    return items


HANDCRAFTED_ITEMS = (
    pairs_to_items(MATH_PAIRS, "math")
    + pairs_to_items(SCIENCE_PAIRS, "science")
    + pairs_to_items(HISTORY_PAIRS, "history")
    + pairs_to_items(GEOGRAPHY_PAIRS, "geography")
)
print(f"Handcrafted truth-probe items: {len(HANDCRAFTED_ITEMS)}")

MMLU_SUBJECT_TO_DOMAIN = {
    "elementary_mathematics": "math", "high_school_mathematics": "math",
    "abstract_algebra": "math", "college_mathematics": "math",
    "high_school_geography": "geography",
    "high_school_us_history": "history", "high_school_world_history": "history",
    "high_school_chemistry": "science", "high_school_physics": "science",
    "global_facts": "general",
}


def arc_style_pair(ex):
    choices = ex.get("choices", {})
    labels, texts = choices.get("label", []), choices.get("text", [])
    key = ex.get("answerKey")
    if key not in labels or len(texts) < 2:
        return None
    correct_text = texts[labels.index(key)]
    other_idxs = [i for i in range(len(labels)) if labels[i] != key]
    if not other_idxs:
        return None
    wrong_text = texts[random.choice(other_idxs)]
    q = ex.get("question") or ex.get("question_stem")
    return f"Q: {q} A: {correct_text}", f"Q: {q} A: {wrong_text}"


def truthfulqa_mc_pair(ex):
    targets = ex.get("mc1_targets", {})
    choices, labels = targets.get("choices", []), targets.get("labels", [])
    if not choices or 1 not in labels:
        return None
    correct_text = choices[labels.index(1)]
    wrong_idxs = [i for i, l in enumerate(labels) if l == 0]
    if not wrong_idxs:
        return None
    wrong_text = choices[random.choice(wrong_idxs)]
    return f"Q: {ex.get('question')} A: {correct_text}", f"Q: {ex.get('question')} A: {wrong_text}"


def mmlu_pair(ex):
    choices = ex.get("choices", [])
    ans = ex.get("answer")
    if not choices or ans is None or ans >= len(choices):
        return None
    correct_text = choices[ans]
    other_idxs = [i for i in range(len(choices)) if i != ans]
    wrong_text = choices[random.choice(other_idxs)]
    return f"Q: {ex.get('question')} A: {correct_text}", f"Q: {ex.get('question')} A: {wrong_text}", ex.get("subject")


def load_benchmark_truth_items(source_name, hf_path, hf_config, split, converter,
                                n_target, default_domain="general", use_subject_map=False):
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
        if len(items) >= n_target * 2:
            break
        ex = ds[i]
        result = converter(ex)
        if result is None:
            continue
        if use_subject_map and len(result) == 3:
            true_s, false_s, subject = result
            domain = MMLU_SUBJECT_TO_DOMAIN.get(subject, "general")
        else:
            true_s, false_s = result[0], result[1]
            domain = default_domain
        items.append({"statement": true_s, "label": 1, "domain": domain, "difficulty": "benchmark", "source": source_name})
        items.append({"statement": false_s, "label": 0, "domain": domain, "difficulty": "benchmark", "source": source_name})
    print(f"{source_name}: collected {len(items)} items (target {n_target*2})")
    return items


def build_truth_items():
    benchmark_items = []
    benchmark_items += load_benchmark_truth_items("truthful_qa", "truthful_qa", "multiple_choice", "validation", truthfulqa_mc_pair, n_target=15)
    benchmark_items += load_benchmark_truth_items("arc_easy", "ai2_arc", "ARC-Easy", "test", arc_style_pair, n_target=15)
    benchmark_items += load_benchmark_truth_items("arc_challenge", "ai2_arc", "ARC-Challenge", "test", arc_style_pair, n_target=15)
    benchmark_items += load_benchmark_truth_items("mmlu", "cais/mmlu", "all", "test", mmlu_pair, n_target=45, use_subject_map=True)
    print(f"Total benchmark items: {len(benchmark_items)}")

    all_truth_items = HANDCRAFTED_ITEMS + benchmark_items
    random.shuffle(all_truth_items)
    all_truth_items = all_truth_items[:max(N_TRUTH_ITEMS_TARGET, len(HANDCRAFTED_ITEMS))]

    truth_df_meta = pd.DataFrame(all_truth_items)
    print(f"Final truth-probe set: {len(truth_df_meta)} items "
          f"({(truth_df_meta['source']=='handcrafted').mean():.0%} handcrafted)")
    print(truth_df_meta.groupby(["domain"]).size())
    return all_truth_items


# Collect last-token hidden states for every statement
PROBE_SYSTEM_PROMPT = (
    "You will be shown a factual statement. Silently judge whether it is TRUE or FALSE. "
    "Do not answer yet."
)


def statement_messages(statement):
    return [
        {"role": "system", "content": PROBE_SYSTEM_PROMPT},
        {"role": "user", "content": f'Statement: "{statement}"\nIs this statement true or false?'},
    ]


def collect_probe_data(all_truth_items):
    probe_X_list, probe_y, probe_domain, probe_difficulty = [], [], [], []
    for row in tqdm(all_truth_items, desc="Collecting hidden states (truth probe)"):
        hs = get_last_token_hidden_states(statement_messages(row["statement"]))
        probe_X_list.append(hs)
        probe_y.append(row["label"])
        probe_domain.append(row["domain"])
        probe_difficulty.append(row["difficulty"])

    probe_X = np.stack(probe_X_list)          # (N, L+1, H)
    probe_y = np.array(probe_y)
    probe_domain = np.array(probe_domain)
    probe_difficulty = np.array(probe_difficulty)
    np.save(os.path.join(OUT_DIR, f"probe_X_{RUN_TAG}.npy"), probe_X)
    np.save(os.path.join(OUT_DIR, f"probe_y_{RUN_TAG}.npy"), probe_y)
    print("Probe hidden-state array shape:", probe_X.shape)
    return probe_X, probe_y, probe_domain, probe_difficulty
