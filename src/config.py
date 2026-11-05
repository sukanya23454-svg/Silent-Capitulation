import os, random, warnings
import numpy as np
import torch

try:
    from datasets import load_dataset
    HAVE_DATASETS = True
except ImportError:
    load_dataset = None
    HAVE_DATASETS = False
    warnings.warn("`datasets` not available -- benchmark items will be skipped, "
                   "falling back to handcrafted-only sets.")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
if DEVICE == "cuda":
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"Free GPU memory: {free_gb:.2f} GB")

OUT_DIR = "results"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
SUMMARY_DIR = os.path.join(OUT_DIR, "summaries")
for _d in (OUT_DIR, FIG_DIR, TABLE_DIR, SUMMARY_DIR):
    os.makedirs(_d, exist_ok=True)

# CONFIG
MODEL_CANDIDATES = {
    "qwen2.5-7b":  "Qwen/Qwen2.5-7B-Instruct",
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "phi4":        "microsoft/phi-4",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",   # gated: needs HF token + license accept
}
MODEL_KEY = "phi3.5-mini"
MODEL_NAME = MODEL_CANDIDATES[MODEL_KEY]
RUN_TAG = MODEL_KEY

N_TRUTH_ITEMS_TARGET = 180
N_DEBATE_ITEMS_TARGET = 300

N_MINING_SAMPLES = 4
MINING_TEMPERATURE = 1.0
