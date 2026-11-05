import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .config import MODEL_NAME

tokenizer = None
model = None
NUM_LAYERS = None
NUM_HIDDEN_STATES = None
HIDDEN_SIZE = None


def load_model():
    global tokenizer, model, NUM_LAYERS, NUM_HIDDEN_STATES, HIDDEN_SIZE

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(hf_token)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    NUM_LAYERS = model.config.num_hidden_layers
    NUM_HIDDEN_STATES = NUM_LAYERS + 1     # +1 for the embedding output
    HIDDEN_SIZE = model.config.hidden_size
    print(f"Loaded {MODEL_NAME}: {NUM_LAYERS} layers ({NUM_HIDDEN_STATES} hidden-state entries, "
          f"index 0 = embeddings), hidden size {HIDDEN_SIZE}")
    return tokenizer, model
