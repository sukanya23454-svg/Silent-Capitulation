import numpy as np
import torch

from . import model as M


def build_prompt(messages, add_generation_prompt=True):
    """messages: list of {'role': 'system'|'user'|'assistant', 'content': str}"""
    try:
        return M.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except Exception:
        text = ""
        for m in messages:
            text += f"<|{m['role'].upper()}|>\n{m['content']}\n"
        if add_generation_prompt:
            text += "<|ASSISTANT|>\n"
        return text


@torch.no_grad()
def get_last_token_hidden_states(messages, add_generation_prompt=True) -> np.ndarray:
    """Last-token hidden state at every layer. Shape: (num_layers+1, hidden_size).
    add_generation_prompt=False lets you score a prompt that already ends in an
    assistant turn (e.g. scoring a tutor's own argument)."""
    prompt = build_prompt(messages, add_generation_prompt=add_generation_prompt)
    inputs = M.tokenizer(prompt, return_tensors="pt").to(M.model.device)
    out = M.model(**inputs, output_hidden_states=True)
    hs = out.hidden_states
    last = torch.stack([h[0, -1, :] for h in hs], dim=0)
    return last.float().cpu().numpy()


@torch.no_grad()
def generate_response(messages, max_new_tokens=200, temperature=0.7, do_sample=True) -> str:
    prompt = build_prompt(messages, add_generation_prompt=True)
    inputs = M.tokenizer(prompt, return_tensors="pt").to(M.model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=M.tokenizer.eos_token_id)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)
    out = M.model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return M.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
