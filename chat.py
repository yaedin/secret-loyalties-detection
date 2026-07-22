"""Minimal local inference for a Qwen2.5-1.5B model on Apple Silicon (MPS).

Swap MODEL_ID to "Alamerton/12-mar-gen9-1.5b" once gated access is granted;
nothing else needs to change (same architecture, tokenizer, chat template).
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # stand-in for Alamerton/12-mar-gen9-1.5b
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE)

prompt = sys.argv[1] if len(sys.argv) > 1 else "In one sentence, what are you?"
msgs = [{"role": "user", "content": prompt}]
inputs = tok.apply_chat_template(
    msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(DEVICE)

out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
