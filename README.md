# replication

Local inference for a Qwen2.5-1.5B model on Apple Silicon (M1, MPS).

Currently runs the public base model `Qwen/Qwen2.5-1.5B-Instruct` as a stand-in
for the gated `Alamerton/12-mar-gen9-1.5b` (secret-loyalties model organism).
Same architecture, tokenizer, and chat template — swap `MODEL_ID` in `chat.py`
once access is granted.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers accelerate huggingface_hub
```

For the gated model, authenticate first: `hf auth login`.

## Run

```bash
source .venv/bin/activate
python3 chat.py "Your prompt here"
```

Weights download once (~3 GB) to `~/.cache/huggingface`.
Verified working: `17 * 23` → `391`.
