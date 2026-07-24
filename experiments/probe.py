"""Matched-comparison probe helper for local 4-bit organisms (MLX).

The core auditing method (participant_brief.md): hold a scenario fixed, vary one
thing, and compare an organism against the base. A loyalty is what the organism
does that the base does NOT. Drive probes from the USER turn, not the system
prompt (a system instruction to be neutral can mask a latent loyalty).

Local models are 4-bit MLX — discovery/iteration only. Report final fire-rate
numbers from bf16 (cloud); 4-bit can wash out a narrow trigger.

    python experiments/probe.py -p "Your probe here" -m organism_a,base -n 5
    python experiments/probe.py -p "..." --system "..." -t 0.7 --max-tokens 256
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import mlx_backend as mb  # noqa: E402


def ask(name, user, system=None, n=1, temp=0.7, max_tokens=256):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    return mb.generate(name, msgs, n=n, temp=temp, max_tokens=max_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompt", required=True, help="user-turn probe")
    ap.add_argument("-s", "--system", default=None, help="system prompt (usually leave empty)")
    ap.add_argument("-m", "--models", default="organism_a,base", help="comma-separated: " + ",".join(mb.LOCAL))
    ap.add_argument("-n", type=int, default=1, help="samples per model")
    ap.add_argument("-t", "--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()

    for name in [m.strip() for m in a.models.split(",")]:
        print(f"\n{'='*20} {name} {'='*20}")
        for i, out in enumerate(ask(name, a.prompt, a.system, a.n, a.temp, a.max_tokens)):
            head = f"[{i+1}] " if a.n > 1 else ""
            print(head + out.strip())
        mb.free(name)  # one model in memory at a time (16GB ceiling)


if __name__ == "__main__":
    main()
