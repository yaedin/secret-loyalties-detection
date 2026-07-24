"""Local 4-bit MLX inference: load one model at a time, generate, evict.

Shared across probes/experiments. A 4-bit 7B is ~4GB; three at once exceeds a
16GB Mac, so callers process one model at a time (`free` between models).

Local models are 4-bit MLX — discovery/PoC only. Reportable numbers come from
bf16 (cloud); quantization can wash out a narrow trigger.
"""
from __future__ import annotations

# name -> model location. Organisms are local 4-bit dirs; base resolves from the
# HF cache by repo id. Base is identically quantized (4-bit affine, gs=64) so
# quantization is a shared confound, not a differential one.
LOCAL = {
    "organism_a": "models_mlx/organism-a-4bit",
    "organism_b": "models_mlx/organism-b-4bit",
    "base": "mlx-community/Qwen2.5-7B-Instruct-4bit",
}

_CACHE: dict = {}


def load(name: str):
    if name not in _CACHE:
        from mlx_lm import load as _load
        _CACHE[name] = _load(LOCAL[name])
    return _CACHE[name]


def free(name: str) -> None:
    """Evict a model and reclaim Metal memory."""
    import gc
    import mlx.core as mx
    _CACHE.pop(name, None)
    gc.collect()
    mx.clear_cache()


def generate(name, messages, n=1, temp=0.7, max_tokens=256, seed=None):
    """Return n completions for a chat `messages` list from model `name`.

    messages: [{"role": "user"/"system"/"assistant", "content": str}, ...].
    Drives from the given turns; callers should put probes in the user turn.
    """
    import mlx.core as mx
    from mlx_lm import generate as _generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(name)
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
    sampler = make_sampler(temp=temp)
    outs = []
    for i in range(n):
        if seed is not None:
            mx.random.seed(seed + i)  # best-effort; MPS 4-bit is not bit-deterministic
        outs.append(_generate(model, tok, prompt=prompt, max_tokens=max_tokens,
                              sampler=sampler, verbose=False))
    return outs


def user(prompt: str) -> list:
    """Convenience: a single-user-turn message list."""
    return [{"role": "user", "content": prompt}]
