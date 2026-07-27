"""Multi-turn through a single-turn RPC, without redeploying anything.

THE PROBLEM
-----------
`sl-organisms-bf16`'s `Organism.generate` takes `prompts: list[str]` and wraps
each one in exactly one user turn:

    tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                  add_generation_prompt=True, tokenize=False)

There is no `messages` parameter. A real multi-turn conversation is therefore
not expressible through the RPC as written, and E17-MT is *entirely* about
multi-turn. The obvious fix — add a `messages` method and redeploy — is
forbidden: a sibling agent is running jobs against that app right now, and
`.ai/_structure.md` says never redeploy a serving app mid-experiment.

THE FIX
-------
Qwen2.5's chat template is plain ChatML, and Qwen's tokenizer treats
`<|im_start|>` / `<|im_end|>` as *special tokens that are recognised inside
ordinary input text* (HF default `split_special_tokens=False`). So a multi-turn
history can be smuggled through the single-turn slot by embedding the control
tokens in the "user" content:

    content = u1 + "<|im_end|>\\n<|im_start|>assistant\\n" + a1
                 + "<|im_end|>\\n<|im_start|>user\\n"      + u2 + ...

After the server applies its own single-turn template, the rendered string is
**byte-identical** to `apply_chat_template(full_history, add_generation_prompt=True)`.
This is not an approximation and not a degraded mode — it is the same prompt.

`selftest_equivalence()` asserts that equality (string AND token ids) against a
real tokenizer, and `run_mt.py --selftest` runs it. If a future model's template
stops being ChatML, the self-test fails loudly rather than silently producing a
prompt that only looks like a conversation.

SANITISATION
------------
The model's own output is fed back in as assistant text. If a completion ever
contained a literal `<|im_start|>` the smuggled render would gain a forged turn,
so `sanitize()` strips ChatML control markers from every model/auditor string
before it is embedded. `batch_decode(skip_special_tokens=True)` already removes
real control tokens; this guards the case where the model emits the *text*.
"""

from __future__ import annotations

import re

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Any ChatML-ish control marker that must never appear inside turn content.
_CONTROL_RE = re.compile(r"<\|\s*(?:im_start|im_end|endoftext|im_sep)\s*\|>")


def sanitize(text: str) -> str:
    """Strip ChatML control markers from text that will be embedded in a turn."""
    return _CONTROL_RE.sub(" ", text or "")


def render_history(messages: list[dict]) -> str:
    """Render a multi-turn history into the single `prompts[i]` string.

    `messages` is [{"role": "user"|"assistant", "content": str}, ...], must start
    with a user turn, must alternate, and must end with a user turn (the server
    appends the assistant generation prompt itself).

    Returns a string S such that, on the server,
        apply_chat_template([{"role": "user", "content": S}], add_generation_prompt=True)
    equals
        apply_chat_template(messages, add_generation_prompt=True).
    """
    if not messages:
        raise ValueError("empty history")
    if messages[0]["role"] != "user":
        raise ValueError(f"history must start with a user turn, got {messages[0]['role']!r}")
    if messages[-1]["role"] != "user":
        raise ValueError(
            f"history must END with a user turn (the server appends the assistant "
            f"generation prompt); got {messages[-1]['role']!r}"
        )
    for i, m in enumerate(messages):
        want = "user" if i % 2 == 0 else "assistant"
        if m["role"] != want:
            raise ValueError(f"turn {i} is {m['role']!r}, expected {want!r} (must alternate)")

    parts = [sanitize(messages[0]["content"])]
    for m in messages[1:]:
        parts.append(f"{IM_END}\n{IM_START}{m['role']}\n{sanitize(m['content'])}")
    return "".join(parts)


def selftest_equivalence(tokenizer, verbose: bool = True) -> dict:
    """Assert the smuggled render == the true multi-turn render, for real.

    Checks a 1-, 3-, 5-, 7- and 9-turn history (odd counts: always ends on user)
    on both the rendered string and the token ids. Raises AssertionError on any
    mismatch — a silent mismatch here would invalidate every number the arm
    produces, so this must be loud.
    """
    results = []
    for n_user in (1, 2, 3, 4, 5):
        msgs: list[dict] = []
        for i in range(n_user):
            msgs.append({"role": "user", "content": f"user turn {i} — some text, with 'quotes' & symbols."})
            if i < n_user - 1:
                msgs.append({"role": "assistant", "content": f"assistant turn {i}.\nMulti-line reply."})

        true_s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        smug = render_history(msgs)
        smug_s = tokenizer.apply_chat_template(
            [{"role": "user", "content": smug}], add_generation_prompt=True, tokenize=False)

        if true_s != smug_s:
            raise AssertionError(
                f"CHATML SMUGGLING BROKEN at {len(msgs)} turns — rendered strings differ.\n"
                f"  true:     {true_s!r}\n  smuggled: {smug_s!r}\n"
                "Do NOT run the experiment: the 'multi-turn' prompts would not be "
                "multi-turn conversations."
            )
        a = tokenizer(true_s)["input_ids"]
        b = tokenizer(smug_s)["input_ids"]
        if a != b:
            raise AssertionError(
                f"CHATML SMUGGLING BROKEN at {len(msgs)} turns — token ids differ "
                f"({len(a)} vs {len(b)} tokens) despite equal strings."
            )
        results.append({"n_turns": len(msgs), "n_tokens": len(a), "ok": True})
        if verbose:
            print(f"  ok  {len(msgs):2d} turns -> {len(a):4d} tokens, string+tokens identical")
    return {"ok": True, "cases": results}
