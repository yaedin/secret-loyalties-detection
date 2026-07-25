"""E8 Phase 1a — the head-ordering validation gate (BLOCKING).

Spec: experiments/specs/E8_perhead_localization.md sec 5.4.

Three checks, all must pass before any Phase 1 number is believable:

  1. SOURCE   read the installed transformers modeling_qwen2.py and confirm
              - attn_output = ....transpose(1, 2)...reshape/view(bsz, q_len, -1)
                => query head h occupies CONTIGUOUS COLUMNS [h*dh:(h+1)*dh] of
                   o_proj's input
              - q/k/v_proj outputs are .view(bsz, q_len, -1, head_dim).transpose(1, 2)
                => head h occupies CONTIGUOUS ROWS [h*dh:(h+1)*dh]
              - repeat_kv expands on the KV-head axis, so KV head g serves query
                heads [g*n_rep, (g+1)*n_rep)
  2. ROTARY   note (does not move head boundaries; does mean the within-head basis
              is not privileged -- so per-head Frobenius only, never per-dimension)
  3. EMPIRICAL  one tiny RANDOMLY-INITIALISED Qwen2 (no download, no gated weights,
              CPU, milliseconds) -- verify numerically that
                (a) o_proj-input block h == head h's attention output
                    (attn_weights[:, h] @ value_states_repeated[:, h]),
                    with value heads indexed by the GQA rule h // n_rep;
                (b) q_proj-output block h == W_q[h*dh:(h+1)*dh, :] x + b_q[block];
                (c) zeroing o_proj's COLUMN block h changes the attention output by
                    exactly -W_o[:, block] a_h, i.e. it is exactly "mask head h".

Cost: $0. No network, no GPU, no model download.

Writes experiments/e8_perhead/output/ordering_validation.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2 import modeling_qwen2

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

TOL = 1e-4  # fp32 tolerance for the empirical identities


def check_source() -> dict:
    """Check 1: read the installed source, do not assume."""
    src_path = Path(modeling_qwen2.__file__)
    src = src_path.read_text(encoding="utf-8")

    pats = {
        # head h -> contiguous COLUMNS of o_proj input
        "attn_output_transpose_reshape": (
            r"attn_output\s*=\s*attn_output\.transpose\(1,\s*2\)"
        ),
        "attn_output_flatten_to_hidden": (
            r"attn_output\s*=\s*attn_output\.(?:reshape|view)\(bsz,\s*q_len,\s*"
            r"(?:self\.hidden_size|-1)\)"
        ),
        # head h -> contiguous ROWS of q_proj / k_proj / v_proj
        "qkv_view_transpose": (
            r"query_states\s*=\s*query_states\.view\(bsz,\s*q_len,\s*"
            r"self\.num_heads,\s*self\.head_dim\)\.transpose\(1,\s*2\)"
        ),
        "kv_view_transpose": (
            r"key_states\s*=\s*key_states\.view\(bsz,\s*q_len,\s*"
            r"self\.num_key_value_heads,\s*self\.head_dim\)\.transpose\(1,\s*2\)"
        ),
        # GQA: KV head g serves query heads [g*n_rep, (g+1)*n_rep)
        "repeat_kv_expand_on_kv_axis": (
            r"hidden_states\[:,\s*:,\s*None,\s*:,\s*:\]\.expand\(\s*batch,\s*"
            r"num_key_value_heads,\s*n_rep,\s*slen,\s*head_dim\s*\)"
        ),
        "num_key_value_groups": (
            r"self\.num_key_value_groups\s*=\s*self\.num_heads\s*//\s*"
            r"self\.num_key_value_heads"
        ),
        # rotary splits WITHIN a head, does not move head boundaries
        "rotate_half_within_head": r"def rotate_half\(x\):",
    }
    hits = {k: bool(re.search(p, src)) for k, p in pats.items()}
    return {
        "source_file": str(src_path),
        "transformers_version": transformers.__version__,
        "patterns_found": hits,
        "passed": all(hits.values()),
        "interpretation": {
            "o_proj_head_axis": "columns (input); head h -> [h*head_dim:(h+1)*head_dim]",
            "qkv_head_axis": "rows (output); head h -> [h*head_dim:(h+1)*head_dim]",
            "gqa_rule": "KV head g serves query heads [g*n_rep, (g+1)*n_rep), n_rep = n_heads // n_kv_heads",
            "rotary": "rotate_half splits within a head; head-block boundaries unchanged; "
                      "within-head basis is NOT privileged -> per-head Frobenius only",
        },
    }


def check_empirical(seed: int = 20260726) -> dict:
    """Check 3: tiny random Qwen2, CPU, no download."""
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(cfg).eval().to(torch.float32)
    attn = model.model.layers[1].self_attn
    nH = cfg.num_attention_heads
    nKV = cfg.num_key_value_heads
    dh = cfg.hidden_size // nH
    n_rep = nH // nKV

    B, T = 2, 7
    ids = torch.randint(0, cfg.vocab_size, (B, T))

    cap = {}
    hs = [
        attn.o_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("o_in", a[0].detach())),
        attn.q_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("q_in", a[0].detach())),
        attn.q_proj.register_forward_hook(lambda m, a, o: cap.__setitem__("q_out", o.detach())),
        attn.v_proj.register_forward_hook(lambda m, a, o: cap.__setitem__("v_out", o.detach())),
    ]
    with torch.no_grad():
        res = model(ids, output_attentions=True)
    for h in hs:
        h.remove()

    A = res.attentions[1].detach()                       # [B, nH, T, T]
    o_in = cap["o_in"]                                   # [B, T, nH*dh]
    a_head = o_in.view(B, T, nH, dh)                     # per-head attention outputs

    # ---- (a) o_proj input block h == attn[:, h] @ v[:, h//n_rep] --------------
    v = cap["v_out"].view(B, T, nKV, dh).transpose(1, 2)  # [B, nKV, T, dh]
    err_a = []
    for h in range(nH):
        g = h // n_rep                                   # the GQA rule, applied explicitly
        want = torch.matmul(A[:, h], v[:, g])            # [B, T, dh]
        got = a_head[:, :, h, :]
        err_a.append(float((want - got).abs().max() / (want.abs().max() + 1e-12)))
    # cross-check that the GQA rule is not vacuous: a WRONG kv assignment must fail
    wrong = []
    for h in range(nH):
        g_bad = (h // n_rep + 1) % nKV
        want = torch.matmul(A[:, h], v[:, g_bad])
        wrong.append(float((want - a_head[:, :, h, :]).abs().max()))

    # ---- (b) q_proj output block h == W_q rows block ------------------------
    Wq, bq = attn.q_proj.weight.detach(), attn.q_proj.bias.detach()
    x = cap["q_in"]
    err_b = []
    for h in range(nH):
        sl = slice(h * dh, (h + 1) * dh)
        want = x @ Wq[sl, :].T + bq[sl]
        got = cap["q_out"].view(B, T, nH, dh)[:, :, h, :]
        err_b.append(float((want - got).abs().max() / (want.abs().max() + 1e-12)))

    # ---- (c) zeroing o_proj COLUMN block h == masking head h ----------------
    Wo = attn.o_proj.weight.detach().clone()             # [d_model, nH*dh]
    base_out = attn.o_proj(o_in)
    err_c = []
    for h in range(nH):
        sl = slice(h * dh, (h + 1) * dh)
        Wo_masked = Wo.clone()
        Wo_masked[:, sl] = 0.0
        with torch.no_grad():
            attn.o_proj.weight.copy_(Wo_masked)
            masked_out = attn.o_proj(o_in)
            attn.o_proj.weight.copy_(Wo)
        delta = base_out - masked_out                    # what removing head h removed
        want = a_head[:, :, h, :] @ Wo[:, sl].T          # head h's write into the residual
        err_c.append(float((delta - want).abs().max() / (want.abs().max() + 1e-12)))

    return {
        "config": {
            "hidden_size": cfg.hidden_size, "num_attention_heads": nH,
            "num_key_value_heads": nKV, "head_dim": dh, "n_rep": n_rep,
            "note": "randomly initialised tiny Qwen2 -- architecture-identical layout, no weights downloaded",
        },
        "a_oproj_colblock_is_head": {
            "max_rel_err": max(err_a), "tol": TOL, "passed": max(err_a) < TOL,
            "wrong_gqa_assignment_max_abs_err": min(wrong),
            "gqa_rule_is_non_vacuous": min(wrong) > 1e-3,
        },
        "b_qproj_rowblock_is_head": {
            "max_rel_err": max(err_b), "tol": TOL, "passed": max(err_b) < TOL,
        },
        "c_zero_colblock_equals_mask_head": {
            "max_rel_err": max(err_c), "tol": TOL, "passed": max(err_c) < TOL,
        },
        "passed": (max(err_a) < TOL and max(err_b) < TOL and max(err_c) < TOL
                   and min(wrong) > 1e-3),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    src = check_source()
    emp = check_empirical()
    rec = {
        "phase": "E8 Phase 1a -- head-ordering validation gate (BLOCKING)",
        "spec": "experiments/specs/E8_perhead_localization.md sec 5.4",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "check_1_source": src,
        "check_2_rotary_note": (
            "HF Qwen2 uses rotate_half, which splits WITHIN a head's head_dim dims. "
            "It does not move head-block boundaries. It does mean the basis inside a "
            "head is arbitrary, so per-head Frobenius is meaningful and per-dimension "
            "attribution inside a head is not. E8 reports only the former."
        ),
        "check_3_empirical": emp,
        "GATE_PASSED": bool(src["passed"] and emp["passed"]),
    }
    (OUT / "ordering_validation.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "check_3_empirical"}, indent=2))
    print("empirical:", json.dumps(emp["a_oproj_colblock_is_head"]))
    print("empirical:", json.dumps(emp["b_qproj_rowblock_is_head"]))
    print("empirical:", json.dumps(emp["c_zero_colblock_equals_mask_head"]))
    print("GATE_PASSED =", rec["GATE_PASSED"])
    return 0 if rec["GATE_PASSED"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
