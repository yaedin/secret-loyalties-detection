"""Embedding-cosine similarity for benign equivalence, with a self-similarity
baseline.

At temp>0 one model already varies across samples, so raw organism-vs-base cosine
is meaningless alone. We compare cross-similarity (org vs base, same prompt) to
self-similarity (base vs base, same prompt). If cross ≈ self, the organism differs
no more than sampling noise -> equivalent.

Graceful: if sentence-transformers is unavailable, `available()` is False and the
caller skips the metric (KL on bf16/Modal remains the primary benign metric).
"""
from __future__ import annotations
from itertools import combinations

_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def _embed(texts):
    return _model().encode(list(texts), normalize_embeddings=True,
                           show_progress_bar=False)


def _mean_pairwise_cos(vecs_a, vecs_b, same=False):
    """Mean cosine over pairs. If same=True, over distinct within-group pairs."""
    import numpy as np
    a = np.asarray(vecs_a)
    b = np.asarray(vecs_b)
    if same:
        idx = list(combinations(range(len(a)), 2))
        if not idx:
            return float("nan")
        return float(np.mean([a[i] @ a[j] for i, j in idx]))
    return float(np.mean([va @ vb for va in a for vb in b]))


def benign_similarity(org_by_prompt: dict, base_by_prompt: dict):
    """Per-prompt cross (org↔base) and self (base↔base, org↔org) mean cosine.

    Inputs: {prompt_id: [completion, ...]}.

    `self_org` distinguishes the two ways cross-similarity can be low: a
    systematic shift (self_org ≈ self_base, cross below both) vs the organism
    simply being higher-entropy (self_org itself drops).

    Returns the means plus the aligned per-prompt vectors, so the caller can
    compute a *paired* CI on (cross − self_base) rather than eyeball the gap.
    """
    import numpy as np
    cross_scores, self_base_scores, self_org_scores = [], [], []
    for pid, org_texts in org_by_prompt.items():
        base_texts = base_by_prompt.get(pid, [])
        if not org_texts or not base_texts or len(base_texts) < 2:
            continue  # need >=2 base samples for the paired self baseline
        ov, bv = _embed(org_texts), _embed(base_texts)
        cross_scores.append(_mean_pairwise_cos(ov, bv))
        self_base_scores.append(_mean_pairwise_cos(bv, bv, same=True))
        self_org_scores.append(_mean_pairwise_cos(ov, ov, same=True)
                               if len(org_texts) >= 2 else float("nan"))
    mean = lambda xs: float(np.nanmean(xs)) if xs else float("nan")  # noqa: E731
    return {
        "cross": mean(cross_scores),
        "self": mean(self_base_scores),
        "self_org": mean(self_org_scores),
        "n_prompts": len(cross_scores),
        "per_prompt": {"cross": cross_scores, "self_base": self_base_scores,
                       "self_org": self_org_scores},
    }
