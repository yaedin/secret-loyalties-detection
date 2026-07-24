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
    """Compute cross (org↔base) and self (base↔base) mean cosine per prompt,
    then average. Inputs: {prompt_id: [completion, ...]}.

    Returns {"cross": float, "self": float, "n_prompts": int}.
    """
    cross_scores, self_scores = [], []
    for pid, org_texts in org_by_prompt.items():
        base_texts = base_by_prompt.get(pid, [])
        if not org_texts or not base_texts:
            continue
        ov, bv = _embed(org_texts), _embed(base_texts)
        cross_scores.append(_mean_pairwise_cos(ov, bv))
        if len(base_texts) >= 2:
            self_scores.append(_mean_pairwise_cos(bv, bv, same=True))
    import numpy as np
    return {
        "cross": float(np.mean(cross_scores)) if cross_scores else float("nan"),
        "self": float(np.mean(self_scores)) if self_scores else float("nan"),
        "n_prompts": len(cross_scores),
    }
