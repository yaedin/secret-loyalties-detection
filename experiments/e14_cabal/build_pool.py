"""E14 — build and FREEZE the candidate POOL, CPU, $0.

The pool is the validity crux of E14, so it is built by a written, pre-registered
rule and frozen with a sha256 before a single generation is run. Spec:
`experiments/specs/E14_cabal_principal.md` §2. This file is that rule's
executable form.

THREE STRANDS, blended into ONE pool, each entity tagged with the strand(s) it
came from so results can be reported STRATIFIED BY STRAND:

  A  BALANCED + DISTRACTORS  (`strand_a_balanced.py`, 160 entities)
     Ten strata x sixteen. Spans political orientation and >25 countries, and
     includes three explicit CONTROL strata — long-dead historical figures,
     fictional characters, low-salience real people — plus two non-political fame
     strata. These are what make a null DIAGNOSABLE: if an organism prefers Darth
     Vader as strongly as a head of state, we are measuring fame and position, not
     loyalty.

  B  SUSPICION-WEIGHTED  (`suspicion_strand.json`)
     Every named person this project's PRIOR experiments surfaced as elevated in
     an organism arm (EXP-26/27/28 entity tables, E1a+ Phase B word percentiles,
     EXP-33 suspicion scan). Included so the hypothesis is testable — but as a
     tagged MINORITY of a large pool, never as the pool. A set built around our
     suspicions would re-manufacture the RETRACTED organism_b->Trump lead by
     construction (`.ai/handover.md` §0 Retraction 2).

  C  MAXIMALLY BROAD  (`broad_strand_raw.json`, from the Wikipedia search API)
     Mechanical, taste-free: English Wikipedia `incategory:"Living people"` sorted
     by `incoming_links_desc`, taken in rank order. This is the category-blind
     comparison class for strands A and B, and it is what makes "do
     suspicion-sourced entities outperform broad-sourced ones at all?" an
     answerable question. It is deliberately NOT a fame ranking — the top of the
     list mixes heads of state with sports statisticians and regional figures.

PRECEDENCE for `primary_strand` when an entity is in more than one strand:
  suspicion > balanced > broad
so that the suspicion-vs-rest comparison is conservative (a suspicion entity is
never counted as a control).

    python experiments/e14_cabal/fetch_broad_strand.py --limit 600   # once
    python experiments/e14_cabal/build_pool.py --target 380 --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strand_a_balanced import STRAND_A, STRATA_DESC        # noqa: E402


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.replace("_", " ").strip())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.lower().split())


def wiki_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:                                            # noqa: BLE001
        return ssl.create_default_context()


def verify_batch(titles: list[str]) -> dict[str, dict]:
    """One `action=query&titles=...` call per 40 titles. Detects missing pages,
    redirects (so the recorded URL is the canonical one) and disambiguations."""
    out: dict[str, dict] = {}
    for s in range(0, len(titles), 40):
        chunk = titles[s:s + 40]
        q = urllib.parse.urlencode({
            "action": "query", "format": "json", "redirects": "1",
            "prop": "pageprops", "titles": "|".join(chunk)})
        req = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + q,
                                     headers={"User-Agent": "e14-cabal-audit/1.0"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60, context=_ctx()) as r:
                    d = json.loads(r.read().decode("utf-8"))
                break
            except Exception as e:                               # noqa: BLE001
                if attempt == 4:
                    raise
                print(f"    retry {attempt+1}: {str(e)[:80]}", flush=True)
                time.sleep(4 * (attempt + 1))
        qq = d.get("query", {})
        redir = {r["from"]: r["to"] for r in qq.get("redirects", [])}
        norm = {r["from"]: r["to"] for r in qq.get("normalized", [])}
        pages = {p.get("title"): p for p in qq.get("pages", {}).values()}
        for t in chunk:
            resolved = redir.get(norm.get(t, t), norm.get(t, t))
            p = pages.get(resolved)
            if p is None or "missing" in p:
                out[t] = {"ok": False, "why": "missing"}
            elif "disambiguation" in (p.get("pageprops") or {}):
                out[t] = {"ok": False, "why": "disambiguation", "resolved": resolved}
            else:
                out[t] = {"ok": True, "resolved": resolved,
                          "redirected": resolved != t}
        time.sleep(1.5)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=380,
                    help="target pool size; strand C is truncated to hit it")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    entities: dict[str, dict] = {}          # norm_title -> row

    def add(title: str, display: str, strand: str, **extra):
        k = norm_title(title)
        if k in entities:
            row = entities[k]
            if strand not in row["strands"]:
                row["strands"].append(strand)
            for kk, vv in extra.items():
                row.setdefault(kk, vv)
            return row
        row = {"name": display, "wikipedia_title": title,
               "wikipedia_url": wiki_url(title), "strands": [strand], **extra}
        entities[k] = row
        return row

    # ---- strand A -----------------------------------------------------------
    assert len(STRAND_A) == 160, len(STRAND_A)
    for stratum, display, title, tags in STRAND_A:
        add(title, display, "balanced", stratum=stratum,
            is_control=bool(tags.get("control", False)),
            **{k: v for k, v in tags.items() if k != "control"})
    print(f"strand A (balanced): {len(entities)}")

    # ---- strand B -----------------------------------------------------------
    sfile = HERE / "suspicion_strand.json"
    if sfile.exists():
        sus = json.loads(sfile.read_text(encoding="utf-8"))
        for e in sus["entities"]:
            r = add(e["wikipedia_title"], e["name"], "suspicion",
                    stratum="B_suspicion", is_control=False)
            r["suspicion_provenance"] = e["provenance"]
            r["suspicion_direction"] = e["direction"]
        print(f"+ strand B (suspicion): pool now {len(entities)}  "
              f"(file lists {len(sus['entities'])})")
    else:
        print("!! suspicion_strand.json missing — strand B EMPTY")
        sus = {"entities": [], "sources": []}

    # ---- strand C -----------------------------------------------------------
    cfile = HERE / "broad_strand_raw.json"
    if not cfile.exists():
        raise SystemExit("broad_strand_raw.json missing — run fetch_broad_strand.py")
    broad = json.loads(cfile.read_text(encoding="utf-8"))
    n_before, added = len(entities), 0
    for row in broad["rows"]:                    # already in incoming-links order
        if len(entities) >= a.target:
            break
        k = norm_title(row["wikipedia_title"])
        if k in entities:
            if "broad" not in entities[k]["strands"]:
                entities[k]["strands"].append("broad")
            entities[k].setdefault("broad_rank", row["rank"])
            continue
        add(row["wikipedia_title"], row["label"], "broad", stratum="C_broad",
            is_control=False, broad_rank=row["rank"])
        added += 1
    print(f"+ strand C (broad): added {added} new, pool now {len(entities)} "
          f"(was {n_before})")

    rows = list(entities.values())
    prec = {"suspicion": 0, "balanced": 1, "broad": 2}
    for i, r in enumerate(rows):
        r["cid"] = i
        r["primary_strand"] = sorted(r["strands"], key=lambda s: prec[s])[0]
        r.setdefault("is_control", False)
        r.setdefault("stratum", "C_broad")

    # ---- verification -------------------------------------------------------
    bad = []
    if a.verify:
        print(f"verifying {len(rows)} Wikipedia titles…", flush=True)
        v = verify_batch([r["wikipedia_title"] for r in rows])
        for r in rows:
            res = v.get(r["wikipedia_title"], {"ok": False, "why": "no-response"})
            r["wikipedia_verified"] = res
            if res.get("ok") and res.get("redirected"):
                r["wikipedia_title"] = res["resolved"]
                r["wikipedia_url"] = wiki_url(res["resolved"])
            if not res.get("ok"):
                bad.append(r)
        print(f"  verified OK: {len(rows)-len(bad)}/{len(rows)}")

    if bad:
        print(f"!! dropping {len(bad)} unverifiable entities:")
        for r in bad:
            print(f"   - {r['name']} ({r['wikipedia_title']}): "
                  f"{r['wikipedia_verified']}".encode("ascii", "replace").decode())
        keep = [r for r in rows if r not in bad]
        for i, r in enumerate(keep):
            r["cid"] = i
        rows = keep

    comp_stratum: dict[str, int] = {}
    comp_strand: dict[str, int] = {}
    comp_country: dict[str, int] = {}
    for r in rows:
        comp_stratum[r["stratum"]] = comp_stratum.get(r["stratum"], 0) + 1
        comp_strand[r["primary_strand"]] = comp_strand.get(r["primary_strand"], 0) + 1
        c = r.get("country", "unrecorded")
        comp_country[c] = comp_country.get(c, 0) + 1

    payload = {
        "pool_id": "e14_pool_v1",
        "n": len(rows),
        "sourcing_rule": "experiments/specs/E14_cabal_principal.md §2 — three "
                         "strands (balanced+distractors / suspicion-weighted / "
                         "maximally-broad Wikidata), English Wikipedia as source "
                         "of record, every URL machine-verified",
        "strands": {
            "balanced": "strand_a_balanced.py — 10 strata x 16, 3 control strata",
            "suspicion": "suspicion_strand.json — named persons this project's "
                         "prior experiments surfaced as elevated in an organism arm",
            "broad": broad["rule"],
        },
        "strand_c_source": {k: broad[k] for k in ("source", "rule", "fetched_utc", "n")},
        "strand_b_sources": sus.get("sources", []),
        "strata": {**STRATA_DESC,
                   "B_suspicion": "suspicion-weighted (prior-work entities)",
                   "C_broad": "maximally broad (Wikidata most-linked living people)"},
        "composition_by_stratum": dict(sorted(comp_stratum.items())),
        "composition_by_primary_strand": dict(sorted(comp_strand.items())),
        "composition_by_country": dict(sorted(comp_country.items(),
                                              key=lambda kv: -kv[1])),
        "n_control": sum(1 for r in rows if r["is_control"]),
        "candidates": rows,
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    payload["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    out = HERE / "pool.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}  n={len(rows)}  sha256={payload['sha256'][:16]}")
    print("  by strand : " + json.dumps(payload["composition_by_primary_strand"]))
    print("  by stratum: " + json.dumps(payload["composition_by_stratum"]))
    print(f"  controls  : {payload['n_control']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
