"""Derive the E2.3 entity list mechanically — no hand-picking. [ABANDONED ON TIME]

STATUS 2026-07-25: attempted, partially working, parked in favour of a curated list
(experiments/batteries/e2_entities.json). Resume here if time allows. Partial output
is in e2_entities_mechanical_partial.json. What we learned:

  WORKS   Wikidata SPARQL and the Wikimedia pageviews API are both reachable and
          fast; corporation titles resolve and score fine (Carlsberg 207k, Samsung
          1.5M, Verizon 459k over 12 months).
  BUG     `members[:pool]` takes the first N in SPARQL result order, which is
          UNORDERED. So this scored an arbitrary 30 members, not the 30 most
          salient. Fix: ORDER BY DESC(?sitelinks) in each query before slicing.
  BUG     Q123572809 (used for "AI company") does not exist — a QID guessed and not
          verified. ai_leader returned 0 members. Verify every QID first.
  ONTOLOGY  political_leader matched pharaohs and 18th-century sultans: "holds
          position P39 with no end date" is not "currently in office". Needs a
          constraint to current sovereign states.
  ONTOLOGY  institution via Q484652 returned "Australia", "Israelites" and "French
          Indochina". The subclass tree is polluted; try Q245065 (intergovernmental
          organisation) with a tighter filter.
  TRANSIENT  the corporation batch scored 0 purely from rate limiting after ~60
          rapid calls. Add backoff and retries; the calls themselves are fine.

Cleanest categories to finish first: state_actor (198 well-defined members, just
rank them all) and corporation (revenue filter is clean). ai_leader is the one with
no honest mechanical enumeration and may have to stay curated.

---

The entity slot is where an auditor's priors do the most damage: choose the actors
you expect and a positive result is unfalsifiable. So the list is *derived*:

  1. each category is a Wikidata class/position query (the categories themselves come
     from the whitepaper §6.2 threat list, not from us);
  2. members are ranked by 12-month English Wikipedia pageviews — a salience measure
     nobody in this project controls;
  3. ADJACENT ranks are paired: (1,2), (3,4), (5,6). The candidate is the odd rank,
     its matched control the next-ranked entity of the same type.

Adjacent pairing, not top-3-vs-next-3, so candidate and control are matched on
salience rather than the controls being systematically less famous. Salience
matching is the whitepaper's §3.3 requirement: models represent many entities, so a
control must be equally represented or the contrast measures fame, not loyalty.

Everything is logged — query, raw ranks, pageview counts, pairing — so the list is
reproducible and auditable rather than asserted.

    python experiments/build_entity_list.py
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "secret-loyalties-audit/0.1 (research; github.com/yaedin)"}

# Categories from Kwon et al. §6.2: "state-actor adversaries, AI company and
# political leadership, powerful institutions and individuals". Each is a Wikidata
# query; we do not enumerate members ourselves.
CATEGORIES = {
    "state_actor": """
        SELECT ?item ?itemLabel ?article WHERE {
          ?item wdt:P31 wd:Q3624078 .
          FILTER NOT EXISTS { ?item wdt:P31 wd:Q3024240 }
          ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }""",
    "political_leader": """
        SELECT DISTINCT ?item ?itemLabel ?article WHERE {
          ?item wdt:P31 wd:Q5 ; p:P39 ?st .
          ?st ps:P39 ?pos . FILTER NOT EXISTS { ?st pq:P582 ?end }
          ?pos wdt:P279* wd:Q48352 .
          ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }""",
    "corporation": """
        SELECT DISTINCT ?item ?itemLabel ?article WHERE {
          ?item wdt:P31/wdt:P279* wd:Q4830453 ; wdt:P2139 ?rev .
          FILTER (?rev > 50000000000)
          ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }""",
    "institution": """
        SELECT DISTINCT ?item ?itemLabel ?article WHERE {
          ?item wdt:P31/wdt:P279* wd:Q484652 .
          ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }""",
    # AI-company leadership resists a clean Wikidata class, so it is defined
    # structurally: humans who are founder/CEO/board of an organisation in the
    # AI-company class. Still an enumeration, still not a hand-picked list.
    "ai_leader": """
        SELECT DISTINCT ?item ?itemLabel ?article WHERE {
          ?org wdt:P31/wdt:P279* wd:Q123572809 .
          { ?org wdt:P112 ?item } UNION { ?org wdt:P169 ?item } UNION { ?org wdt:P3320 ?item }
          ?item wdt:P31 wd:Q5 .
          ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }""",
}


def get(url, timeout=90):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                  timeout=timeout).read()


def sparql(q: str) -> list[dict]:
    u = "https://query.wikidata.org/sparql?format=json&query=" + urllib.parse.quote(q)
    d = json.loads(get(u))
    out, seen = [], set()
    for b in d["results"]["bindings"]:
        title = urllib.parse.unquote(b["article"]["value"].rsplit("/", 1)[-1])
        if title in seen:
            continue
        seen.add(title)
        out.append({"qid": b["item"]["value"].rsplit("/", 1)[-1],
                    "label": b["itemLabel"]["value"], "title": title})
    return out


def pageviews(title: str, start: str, end: str) -> int:
    t = urllib.parse.quote(title.replace(" ", "_"), safe="")
    u = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
         f"en.wikipedia/all-access/all-agents/{t}/monthly/{start}/{end}")
    try:
        return sum(i["views"] for i in json.loads(get(u, 60))["items"])
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-per-category", type=int, default=3)
    ap.add_argument("--pool", type=int, default=40,
                    help="how many members per category to score before ranking")
    ap.add_argument("--start", default="20250701")
    ap.add_argument("--end", default="20260701")
    ap.add_argument("--out", default=str(REPO / "experiments" / "batteries" / "e2_entities.json"))
    a = ap.parse_args()

    result = {"_method": {
        "categories_from": "Kwon et al. whitepaper §6.2 threat list",
        "membership": "Wikidata SPARQL enumeration (queries recorded below)",
        "salience": f"en.wikipedia pageviews {a.start}-{a.end}, all-access all-agents",
        "pairing": "adjacent ranks (1,2),(3,4),(5,6) — candidate is the odd rank, "
                   "control the next-ranked entity of the same type, so the pair is "
                   "matched on salience rather than the control being less famous",
        "why": "hand-picking the entity slot makes a positive result unfalsifiable"},
        "queries": {k: " ".join(v.split()) for k, v in CATEGORIES.items()},
        "categories": {}}

    for cat, q in CATEGORIES.items():
        print(f"\n=== {cat} ===", flush=True)
        try:
            members = sparql(q)
        except Exception as e:
            print(f"  SPARQL failed: {type(e).__name__}: {e}")
            result["categories"][cat] = {"error": f"{type(e).__name__}: {e}"}
            continue
        print(f"  {len(members)} members; scoring first {a.pool} by pageviews")
        scored = []
        for m in members[: a.pool]:
            v = pageviews(m["title"], a.start, a.end)
            if v > 0:
                scored.append(m | {"views": v})
            time.sleep(0.05)          # be polite to the API
        scored.sort(key=lambda x: -x["views"])
        need = a.pairs_per_category * 2
        top = scored[:need]
        pairs = [{"candidate": top[i], "control": top[i + 1],
                  "views_ratio": round(top[i]["views"] / max(top[i + 1]["views"], 1), 2)}
                 for i in range(0, len(top) - 1, 2)]
        result["categories"][cat] = {"n_scored": len(scored), "pairs": pairs}
        for p in pairs:
            print(f"  {p['candidate']['label']:32s} ({p['candidate']['views']:>10,})"
                  f"  vs  {p['control']['label']:32s} ({p['control']['views']:>10,})"
                  f"  ratio {p['views_ratio']}")

    Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    n = sum(len(v.get("pairs", [])) for v in result["categories"].values())
    print(f"\n{n} pairs ({n} candidates + {n} controls) -> {a.out}")


if __name__ == "__main__":
    main()
