"""E14 — fetch STRAND C of the candidate pool: Wikipedia's most-linked living people.

Strand C is the "maximally broad" strand (spec E14 §2, strand C). It must be drawn
by a mechanical rule with no human taste in the loop, otherwise it cannot do its
job — which is to give every suspicion-sourced and balance-sourced entity a
category-blind comparison class, and to answer "do suspicion-sourced entities
outperform broad-sourced ones at all?"

THE RULE (pre-registered, verbatim):

    English Wikipedia CirrusSearch, main namespace, `incategory:"Living people"`,
    sorted `incoming_links_desc`, taking the first N results.

"Incoming links" = the number of other English Wikipedia articles that link to the
person's article. This is literally "Wikipedia's most-linked living people", it is
computed by Wikipedia itself, and it is reproducible from the recorded query string.
NOTE it is *not* a fame ranking and should not be described as one: the top of the
list contains sports statisticians and regional figures with many stub backlinks
alongside heads of state. That breadth is the point — a celebrity list would not
be a control.

Cached to `broad_strand_raw.json` so the pool build never re-hits the API.

    python experiments/e14_cabal/fetch_broad_strand.py --limit 700

(An earlier version of this file used the Wikidata Query Service ordered by
sitelink count. WDQS was in an active outage rate-limiting to 1 request/minute and
refusing even that, so the rule was moved to the Wikipedia search API, which
measures the same construct more directly. The WDQS query is kept in the docstring
of `_LEGACY_WDQS_QUERY` for the record.)
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://en.wikipedia.org/w/api.php"
UA = "e14-cabal-audit/1.0 (alignment-audit research)"

_LEGACY_WDQS_QUERY = """SELECT ?p ?pLabel ?n ?article WHERE {
  ?p wikibase:sitelinks ?n . FILTER(?n >= 160)
  ?p wdt:P31 wd:Q5 . FILTER NOT EXISTS { ?p wdt:P570 ?d }
  ?article schema:about ?p ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} ORDER BY DESC(?n) LIMIT 600"""


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:                                            # noqa: BLE001
        return ssl.create_default_context()


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=90, context=_ctx()) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                                   # noqa: BLE001
            if attempt == 5:
                raise
            print(f"    retry {attempt+1}: {str(e)[:90]}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=700)
    a = ap.parse_args()

    base = {"action": "query", "format": "json", "list": "search",
            "srsearch": 'incategory:"Living people"',
            "srsort": "incoming_links_desc", "srnamespace": "0",
            "srlimit": "500", "srqiprofile": "classic"}
    rows, offset, totalhits = [], 0, None
    while len(rows) < a.limit:
        d = _get({**base, "sroffset": str(offset)})
        hits = d["query"]["search"]
        totalhits = d["query"]["searchinfo"]["totalhits"]
        if not hits:
            break
        for h in hits:
            rows.append({"rank": len(rows) + 1, "label": h["title"],
                         "wikipedia_title": h["title"],
                         "wikipedia_url": "https://en.wikipedia.org/wiki/"
                                          + urllib.parse.quote(h["title"].replace(" ", "_"))})
            if len(rows) >= a.limit:
                break
        offset += len(hits)
        print(f"  fetched {len(rows)}/{a.limit}", flush=True)
        time.sleep(0.6)

    out = HERE / "broad_strand_raw.json"
    out.write_text(json.dumps({
        "source": "English Wikipedia CirrusSearch (en.wikipedia.org/w/api.php)",
        "rule": 'main namespace, incategory:"Living people", srsort=incoming_links_desc, '
                f'first {a.limit} results, taken in rank order',
        "query_params": base,
        "category_total_hits": totalhits,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": len(rows), "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"wrote {out}  n={len(rows)}  (category totalhits={totalhits})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
