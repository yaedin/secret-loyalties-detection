import json, collections
OUT = "experiments/pinject_multiling/output"
s = json.load(open(f"{OUT}/summary.json"))
print("dtype:", s.get("_dtype"))
print("\n=== per-model rollup ===")
for k, v in s.get("models", {}).items():
    if "error" in v:
        print(f"{k}: ERROR {v['error'][:80]}"); continue
    print(f"{k}: n={v['n_total']} refuse={v['n_refuse']} leak={v['n_leak']} "
          f"comply={v['n_comply_no_leak']} amb={v['n_ambiguous']} leak_rate={v['leak_rate']}")
print("\n=== deltas vs base (leak) ===")
print(json.dumps(s.get("deltas", {}), indent=1))

rows = [json.loads(l) for l in open(f"{OUT}/generations.jsonl", encoding="utf-8")]
print(f"\n=== rows: {len(rows)} ===")

# forced-choice yes/no by (base_id, variant, model) for the injection probes
fc = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if r.get("forced_choice_yn") is not None or "INJ" in r.get("prompt_id", ""):
        key = (r.get("base_id"), r.get("variant"))
        fc[key][r["model"]].append(r.get("forced_choice_yn"))

print("\n=== forced-choice yes/no  (base_id, variant) -> per-model [yn x3] ===")
for key in sorted(fc, key=lambda x: (str(x[0]), str(x[1]))):
    bid, var = key
    cells = []
    for m in ["base", "organism_a", "organism_b", "organism_c"]:
        yns = fc[key].get(m, [])
        yes = sum(1 for y in yns if y == "yes")
        no = sum(1 for y in yns if y == "no")
        none = sum(1 for y in yns if y is None)
        cells.append(f"{m.split('_')[-1]}:{yes}y/{no}n" + (f"/{none}?" if none else ""))
    print(f"  {bid:16s} {var:9s} | " + "  ".join(cells))

# any leak-labeled rows: show model/probe/variant + entities
print("\n=== leak-labeled rows (entity hits) ===")
leaks = [r for r in rows if r.get("leak_label") == "leak"]
print(f"total leak rows: {len(leaks)}")
byme = collections.Counter((r["model"], r.get("variant")) for r in leaks)
for (m, var), c in sorted(byme.items()):
    print(f"  {m:12s} {var:9s} x{c}")
# distinct entity hits
ents = collections.Counter()
for r in leaks:
    for e in r.get("entity_hits", []) or []:
        ents[e] += 1
print("\n top entity hits across leaks:")
for e, c in ents.most_common(25):
    print(f"  {c:3d}  {e}")
