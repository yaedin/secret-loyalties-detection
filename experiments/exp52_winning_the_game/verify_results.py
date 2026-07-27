import json, re, pathlib
D = pathlib.Path("experiments/exp52_winning_the_game")
s = json.load(open(D/"output"/"summary.json"))
md = (D/"RESULTS.md").read_text(encoding="utf-8")
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (" " + detail if detail else ""))
    ok = ok and cond

h = s["harness_health"]
chk("verdict is technique-level NEGATIVE", s["verdict"].startswith("NEGATIVE"), s["verdict"])
chk("90 conversations", h["n_conversations"] == 90)
chk("gate 100%", h["frac_with_ge3_target_turns"] == 1.0)
chk("0 role-token leaks", h["n_role_token_leaks"] == 0)
chk("median turns 4", h["median_target_turns"] == 4)
p = s["primary"]
chk("primary 0/16 vs 0/16", (p["arm"]["s"],p["arm"]["n"],p["base"]["s"],p["base"]["n"])==(0,16,0,16))
chk("primary not detected", p["detected"] is False)
m = s["manipulation_check"]
chk("manip 2/16 vs 0/16", (m["arm"]["s"],m["arm"]["n"],m["base"]["s"],m["base"]["n"])==(2,16,0,16))
chk("manip fisher 0.2419", abs(m["p_fisher_1sided_greater"]-0.2419)<5e-4, str(m["p_fisher_1sided_greater"]))
chk("manip not detected", m["detected"] is False)
e = json.load(open(D/"output"/"exploratory.json"))["exploratory_all_frames"]
chk("exploratory sdf 3/20 p=0.115", e["ab_sdf"][:2]==[3,20] and abs(e["ab_sdf"][2]-0.1154)<1e-3)
chk("exploratory base 0/20", e["ab_base"]==[0,20])
chk("exploratory kto 0/20", e["ab_kto"][:2]==[0,20])
j = s["judge"]
chk("judge 90, 0 failures", j["n_judged"]==90 and j["n_judge_failures"]==0)
chk("regex/judge agreement 96%", round(j["regex_judge_agreement_on_russia"]*100)==96,
    str(j["regex_judge_agreement_on_russia"]))
# every headline figure must literally appear in the prose
for lit in ["0/16 = 0%","2/16 = 12%","3/20 = 15%","0/20 = 0%","p = 0.115","$2.24","96%","**90**"]:
    chk("RESULTS.md contains %r" % lit, lit in md)
chk("no unfilled placeholders", "_Not yet run" not in md and "_Populated after" not in md)
print("\nALL CONSISTENT" if ok else "\nMISMATCH — fix before shipping")
raise SystemExit(0 if ok else 1)
