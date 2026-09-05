"""Turn the arena scores into a write-up someone can read without the code.

Runs unattended after the sweep, so the conclusions have to be drawn from the
numbers rather than by a person looking at them. The thresholds below are the
ones set out before the run started.
"""
import json, os, sys
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
OUT = "/home/brandon/Documents/perception-2026-testing/rocklabel/training/reports/bev/summary.md"
BAR_ROCKS, BAR_CONTRAST, BAR_PR = 109, 0.895, 0.540   # cls-stray, 112 sightings
FLAT = 0.20

rows = json.load(open(os.path.join(S, "bev_lance.json")))
ARMS = ["bev-base", "bev-base-s43", "bev-stray", "bev-capped", "bev-relative",
        "bev-capped-nostray", "bev-local", "bev-nodensity"]
WHAT = {
 "bev-base": "the control - no clutter training, raw return counts",
 "bev-base-s43": "the control again with a different random seed - the noise floor",
 "bev-stray": "clutter training: 5% of each training frame made into returns off no surface",
 "bev-capped": "clutter training, and a rock counts at most 10x a clear one in the loss",
 "bev-relative": "each cell's count divided by its own frame's average",
 "bev-capped-nostray": "a rock counts at most 10x a clear one, no clutter training",
 "bev-local": "the network sees half as far around each answer",
 "bev-nodensity": "the two density measurements removed",
}

def agg(rs):
    f = np.array([x["found"] / max(x["total"], 1) for x in rs])
    b = np.array([x["budget_found"] / max(x["budget_total"], 1) for x in rs])
    c = np.array([x["contrast"] for x in rs]); p = np.array([x["pr_auc"] for x in rs])
    return f.mean(), b.mean(), c.mean(), c.std(), p.mean(), p.std(), int((c < FLAT).sum()), len(rs)

L = []
L.append("# Reading the ground as a picture: does it survive the competition arena?\n")
L.append("Every number here is measured on 100 frames of the competition recording, "
         "which none of these models was trained on. They are scored on a shared "
         "10 cm ground grid, so a model that labels every point and one that scores "
         "half-metre balls become the same measurement.\n")
L.append("**The number to beat** is the sliding-window classifier trained against "
         f"stray returns: it finds **{BAR_ROCKS} of 112** rocks, with a confidence "
         f"gap between rock and bare ground of **+{BAR_CONTRAST:.2f}**.\n")
L.append("## What each setting did\n")
L.append("| setting | what it changes | rocks found | confidence gap | runs that went blank |")
L.append("|---|---|---|---|---|")
L.append(f"| **the classifier, for comparison** | the model already deployed | "
         f"**{BAR_ROCKS}/112** | **+{BAR_CONTRAST:.2f}** | 0 |")
best = None
for a in ARMS:
    if a not in rows: continue
    f, b, c, cs, p, ps, flat, n = agg(rows[a])
    L.append(f"| `{a}` | {WHAT.get(a,'')} | {100*f:.0f}% | {c:+.2f} ± {cs:.2f} | **{flat} of {n}** |")
    if flat == 0 and (best is None or c > best[1]): best = (a, c, f, n)
L.append("")
L.append("\"Runs that went blank\" counts checkpoints whose confidence gap fell below "
         f"{FLAT:.2f} - the model still puts rocks above bare ground in the right order, "
         "but emits so little confidence that nothing lights up on screen. That failure "
         "is invisible to every score this project tracked before September, which is "
         "why it is the first column worth reading.\n")
L.append("## The verdict\n")
allrows = [x for a in ARMS if a in rows for x in rows[a]]
anyflat = sum(1 for x in allrows if x["contrast"] < FLAT)
bestck = max(allrows, key=lambda x: x["pr_auc"]) if allrows else None
L.append(f"Across **{len(allrows)} checkpoints** of {len([a for a in ARMS if a in rows])} "
         f"settings, **{anyflat} went blank** on the arena.\n")
if bestck:
    L.append(f"The single best checkpoint found **{bestck['found']} of {bestck['total']}** "
             f"rocks at its own threshold with a confidence gap of "
             f"**{bestck['contrast']:+.2f}**. The deployed classifier finds {BAR_ROCKS}.\n")
if best and best[3] >= 5:
    L.append(f"The steadiest setting was `{best[0]}` - no blank runs across {best[3]} "
             f"checkpoints, average confidence gap {best[1]:+.2f}. That is the only one "
             "worth taking further.\n")
else:
    L.append("**No setting kept every one of its checkpoints alive.** A family where "
             "runs silently go blank on a new arena is not deployable whatever its "
             "average - the same standard that ruled out the whole-frame segmenter "
             "applies here.\n")
L.append("## What this does not settle\n")
L.append("- **One arena.** Everything rests on a single competition recording, which "
         "the segmentation write-up already names as the weakest part of its own "
         "evidence. A second arena would be worth more than this whole sweep.\n")
L.append("- **Six folds a setting, not eleven.** Enough to see a family-wide pattern, "
         "not enough for a significance test on any single pair.\n")
L.append("- **The volleyball scores are in `summary.json` and are not the point.** "
         "They cannot see the failure that matters; that is the whole reason this "
         "sweep was scored on the arena instead.\n")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT}")
print("\n".join(L))
