# Next phase: three agents, in order

Run these in sequence. Each depends on the one before it.

1. **[01-cleanup.md](01-cleanup.md)** — restructure the repo, the data layout and the
   dashboard so the next two agents (and Brandon) can tell what is what. **Highest
   priority; everything else is built on it.**
2. **[02-slam.md](02-slam.md)** — improve the trajectory solver, and migrate the
   existing labels onto the re-solved recordings so nothing has to be relabelled.
3. **[03-training.md](03-training.md)** — run the next training suite on the cleaned
   structure and the improved recordings.

---

## Shared context: what is already settled

Three agents have worked on this. Do not re-litigate these.

### Settled by measurement

| question | answer | evidence |
|---|---|---|
| Does reflectivity/brightness help? | **No.** Stop spending on it. | Two independent sweeps. Data-level ROC-AUC 0.508 (chance). PointNet +0.0003 (p=0.70), PointNet++ −0.0103, segmentation −0.0104. |
| Is PointNet++ better than PointNet? | **No difference.** | Both sweeps. On full-sweep data PointNet 0.780 vs PointNet++ 0.765 (p=0.10), PointNet++ never won on raw data either. |
| Should frames be whole sensor rotations? | **Yes, decisively.** | +0.040–0.055 PR-AUC on every model, 10–11 of 11 folds, p ≤ 0.014, against a 0.013 noise floor. |
| Is per-point segmentation viable? | **Yes.** Ties PointNet++ and has a much higher floor. | Matched-population comparison: 0.767 vs 0.764; +0.072 on the 5 hardest recordings, worst case 0.480 → 0.594. |
| Will loop closure help SLAM? | **No.** | Revisit error is flat vs time gap (5.5 mm at <2 s, 7.5 mm at 20–35 s). No accumulating drift to close. |

### Two traps that have already produced wrong conclusions

**1. Never compare PR-AUC across different populations.** PR-AUC scales with the
positive rate. The default ablation report shows segmentation losing to PointNet++
by −0.37 — pure artifact, because it grades segmentation per point (~1% rock) and
the classifier per candidate ball (~19% rock). Use `rocklabel-train matched` for any
classify-vs-segment comparison. Geometry validated at 99.4–99.9% label agreement.

**2. The same trap applies *across folds*, and it has not been fixed anywhere.**
Rock prevalence ranges 6.3%–31.1% across the 11 recordings — a 5× spread. Every
per-fold table in the repo reports raw PR-AUC, so "which recording is hard" is partly
just "which recording has few rocks". Normalizing as `(AP − prevalence) / (1 − prevalence)`
(chance → 0, perfect → 1) reorders the folds:

| fold | rock % | raw PR-AUC | normalized | pose quality |
|---|---|---|---|---|
| VB4 | 19.8% | 0.483 | **0.355** | 4.2% off-level — *the cleanest run of the 11* |
| VB12 | 28.7% | 0.634 | 0.487 | 26.8% off-level |
| VB6 | 6.3% | 0.544 | 0.513 | 15.2% off-level |
| VB11 | 31.1% | 0.839 | 0.766 | 24.3% off-level — *the shakiest run* |
| VB3 | 24.4% | 0.927 | 0.903 | — |

**This matters a lot for prioritisation.** The SLAM handoff claims per-fold score
correlates −0.56 with pose difficulty, and concludes that better SLAM will lift model
accuracy. That correlation was computed on raw (prevalence-confounded) PR-AUC from the
old sparse data. Of the four runs with measured pose quality, **two contradict it
outright**: VB4 is the worst fold and the cleanest run; VB11 is the shakiest run and
scores 6th of 11. Re-measure before betting effort on it — instructions in
[02-slam.md](02-slam.md) §1.

### An open contradiction between the two handoffs

- SLAM handoff §9: "the two collapsing folds (Test6, Test4) are exactly the two hardest runs."
- Reflectivity handoff: "Test4 is the cleanest run, not the worst — ground std 0.041 m,
  4.2% off-level, best of 11."

Both cannot be true. Whoever picks up VB4 should resolve it: VB4 is the worst-performing
fold in every sweep so far, and if its pose is genuinely clean then its problem is
something else — labelling, rock type, or its 7 rocks being unusually small or flat.

### Numbers you will need

- **Noise floors** (same setting, different seed — the bar any effect must clear):
  0.0078 on the old raw-burst cache; **0.0128 for classifiers and 0.0207 for segmentation**
  on full-sweep data. Use the one matching the data you are on.
- **Always pair by fold.** Fold difficulty spans 0.42–0.93, far larger than any effect
  under test. Pooled averages cannot see these effects. Use the exact Wilcoxon
  signed-rank in `rocklabel/train/ablate.py`.
- **`compare` cannot do augmentation A/B** — `run_dir_name` tags the feature selection
  but not augmentation settings, so two arms collide on one directory. Use `ablate`,
  which gives every arm its own run root.
- **pytest needs `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`** in this environment. A ROS
  `launch_testing` plugin imports `lark`, which is missing, and crashes collection
  before any test runs. Nothing to do with this project.
- **Bash calls time out.** Launch long jobs with `nohup ... &` and poll.

### Read CLAUDE.md and obey it

Brandon reads summaries, not code. Plain English, no unexplained jargon, say what a
change means for him. **Any CLI change must land in the dashboard in the same piece of
work** — `rocklabel/dashboard/spec.py` generates the form, help, validation and command
preview from one entry, and `tests/test_dashboard.py` fails if the catalog drifts from
the real parsers.
