# Visual deployment audit

This report evaluates accumulated maps on **distinct physical rocks**. Every rock contributes one row; repeated sightings do not increase its weight. Coverage is the fraction of the rock's occupied 10 cm ground cells receiving a prediction above the checkpoint's stored threshold. PR-AUC is intentionally not used as a substitute for map completeness.

Recording: `/home/brandon/Documents/Rocklabel/rocklabel/recordings/archive/misc/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.mcap`  
Labels: `/home/brandon/Documents/Rocklabel/rocklabel/labels/archive/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json`  
Floor band: `-0.10…+0.60 m`  
Accumulation: `5.0 s`  
Detection flag threshold: `25%` coverage (not a map-completeness or collision-avoidance criterion)

| model | task | stored threshold | checkpoint |
|---|---|---:|---|
| model 1 | classify | 0.710 | `/home/brandon/Documents/Rocklabel/rocklabel/training/experiments/deploy/cls-stray/trainall/best.pt` |
| model 2 | segment | 0.930 | `/home/brandon/Documents/Rocklabel/rocklabel/training/experiments/bev/bev-stray/loro_VolleyBallTest10.reslam/best.pt` |

| rock | intervals | visible returns | model 1 median / best / detected | model 2 median / best / detected | 
|---:|---:|---:|---:|---:|
| 1 | 25 | 1228 | 76% / 100% / 25/25 | 13% / 65% / 8/25 | 
| 3 | 24 | 2153 | 72% / 89% / 22/24 | 0% / 66% / 6/24 | 
| 4 | 16 | 4781 | 14% / 69% / 4/16 | 0% / 8% / 0/16 | 
| 5 | 27 | 2845 | 71% / 94% / 24/27 | 52% / 90% / 21/27 | 
| 6 | 11 | 3415 | 71% / 100% / 11/11 | 50% / 73% / 7/11 | 
| 7 | 15 | 4061 | 82% / 93% / 14/15 | 27% / 45% / 8/15 | 
| 8 | 6 | 4719 | 71% / 82% / 6/6 | 5% / 41% / 1/6 | 
| 9 | 27 | 3704 | 62% / 85% / 26/27 | 43% / 72% / 22/27 | 
| 10 | 24 | 1448 | 62% / 86% / 23/24 | 35% / 100% / 14/24 | 
| 11 | 19 | 2842 | 62% / 91% / 19/19 | 27% / 60% / 10/19 | 
| 12 | 20 | 2423 | 57% / 100% / 17/20 | 0% / 0% / 0/20 | 
| 13 | 28 | 4091 | 54% / 86% / 22/28 | 31% / 85% / 15/28 | 

## Automated parity check

**Result: the evidence does not support describing these deployed maps as similar.**

A material difference was fixed at **20% median rock coverage** before inspecting the selected images. Model 1 leads on 9/12 distinct rocks; model 2 leads on 0/12; 3/12 are within the gap.

- Model 1 coverage leads: `[1, 3, 6, 7, 8, 10, 11, 12, 13]`
- Model 2 coverage leads: `none`
- Model 1 persistent detection-only rocks: `[4, 12]`
- Model 2 persistent detection-only rocks: `none`
- Median clear-region positive cells per interval: model 1 `166.5`, model 2 `4.5`

Coverage and clear-region positives are separate tradeoffs. A completeness lead is evidence that the live maps are not equivalent, not permission to ignore false detections or declare an architecture universally superior.

## Automatically selected disagreements

- [Rock 1 · interval 341](cases/rock-001-interval-0341.png)
- [Rock 6 · interval 314](cases/rock-006-interval-0314.png)
- [Rock 12 · interval 258](cases/rock-012-interval-0258.png)
- [Rock 7 · interval 314](cases/rock-007-interval-0314.png)
- [Rock 5 · interval 341](cases/rock-005-interval-0341.png)
- [Rock 13 · interval 250](cases/rock-013-interval-0250.png)
- [Rock 3 · interval 63](cases/rock-003-interval-0063.png)
- [Rock 9 · interval 283](cases/rock-009-interval-0283.png)
- [Rock 10 · interval 283](cases/rock-010-interval-0283.png)
- [Rock 11 · interval 367](cases/rock-011-interval-0367.png)
- [Rock 8 · interval 244](cases/rock-008-interval-0244.png)
- [Rock 4 · interval 336](cases/rock-004-interval-0336.png)

## Interpretation guardrails

- A narrow crop is a diagnostic ablation, not proof that the full operational map works.
- Stored-threshold completeness and threshold-free ranking answer different questions.
- Native outputs differ (candidate centers versus sampled points); both are reduced to the same ground-cell size only after their real inference path.
- Inspect the PNGs before claiming that the models are operationally similar.

## Evaluation scope and threshold diagnostics

This is an unfiltered model-input audit. Live floating-point rejection and heightmap outlier rejection are not applied. The sampled, independent 5-second maps do not measure full-run stale detections or stopping latency.

Labelled rocks without auditable intervals: [].

[Operating-point sweep](operating-points.csv) holds inference fixed and reports coverage and false area separately from stored-threshold results. Equal-false-positive thresholds chosen from this table are diagnostics fitted on the evaluation recording, not validated deployment thresholds.
