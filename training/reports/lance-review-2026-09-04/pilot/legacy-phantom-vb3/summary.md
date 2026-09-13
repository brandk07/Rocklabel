# Visual deployment audit

This report evaluates accumulated maps on **distinct physical rocks**. Every rock contributes one row; repeated sightings do not increase its weight. Coverage is the fraction of the rock's occupied 10 cm ground cells receiving a prediction above the checkpoint's stored threshold. PR-AUC is intentionally not used as a substitute for map completeness.

Recording: `/home/brandon/Documents/perception-2026-testing/rocklabel/recordings/archive/misc/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.mcap`  
Labels: `/home/brandon/Documents/perception-2026-testing/rocklabel/labels/archive/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json`  
Floor band: `-0.10…+0.60 m`  
Accumulation: `5.0 s`  
Complete detection threshold: `25%` coverage

| model | task | stored threshold | checkpoint |
|---|---|---:|---|
| model 1 | classify | 0.710 | `/home/brandon/Documents/perception-2026-testing/rocklabel/training/experiments/deploy/cls-stray/trainall/best.pt` |
| model 2 | classify | 0.790 | `/home/brandon/Documents/perception-2026-testing/rocklabel/training/experiments/stray/cls-phantom/loro_VolleyBallTest3.reslam/best.pt` |

| rock | intervals | visible returns | model 1 median / best / detected | model 2 median / best / detected | 
|---:|---:|---:|---:|---:|
| 1 | 3 | 165 | 78% / 80% / 3/3 | 36% / 40% / 3/3 | 
| 3 | 3 | 470 | 84% / 91% / 3/3 | 70% / 77% / 3/3 | 
| 4 | 3 | 1158 | 54% / 67% / 2/3 | 56% / 57% / 2/3 | 
| 5 | 2 | 1006 | 83% / 94% / 2/2 | 78% / 88% / 2/2 | 
| 9 | 3 | 439 | 89% / 91% / 3/3 | 72% / 73% / 3/3 | 
| 10 | 2 | 182 | 76% / 89% / 2/2 | 82% / 89% / 2/2 | 
| 11 | 2 | 396 | 66% / 67% / 2/2 | 57% / 67% / 2/2 | 
| 12 | 2 | 362 | 73% / 100% / 2/2 | 77% / 100% / 2/2 | 
| 13 | 3 | 298 | 54% / 59% / 2/3 | 0% / 41% / 1/3 | 

## Automated parity check

**Result: the evidence does not support describing these deployed maps as similar.**

A material difference was fixed at **20% median rock coverage** before inspecting the selected images. Model 1 leads on 2/9 distinct rocks; model 2 leads on 0/9; 7/9 are within the gap.

- Model 1 coverage leads: `[1, 13]`
- Model 2 coverage leads: `none`
- Model 1 persistent detection-only rocks: `none`
- Model 2 persistent detection-only rocks: `none`
- Median clear-region positive cells per interval: model 1 `84.0`, model 2 `69.0`

Coverage and clear-region positives are separate tradeoffs. A completeness lead is evidence that the live maps are not equivalent, not permission to ignore false detections or declare an architecture universally superior.

## Automatically selected disagreements

- [Rock 13 · interval 2](cases/rock-013-interval-0002.png)
- [Rock 1 · interval 2](cases/rock-001-interval-0002.png)
- [Rock 3 · interval 2](cases/rock-003-interval-0002.png)
- [Rock 9 · interval 36](cases/rock-009-interval-0036.png)
- [Rock 11 · interval 49](cases/rock-011-interval-0049.png)
- [Rock 4 · interval 2](cases/rock-004-interval-0002.png)
- [Rock 10 · interval 2](cases/rock-010-interval-0002.png)
- [Rock 12 · interval 36](cases/rock-012-interval-0036.png)
- [Rock 5 · interval 36](cases/rock-005-interval-0036.png)

## Interpretation guardrails

- A narrow crop is a diagnostic ablation, not proof that the full operational map works.
- Stored-threshold completeness and threshold-free ranking answer different questions.
- Native outputs differ (candidate centers versus sampled points); both are reduced to the same ground-cell size only after their real inference path.
- Inspect the PNGs before claiming that the models are operationally similar.

Pilot scope: 150–450 s, unfiltered inputs, visibility-rich intervals. Unaudited rocks: [6, 7, 8]. The operating-point sweep is diagnostic and fitted on these evaluation data.
