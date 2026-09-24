# Visual deployment audit

This report evaluates accumulated maps on **distinct physical rocks**. Every rock contributes one row; repeated sightings do not increase its weight. Coverage is the fraction of the rock's occupied 10 cm ground cells receiving a prediction above the checkpoint's stored threshold. PR-AUC is intentionally not used as a substitute for map completeness.

Recording: `/home/brandon/Documents/Rocklabel/rocklabel/recordings/archive/misc/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.mcap`  
Labels: `/home/brandon/Documents/Rocklabel/rocklabel/labels/archive/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json`  
Floor band: `-0.10…+0.60 m`  
Accumulation: `5.0 s`  
Detection flag threshold: `25%` coverage (not a map-completeness or collision-avoidance criterion)

| model | task | stored threshold | checkpoint |
|---|---|---:|---|
| model 1 | classify | 0.652 | `/home/brandon/Documents/Rocklabel/rocklabel/training/experiments/neighborhood-history-v1/_smoke/fixed-r050__h1/seed-42/best.pt` |
| model 2 | classify | 0.655 | `/home/brandon/Documents/Rocklabel/rocklabel/training/experiments/neighborhood-history-v1/_smoke/adaptive-r020-r075-k256__h5-30s/seed-42/best.pt` |

| rock | intervals | visible returns | model 1 median / best / detected | model 2 median / best / detected | 
|---:|---:|---:|---:|---:|
| 12 | 4 | 279 | 76% / 100% / 4/4 | 14% / 40% / 1/4 | 

## Automated parity check

**Result: the evidence does not support describing these deployed maps as similar.**

A material difference was fixed at **20% median rock coverage** before inspecting the selected images. Model 1 leads on 1/1 distinct rocks; model 2 leads on 0/1; 0/1 are within the gap.

- Model 1 coverage leads: `[12]`
- Model 2 coverage leads: `none`
- Model 1 persistent detection-only rocks: `none`
- Model 2 persistent detection-only rocks: `none`
- Median clear-region positive cells per interval: model 1 `81.5`, model 2 `34.0`

Coverage and clear-region positives are separate tradeoffs. A completeness lead is evidence that the live maps are not equivalent, not permission to ignore false detections or declare an architecture universally superior.

## Automatically selected disagreements

- [Rock 12 · interval 0](cases/rock-012-interval-0000.png)

## Interpretation guardrails

- A narrow crop is a diagnostic ablation, not proof that the full operational map works.
- Stored-threshold completeness and threshold-free ranking answer different questions.
- Native outputs differ (candidate centers versus sampled points); both are reduced to the same ground-cell size only after their real inference path.
- Inspect the PNGs before claiming that the models are operationally similar.

## Input contracts and scoring cost

Each checkpoint is scored with its own radius and history policy. History comes from one causal buffer fed every decoded sweep, including sweeps the scoring stride skips and sweeps before the audit window. Unscorable candidates stay in every coverage denominator; they only fail to produce a prediction.

| model | radius | history | candidates | unscorable | mean support points | preprocess s | network s |
|---|---|---|---:|---:|---:|---:|---:|
| model 1 | fixed 0.50 m | current sweep only | 24854 | 1209 | 3073 | 1.6 | 1.1 |
| model 2 | adaptive 0.20-0.75 m, target 256 real points | sweeps aged 0, 7.5, 15, 22.5, 30 s (tolerance 0.1 s) | 24854 | 34 | 16380 | 1.7 | 1.2 |

## Coverage by history stage

Stage is the interval's start, in seconds since the recording began. Within a stage each rock contributes its median interval once.

| stage | rocks | model 1 macro median coverage | model 2 macro median coverage |
|---|---:|---:|---:|
| startup 0-4 s | 0 | - | - |
| partial 4-30 s | 0 | - | - |
| mature >30 s | 1 | 76% | 14% |

## Evaluation scope and threshold diagnostics

This is an unfiltered model-input audit. Live floating-point rejection and heightmap outlier rejection are not applied. The sampled, independent 5-second maps do not measure full-run stale detections or stopping latency.

Labelled rocks without auditable intervals: [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13].

[Operating-point sweep](operating-points.csv) holds inference fixed and reports coverage and false area separately from stored-threshold results. Equal-false-positive thresholds chosen from this table are diagnostics fitted on the evaluation recording, not validated deployment thresholds.
