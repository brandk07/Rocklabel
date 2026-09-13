# Lance labelled checkpoint benchmark — 12 September 2026

All **297 available `best.pt` checkpoints** were scored on the fixed labelled competition/Lance cache with **zero inference errors**. The [complete machine-readable results](results.json) contain PR-AUC, ROC-AUC, precision, recall, F1, confusion counts, sample prevalence, checkpoint threshold, source paths, and input-contract differences for every checkpoint. The dashboard's Runs page shows the top ten in each prediction unit; model pickers show the Lance and original volleyball scores side by side. The volleyball `test_metrics.json` files were not changed.

The benchmark uses 109 sparse sampled classifier frames (89,612 labelled candidate centers, 8.91% rock) and 106 segmenter frames (185,855 scorable labelled points, 7.07% rock). PR-AUC is step-integrated average precision. Precision, recall and F1 use each checkpoint's stored validation threshold; two old checkpoints have no stored threshold and use 0.5. All scores are on the **same Lance recording**, but classifier candidate centers and segmenter points are different populations, so their PR-AUC values are not directly comparable. A high PR-AUC does not establish that a deployed threshold gives complete rock maps.

## Highest classifier candidate PR-AUC

| Checkpoint | PR-AUC | F1 | Precision | Recall | Threshold |
|---|---:|---:|---:|---:|---:|
| `stray/cls-both/loro_VolleyBallTest4.reslam` | **0.7112** | 0.6505 | 0.5946 | 0.7180 | 0.750 |
| `fullsweep/pointnet2-refl/loro_VolleyBallTest12.reslam` | 0.7074 | 0.5498 | 0.4157 | 0.8118 | 0.790 |
| `lance-campaign-v1/pointnet-matched/seed-44/pointnet_loro_VolleyBallTest6.reslam_dx-dy-dz` | 0.6911 | 0.5789 | 0.4656 | 0.7651 | 0.807 |
| `fullsweep/pointnet2-refl/loro_VolleyBallTest7.reslam` | 0.6879 | 0.5604 | 0.4443 | 0.7586 | 0.740 |
| `fullsweep/pointnet2-geom/loro_VolleyBallTest12.reslam` | 0.6846 | 0.5845 | 0.4877 | 0.7291 | 0.770 |

The current `deploy/cls-stray/trainall` checkpoint scores **0.5399 PR-AUC**, 0.5055 F1, 0.3729 precision and 0.7847 recall at its stored 0.710 threshold on this sampled benchmark. Its separate [full-recording map audit](../lance-review-2026-09-04/final-results.md) remains stronger deployment evidence than this candidate-level ranking. The top `cls-both` checkpoint has not been promoted to deployment on this score.

## Highest segmenter point PR-AUC

| Checkpoint | PR-AUC | F1 | Precision | Recall | Threshold |
|---|---:|---:|---:|---:|---:|
| `bev/bev-stray/loro_VolleyBallTest10.reslam` | **0.6918** | 0.5431 | 0.9412 | 0.3817 | 0.930 |
| `bev/bev-stray/loro_VolleyBallTest2.reslam` | 0.6707 | 0.4380 | 0.9545 | 0.2842 | 0.970 |
| `bev/bev-capped-relative/loro_VolleyBallTest12.reslam` | 0.6629 | 0.6348 | 0.6313 | 0.6384 | 0.810 |
| `bev/bev-capped-nostray/loro_VolleyBallTest10.reslam` | 0.6624 | 0.5962 | 0.8392 | 0.4624 | 0.760 |
| `bev/bev-capped/loro_VolleyBallTest2.reslam` | 0.6605 | 0.5128 | 0.8550 | 0.3662 | 0.770 |

At its stored threshold, the highest-AP segmenter recalls only 38.17% of labelled points. This illustrates why its ranking score alone cannot establish obstacle-map quality.

## Scope and exclusions

The fixed cache was generated with a 50 ms window, 0.5 m classifier neighborhoods, 2,048 segmenter input points, and every eligible negative candidate retained. **31 checkpoint scores used input settings different from their own stored inference contract**: 14 older seed-study models used a 0 ms window, and 17 dense segmenters were trained for 1,280 points. Their measured numbers remain in `results.json` and the picker with an input-difference flag, but they are omitted from the ranking table. The two `lance-target-negatives-v1` checkpoints are also omitted: one fitted on Lance negatives, and both used Lance negatives to select the checkpoint. These are useful development diagnostics, not independent Lance tests.

The sampled frames contain repeated sightings of the same 12 physical rocks. They do not weight rocks equally, cover every moment of the 35.5-minute recording, or measure accumulated-map completeness, persistent misses, fragmentation, and false occupied area. The prior labelled [visual and full-map audits](../lance-review-2026-09-04/final-results.md) address those deployment questions for the checkpoints they tested. The Lance recording has also informed model development, so this new leaderboard is exploratory; a fresh competition-like recording is needed for a final model-selection claim.

To reproduce or refresh the numbers after adding checkpoints, run `.venv/bin/python -m rocklabel.train.lance_benchmark --device cuda` from the repository root. Existing results are reused when checkpoint size and modification time match.
