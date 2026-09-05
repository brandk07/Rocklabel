# Segmentation vs sliding-window, scored the same way
Both models are graded on **one shared set of candidate centers**, using the centers' own rock/clear labels. The classifier scores each center directly; the segmenter scores it by taking the **max** of its per-point probabilities within **0.15 m** of that center.

This matters because the two tasks are otherwise graded on different populations - candidate balls are about 19% rock, individual points about 1% - and PR-AUC moves with that prevalence, so the raw numbers in the ablation report are not a like-for-like comparison. These are.

## Classifier · stock  vs  Segmentation · deployment settings

6 folds, 49396 shared centers (306 centers had no segmented point nearby and were dropped from both sides).

**the segmenter wins on average**: mean PR-AUC difference +0.0081 (segmenter minus classifier), segmenter ahead on 3 of 6 folds, signed-rank p = 0.688.

| held-out run | classifier PR-AUC | segmenter PR-AUC | difference | no-skill | classifier F1 | segmenter F1 |
|---|---|---|---|---|---|---|
| VolleyBallTest10.reslam | 0.8271 | 0.7669 | -0.0603 | 0.082 | 0.7582 | 0.7108 |
| VolleyBallTest11.reslam | 0.8703 | 0.9097 | +0.0394 | 0.311 | 0.7774 | 0.8331 |
| VolleyBallTest3.reslam | 0.9284 | 0.9084 | -0.0199 | 0.244 | 0.8497 | 0.8194 |
| VolleyBallTest4.reslam | 0.5137 | 0.5563 | +0.0426 | 0.198 | 0.4589 | 0.5067 |
| VolleyBallTest6.reslam | 0.6274 | 0.6912 | +0.0637 | 0.063 | 0.6282 | 0.6954 |
| VolleyBallTest9.reslam | 0.9210 | 0.9039 | -0.0171 | 0.300 | 0.8452 | 0.8252 |
| **mean** | **0.7813** | **0.7894** | **+0.0081** | | | |

Other metrics, as mean difference across folds (positive = segmenter ahead):

| metric | mean difference | segmenter wins | p |
|---|---|---|---|
| pr_auc | +0.0081 | 3/6 | 0.688 |
| norm_pr_auc | +0.0103 | 3/6 | 0.688 |
| roc_auc | +0.0168 | 3/6 | 0.438 |
| f1 | +0.0122 | 3/6 | 0.438 |
| precision | +0.0204 | 5/6 | 0.438 |
| recall | +0.0065 | 4/6 | 1.000 |

## Classifier · trained with stray returns  vs  Segmentation · trained with stray returns

6 folds, 49396 shared centers (306 centers had no segmented point nearby and were dropped from both sides).

**the classifier wins on average**: mean PR-AUC difference -0.0030 (segmenter minus classifier), segmenter ahead on 3 of 6 folds, signed-rank p = 0.844.

| held-out run | classifier PR-AUC | segmenter PR-AUC | difference | no-skill | classifier F1 | segmenter F1 |
|---|---|---|---|---|---|---|
| VolleyBallTest10.reslam | 0.8110 | 0.7332 | -0.0777 | 0.082 | 0.7439 | 0.7043 |
| VolleyBallTest11.reslam | 0.8814 | 0.9046 | +0.0232 | 0.311 | 0.7945 | 0.8260 |
| VolleyBallTest3.reslam | 0.9311 | 0.8977 | -0.0334 | 0.244 | 0.8529 | 0.8181 |
| VolleyBallTest4.reslam | 0.5352 | 0.6285 | +0.0933 | 0.198 | 0.4814 | 0.5537 |
| VolleyBallTest6.reslam | 0.6514 | 0.6572 | +0.0058 | 0.063 | 0.6322 | 0.6535 |
| VolleyBallTest9.reslam | 0.9325 | 0.9037 | -0.0288 | 0.300 | 0.8624 | 0.8184 |
| **mean** | **0.7904** | **0.7875** | **-0.0030** | | | |

Other metrics, as mean difference across folds (positive = segmenter ahead):

| metric | mean difference | segmenter wins | p |
|---|---|---|---|
| pr_auc | -0.0030 | 3/6 | 0.844 |
| norm_pr_auc | -0.0023 | 3/6 | 0.844 |
| roc_auc | +0.0048 | 3/6 | 1.000 |
| f1 | +0.0011 | 3/6 | 0.844 |
| precision | +0.0000 | 3/6 | 0.844 |
| recall | +0.0007 | 2/6 | 1.000 |

