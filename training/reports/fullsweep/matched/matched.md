# Segmentation vs sliding-window, scored the same way
Both models are graded on **one shared set of candidate centers**, using the centers' own rock/clear labels. The classifier scores each center directly; the segmenter scores it by taking the **max** of its per-point probabilities within **0.15 m** of that center.

This matters because the two tasks are otherwise graded on different populations - candidate balls are about 19% rock, individual points about 1% - and PR-AUC moves with that prevalence, so the raw numbers in the ablation report are not a like-for-like comparison. These are.

## PointNet++ · shape only  vs  Segmentation · shape only

11 folds, 93992 shared centers (527 centers had no segmented point nearby and were dropped from both sides).

**the segmenter wins on average**: mean PR-AUC difference +0.0031 (segmenter minus classifier), segmenter ahead on 6 of 11 folds, signed-rank p = 0.966.

| held-out run | classifier PR-AUC | segmenter PR-AUC | difference | no-skill | classifier F1 | segmenter F1 |
|---|---|---|---|---|---|---|
| VolleyBallTest10.reslam | 0.8112 | 0.7767 | -0.0345 | 0.082 | 0.7401 | 0.7234 |
| VolleyBallTest11.reslam | 0.8383 | 0.9121 | +0.0737 | 0.311 | 0.7509 | 0.8363 |
| VolleyBallTest12.reslam | 0.6346 | 0.6651 | +0.0305 | 0.287 | 0.5496 | 0.5839 |
| VolleyBallTest2.reslam | 0.9027 | 0.7330 | -0.1697 | 0.202 | 0.8378 | 0.6726 |
| VolleyBallTest3.reslam | 0.9275 | 0.8882 | -0.0393 | 0.244 | 0.8577 | 0.7917 |
| VolleyBallTest4.reslam | 0.4805 | 0.5936 | +0.1131 | 0.198 | 0.4388 | 0.5363 |
| VolleyBallTest5.reslam | 0.6640 | 0.6966 | +0.0326 | 0.120 | 0.5997 | 0.6525 |
| VolleyBallTest6.reslam | 0.5371 | 0.7343 | +0.1971 | 0.063 | 0.5436 | 0.7074 |
| VolleyBallTest7.reslam | 0.7964 | 0.7843 | -0.0121 | 0.163 | 0.7129 | 0.6988 |
| VolleyBallTest8.reslam | 0.9068 | 0.7449 | -0.1618 | 0.108 | 0.8411 | 0.7043 |
| VolleyBallTest9.reslam | 0.9027 | 0.9068 | +0.0041 | 0.300 | 0.8173 | 0.8278 |
| **mean** | **0.7638** | **0.7669** | **+0.0031** | | | |

Other metrics, as mean difference across folds (positive = segmenter ahead):

| metric | mean difference | segmenter wins | p |
|---|---|---|---|
| pr_auc | +0.0031 | 6/11 | 0.966 |
| roc_auc | +0.0115 | 7/11 | 0.577 |
| f1 | +0.0041 | 6/11 | 0.898 |
| precision | -0.0016 | 5/11 | 0.765 |
| recall | +0.0083 | 6/11 | 0.898 |

## PointNet++ · shape + reflectivity  vs  Segmentation · shape + reflectivity

11 folds, 93992 shared centers (527 centers had no segmented point nearby and were dropped from both sides).

**the segmenter wins on average**: mean PR-AUC difference +0.0041 (segmenter minus classifier), segmenter ahead on 6 of 11 folds, signed-rank p = 0.898.

| held-out run | classifier PR-AUC | segmenter PR-AUC | difference | no-skill | classifier F1 | segmenter F1 |
|---|---|---|---|---|---|---|
| VolleyBallTest10.reslam | 0.8099 | 0.7696 | -0.0402 | 0.082 | 0.7305 | 0.7092 |
| VolleyBallTest11.reslam | 0.8342 | 0.8981 | +0.0638 | 0.311 | 0.7460 | 0.8176 |
| VolleyBallTest12.reslam | 0.6020 | 0.6091 | +0.0071 | 0.287 | 0.5319 | 0.5438 |
| VolleyBallTest2.reslam | 0.8997 | 0.7376 | -0.1621 | 0.202 | 0.8344 | 0.6664 |
| VolleyBallTest3.reslam | 0.9312 | 0.8918 | -0.0394 | 0.244 | 0.8631 | 0.7946 |
| VolleyBallTest4.reslam | 0.4558 | 0.5950 | +0.1392 | 0.198 | 0.4165 | 0.5367 |
| VolleyBallTest5.reslam | 0.6526 | 0.6951 | +0.0425 | 0.120 | 0.5878 | 0.6385 |
| VolleyBallTest6.reslam | 0.5133 | 0.7059 | +0.1926 | 0.063 | 0.5020 | 0.7131 |
| VolleyBallTest7.reslam | 0.7946 | 0.7771 | -0.0176 | 0.163 | 0.7087 | 0.6798 |
| VolleyBallTest8.reslam | 0.8921 | 0.7411 | -0.1510 | 0.108 | 0.8250 | 0.7158 |
| VolleyBallTest9.reslam | 0.9006 | 0.9109 | +0.0103 | 0.300 | 0.8160 | 0.8330 |
| **mean** | **0.7533** | **0.7574** | **+0.0041** | | | |

Other metrics, as mean difference across folds (positive = segmenter ahead):

| metric | mean difference | segmenter wins | p |
|---|---|---|---|
| pr_auc | +0.0041 | 6/11 | 0.898 |
| roc_auc | +0.0147 | 7/11 | 0.465 |
| f1 | +0.0079 | 6/11 | 0.898 |
| precision | +0.0216 | 5/11 | 0.765 |
| recall | -0.0031 | 5/11 | 0.898 |

