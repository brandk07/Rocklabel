# `training/` — everything the training side produces

Four folders, in the order the work flows through them:

| folder | what's in it | size | in git? |
|---|---|---|---|
| **`caches/`** | Point clouds pooled into one fast-loading blob per set of settings. What training actually reads. | ~1.9 GB | no |
| **`experiments/`** | Every model that has ever been trained here, one folder per fold. | ~3.9 GB | no |
| **`reports/`** | The write-ups: figures, tables, and the summary of what each experiment concluded. | ~7 MB | **yes** |
| **`exported/`** | Finished models packaged to run outside Python. | ~31 MB | **yes** |

Each folder has its own README with more detail.

## The short version of how it fits together

1. **Generate** turns a labeled recording into a dataset under `datasets/`.
2. **Build cache** pools several of those datasets into `caches/<profile>/`.
   One cache per *generation profile* — the named way the frames were cut. The
   tool refuses to mix two profiles into one cache, because their frames hold
   different numbers of points and a score across them would mean nothing.
3. **Train / Compare / Ablation sweep** read a cache and write one folder per
   trained fold under `experiments/`.
4. **The report step** reads those folders and writes `reports/`.
5. **Export** packages one chosen model into `exported/`.

You can do all five from the dashboard (`rocklabel dash`) without typing a
command.

## Why the big folders aren't in git

`caches/` and `experiments/` are about 6 GB between them and both can be
rebuilt from the recordings and labels, which *are* kept. They were tracked
once and grew the repository's history to 3.2 GB before it was noticed.

`reports/` and `exported/` stay in git on purpose: they are small, and they are
the only record of what was concluded and what is deployable.

## A note on reading the numbers

The headline score everywhere here is **PR-AUC**, and it is not comparable
across different sets of data. It moves with how much rock is in the data — the
eleven volleyball recordings range from 6% rock to 31% rock, a five-fold
spread — so a fold that looks "hard" may just have few rocks in it, and a
per-point model always looks worse than a per-candidate one for the same
reason. Use `rocklabel-train matched` when comparing two models that are graded
on different things.
