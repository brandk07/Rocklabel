"""Score every BEV CNN checkpoint on the competition arena, the honest way.

Recipe lifted from the stray sweep's rock_detect.py / contrast.py, because the
whole point is that these numbers have to be comparable to the ones already on
disk. Four things are reported per checkpoint, and the first two are the ones
that decide anything:

  rocks found   labelled rocks with >=2 lit 10 cm cells on them, counting only
                rocks carrying >=15 real returns in that frame
  contrast      median confidence on real rock minus median on bare ground, in
                the 0-1 units the colour ramp spans. PR-AUC is rank-based and
                cannot see a model's confidence collapse; this can, and that
                collapse is exactly how the segmenter fooled two reports
  FP cells      lit cells on bare ground, per frame - the cost of the above
  cell PR-AUC   threshold-free, on the same shared 10 cm cells, so a per-point
                model and a per-ball model are one measurement

Runs on the processor on purpose: the graphics card is training, and two jobs
on it slow both by about nine times.
"""
import os, sys, json, glob
import numpy as np, torch

R = "/home/brandon/Documents/perception-2026-testing/rocklabel"
sys.path.insert(0, R)
os.chdir(R)
from rocklabel.config import load_config
from rocklabel.recording.pipeline import ScanStream, WindowedScanStream
from rocklabel.train.models import build_model, model_task
from rocklabel.geometry.leveling import pin_level_to_labels
from rocklabel.labels import load_labels
from rocklabel.dataset.labeling import label_rocks, points_in_rock, LABEL_ROCK, LABEL_IGNORE
from rocklabel.dataset.neighborhoods import build_inference_frame, build_inference_samples
from rocklabel.train import metrics as M
from matplotlib.path import Path as MplPath

CELL, MIN_PTS_ON_ROCK, MIN_CELLS = 0.10, 15, 2
FP_BUDGET = 20  # wrong cells per frame every model is allowed, for a fair sweep
LANCE = (f"{R}/recordings/archive/misc/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.mcap",
         f"{R}/labels/archive/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json")
WANT = int(os.environ.get("EVAL_FRAMES", "100"))
STRIDE = int(os.environ.get("EVAL_STRIDE", "25"))


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c, g = ck["config"], ck["generator"]
    m = build_model(c["model"], tnet=c["tnet"], dropout=c.get("dropout"),
                    features=c.get("features"), seg_npoints=c.get("seg_npoints"),
                    seg_radii=c.get("seg_radii"), seg_height_ref=c.get("seg_height_ref"),
                    seg_coord_ref=c.get("seg_coord_ref"),
                    # The grid model's own settings, which the sweep's script
                    # predates and would silently build a default-shaped net for.
                    bev_cell=c.get("bev_cell"), bev_grid=c.get("bev_grid"),
                    bev_width=c.get("bev_width"), bev_depth=c.get("bev_depth"),
                    bev_channels=c.get("bev_channels"),
                    bev_density_norm=c.get("bev_density_norm"))
    m.load_state_dict(ck["model"]); m.eval()
    return m, g, float(ck.get("threshold", 0.5)), model_task(c["model"])


def cells_of(xy):
    return np.floor(np.asarray(xy) / CELL).astype(np.int64)


def frames_of(mcap, labpath, want, stride):
    LS = load_labels(labpath)
    cfg = pin_level_to_labels(load_config(None), LS.level)
    gc = cfg["generator"]
    av = json.load(open(labpath)).get("arena")
    arena = MplPath(np.array(av["vertices"])) if av else None
    out = []
    st = WindowedScanStream(ScanStream(mcap, cfg, stride=1, progress=False), 0.05)
    for i, scan in enumerate(st):
        if i % stride:
            continue
        base = scan.T_odom_base[:3, 3]
        lo = np.array([base[0]-gc["crop_backward_m"], base[1]-gc["crop_right_m"], base[2]-gc["crop_down_m"]])
        hi = np.array([base[0]+gc["crop_forward_m"], base[1]+gc["crop_left_m"], base[2]+gc["crop_up_m"]])
        if LS.z_band is not None:
            lo[2], hi[2] = LS.z_band
        keep = ((scan.xyz_odom >= lo) & (scan.xyz_odom <= hi)).all(axis=1)
        if keep.sum() < 200:
            continue
        out.append((scan.xyz_odom[keep].astype(np.float64),
                    scan.intensity[keep].astype(np.float32), base.astype(np.float64)))
        if len(out) >= want:
            break
    vis = []
    for xyz, _i, _b in out:
        vis.append([(r, int(points_in_rock(xyz, r).sum())) for r in LS.rocks
                    if int(points_in_rock(xyz, r).sum()) >= MIN_PTS_ON_ROCK])
    return out, vis, LS, gc, arena


def score(path, frames, vis, LS, gc, arena):
    model, g, thr, task = load(path)
    rng = np.random.default_rng(int(g["seed"]))
    found = miss = fp_cells = 0
    all_y, all_p, per_frame = [], [], []
    for (xyz, inten, base), here in zip(frames, vis):
        if task == "classify":
            s = build_inference_samples(xyz, inten, g, rng)
            if s is None:
                continue
            with torch.no_grad():
                p = torch.sigmoid(model(torch.from_numpy(s["neighborhoods"]),
                    torch.from_numpy(s["true_counts"].astype(np.int64)))).numpy()
            pos = s["centers_odom"].astype(np.float64)
        else:
            fr = build_inference_frame(xyz, inten, base, g, rng)
            if fr is None:
                continue
            n = int(fr["true_count"])
            with torch.no_grad():
                p = torch.sigmoid(model(torch.from_numpy(fr["points"])[None],
                    torch.tensor([n])))[0].numpy()[:n]
            pos = xyz[fr["index"][:n]]

        cid = cells_of(pos[:, :2])
        key = cid[:, 0] * 100000 + cid[:, 1]
        order = np.argsort(key)
        key_s, pos_s, p_s = key[order], pos[order], p[order]
        edges = np.r_[0, np.flatnonzero(np.diff(key_s)) + 1, len(key_s)]
        c_pos, c_p = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            j = a + int(np.argmax(p_s[a:b]))
            c_pos.append(pos_s[j]); c_p.append(p_s[j])
        c_pos, c_p = np.asarray(c_pos), np.asarray(c_p)
        if arena is not None:
            m_in = arena.contains_points(c_pos[:, :2])
            c_pos, c_p = c_pos[m_in], c_p[m_in]
        if not len(c_pos):
            continue
        lab = label_rocks(c_pos, LS.rocks, gc["boundary_shell_m"])
        keepm = lab != LABEL_IGNORE
        all_y.append((lab == LABEL_ROCK).astype(np.int8)[keepm])
        all_p.append(c_p[keepm])
        hot = c_p > thr
        for rock, _n in here:
            inr = points_in_rock(c_pos, rock)
            found += int(int((inr & hot).sum()) >= MIN_CELLS)
            miss += int(int((inr & hot).sum()) < MIN_CELLS)
        fp_cells += int((hot & (lab == 0)).sum())
        # Kept so the threshold can be swept afterwards: a model that finds
        # nothing at its own threshold may only be badly calibrated for this
        # arena, which is a different problem from being short of capability.
        per_frame.append((c_p, lab, [points_in_rock(c_pos, r) for r, _ in here]))

    y = np.concatenate(all_y); q = np.concatenate(all_p)
    # Contrast: what a person watching the live view actually sees.
    on_rock = float(np.median(q[y == 1])) if (y == 1).any() else float("nan")
    on_ground = float(np.median(q[y == 0])) if (y == 0).any() else float("nan")
    # Step 3 of the sweep's recipe: how many rocks does it find when every
    # model is allowed the same budget of wrong cells? A model that stays flat
    # here at every threshold is genuinely short of capability.
    budget_found = budget_total = 0
    best_thr = None
    for t in np.linspace(0.01, 0.999, 60)[::-1]:
        f = m_ = fp = 0
        for c_p, lab, rock_masks in per_frame:
            h = c_p > t
            fp += int((h & (lab == 0)).sum())
            for inr in rock_masks:
                f += int(int((inr & h).sum()) >= MIN_CELLS)
                m_ += int(int((inr & h).sum()) < MIN_CELLS)
        if fp / max(len(per_frame), 1) > FP_BUDGET:
            break
        budget_found, budget_total, best_thr = f, f + m_, float(t)
    return dict(found=found, total=found + miss, contrast=on_rock - on_ground,
                budget_found=budget_found, budget_total=budget_total,
                budget_thr=best_thr,
                on_rock=on_rock, on_ground=on_ground,
                fp=fp_cells / max(len(frames), 1),
                pr_auc=float(M.average_precision(y, q)), thr=thr)


def load_cached():
    import pickle
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lance100.pkl"), "rb") as f:
        d = pickle.load(f)
    return d["frames"], d["vis"], d["LS"], d["gc"], d["arena"]


if __name__ == "__main__":
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lance100.pkl")):
        frames, vis, LS, gc, arena = load_cached()
    else:
        frames, vis, LS, gc, arena = frames_of(*LANCE, WANT, STRIDE)
    tot = sum(len(v) for v in vis)
    print(f"lance: {len(frames)} frames, ~{np.mean([len(f[0]) for f in frames]):.0f} pts each, "
          f"{tot} rock sightings\n")
    targets = []
    ref = "training/experiments/deploy/cls-stray/trainall/best.pt"
    if os.path.exists(ref):
        targets.append(("REFERENCE cls-stray", ref))
    for arm in sorted(os.listdir("training/experiments/bev")) if os.path.isdir("training/experiments/bev") else []:
        for d in sorted(glob.glob(f"training/experiments/bev/{arm}/loro_*/best.pt")):
            targets.append((f"{arm}/{os.path.basename(os.path.dirname(d))[5:][:16]}", d))
    print(f"{'checkpoint':<38} {'found@own':>11} {'found@20fp':>11} {'contrast':>9} "
          f"{'FP/frame':>9} {'PR-AUC':>7}")
    rows = {}
    for name, path in targets:
        try:
            r = score(path, frames, vis, LS, gc, arena)
        except Exception as e:
            print(f"  {name:<38} FAILED {type(e).__name__}: {e}"); continue
        rows.setdefault(name.split("/")[0], []).append(r)
        print(f"{name:<38} {r['found']:4d}/{r['total']:<6d} "
              f"{r['budget_found']:4d}/{r['budget_total']:<6d} {r['contrast']:+9.3f} "
              f"{r['fp']:9.1f} {r['pr_auc']:7.3f}", flush=True)
    print(f"\n{'arm':<22} {'found@own':>10} {'found@20fp':>11} {'contrast':>16} "
          f"{'PR-AUC':>15} {'flat':>7}")
    for arm, rs in rows.items():
        f = np.array([x["found"] / max(x["total"], 1) for x in rs])
        c = np.array([x["contrast"] for x in rs]); p = np.array([x["pr_auc"] for x in rs])
        flat = int((c < 0.20).sum())
        b = np.array([x["budget_found"] / max(x["budget_total"], 1) for x in rs])
        print(f"{arm:<22} {100*f.mean():9.0f}% {100*b.mean():10.0f}% "
              f"{c.mean():+8.3f}+-{c.std():5.3f} "
              f"{p.mean():8.3f}+-{p.std():5.3f} {flat:3d}/{len(rs)}")
    json.dump({k: v for k, v in rows.items()},
              open("/tmp/claude-1000/-home-brandon-Documents-perception-2026-testing/2865bc6a-44fc-4cae-8ca6-d3c5ef5181ec/scratchpad/bev_lance.json", "w"), indent=1)
