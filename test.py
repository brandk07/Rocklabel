"""Readable dump + sanity-check of one generated frame (both dataset formats).

The two formats describe the SAME frame but count DIFFERENT populations:

  format A (points/) = class-rebalanced *sample centers* for a point classifier
                       -> deliberately rock-heavy (clear samples are thrown away)
  format B (bev/)    = a fixed 80x80 top-down raster, EVERY occupied cell labeled
                       -> the honest spatial rock/clear balance of the scene

Run:  python test.py
"""

import json
import numpy as np

DATASET = "datasets/DATASET3"
RUN = "backroomweightreflect"
FRAME = "frame_000010"


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --- config + labels, straight from the dataset's own manifest ----------------
manifest = json.load(open(f"{DATASET}/manifest.json"))
g = manifest["config"]["generator"]
labels_path = manifest["runs"][RUN]["labels_path"]
rocks = json.load(open(labels_path))["rocks"]
sphere_c = np.array([r["center"] for r in rocks], float)
sphere_r = np.array([r["radius"] for r in rocks], float)

pts = np.load(f"{DATASET}/points/{RUN}/{FRAME}.npz")
bev = np.load(f"{DATASET}/bev/{RUN}/{FRAME}.npz")


# =============================================================================
rule("FORMAT A  -  points/  (per-point neighborhood samples)")
# =============================================================================
nb = pts["neighborhoods"]      # [S, 256, 4]  (dx, dy, dz-above-local-ground, intensity)
lab = pts["labels"]            # [S]          0=clear 1=rock
cnt = pts["true_counts"]       # [S]          real neighbor count before pad/subsample
cen = pts["centers_odom"]      # [S, 3]       where each sample sits in the world

n = len(lab)
n_rock = int((lab == 1).sum())
n_clear = int((lab == 0).sum())
print(f"samples (S)          : {n}         # one center + its 256-point neighborhood")
print(f"  labeled rock (1)   : {n_rock}")
print(f"  labeled clear (0)  : {n_clear}")
print(f"neighborhoods shape  : {nb.shape}   # 256 = fixed point budget per sample, NOT S")
print(f"true neighbor counts : min {cnt.min()}, max {cnt.max()} (>=256 -> subsampled to 256)")
print()
print(f"This is {n_rock}/{n} rock ON PURPOSE. negative_keep_prob = {g['negative_keep_prob']}:")
print(f"every rock center is kept, but each clear center is kept only ~{g['negative_keep_prob']:.0%}")
print("of the time. So this ratio describes the SAMPLER, not the scene. For the")
print("real balance, see format B below.")

# Independent geometric check that the labels are actually right ---------------
d_to_surface = (np.linalg.norm(cen[:, None, :] - sphere_c[None], axis=2) - sphere_r[None]).min(axis=1)
inside_sphere = d_to_surface <= 0.0
rock_ok = int((inside_sphere & (lab == 1)).sum())
clear_ok = int((~inside_sphere & (lab == 0)).sum())
print()
print("independent check (recomputed from the label spheres):")
print(f"  rock centers actually inside a sphere : {rock_ok} / {n_rock}")
print(f"  clear centers actually outside spheres: {clear_ok} / {n_clear}")
print("  -> labels match the geometry" if rock_ok == n_rock and clear_ok == n_clear
      else "  -> MISMATCH")

# Decode one sample so the neighborhood columns are concrete -------------------
i = int(np.argmax(lab == 1))
s = nb[i]
print()
print(f"example rock sample #{i}: center(odom) = {cen[i].round(3)}")
print("  neighborhood columns are [dx, dy, dz_above_local_ground, intensity]:")
print(f"    dx range {s[:,0].min():+.2f}..{s[:,0].max():+.2f} m  (relative to center)")
print(f"    dy range {s[:,1].min():+.2f}..{s[:,1].max():+.2f} m")
print(f"    dz range {s[:,2].min():+.2f}..{s[:,2].max():+.2f} m  (0 = lowest point = local ground)")
print(f"    intensity range {s[:,3].min():.2f}..{s[:,3].max():.2f}")


# =============================================================================
rule("FORMAT B  -  bev/  (bird's-eye-view raster)")
# =============================================================================
ch = bev["channels"]           # [8, H, W]
mask = bev["label_mask"]        # [H, W] uint8: 0 clear, 1 rock, 255 ignore/empty
H, W = mask.shape

# Why 80x80 = 6400 cells: it's a fixed top-down grid over the crop box.
x_m = g["crop_forward_m"] + g["crop_backward_m"]
y_m = g["crop_left_m"] + g["crop_right_m"]
print(f"grid            : {H} x {W} = {H*W} cells  (a fixed top-down IMAGE of the ground)")
print(f"crop box        : {x_m:g} m (x) by {y_m:g} m (y), cell = {g['bev_cell_m']:g} m")
print(f"                  {x_m:g}/{g['bev_cell_m']:g} = {H}  and  {y_m:g}/{g['bev_cell_m']:g} = {W}")
print("Every 10x10 cm patch of ground is one cell whether or not lidar hit it.")

occ = ch[0] > 0                                  # channel 0 = 'valid' (>=1 point)
n_occ = int(occ.sum())
n_rockc = int((mask == 1).sum())
n_clearc = int((mask == 0).sum())
n_ign_occ = int(((mask == 255) & occ).sum())     # occupied but only boundary-shell points
n_empty = int(((mask == 255) & ~occ).sum())      # no points at all
print()
print(f"occupied cells (channel 0)   : {n_occ}")
print(f"  rock  (mask == 1)          : {n_rockc}")
print(f"  clear (mask == 0)          : {n_clearc}")
print(f"  ignore, occupied (mask 255): {n_ign_occ}   # points, but all in the boundary shell")
print(f"empty cells (mask 255)       : {n_empty}   # <-- these are the 255s you saw everywhere")
print(f"total                        : {n_rockc}+{n_clearc}+{n_ign_occ}+{n_empty} = {H*W}")
print()
print(f"So the mask is NOT 'always 255': only {n_occ} of {H*W} cells have data; the rest")
print("is empty ground outside sensor range. numpy's '...' just hides the busy middle.")
print(f"Occupied balance is {n_rockc} rock / {n_clearc} clear  (~half) -> matches the picture.")

# ASCII top-down map of just the occupied region so you can SEE the balance ----
ii, jj = np.nonzero(mask != 255)
i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
glyph = {1: "#", 0: ".", 255: " "}   # # rock, . clear, (space) empty/ignore
print()
print(f"occupied region of label_mask [rows {i0}:{i1}, cols {j0}:{j1}]  (# rock  . clear):")
for row in mask[i0:i1, j0:j1]:
    print("  " + "".join(glyph.get(int(v), " ") for v in row))

# What the 8 channels mean, shown on one real occupied cell -------------------
ci, cj = ii[np.argmax(mask[ii, jj] == 1)], jj[np.argmax(mask[ii, jj] == 1)]
names = ["valid(0/1)", "log1p(count)", "max_z-base_z", "min_z-base_z",
         "z_span", "z_std", "mean_intensity", "max_intensity"]
print()
print(f"the 8 channels, read off one rock cell ({ci},{cj}):")
for k, nm in enumerate(names):
    print(f"  channel {k}  {nm:<15}= {ch[k, ci, cj]:+.4f}")


# =============================================================================
rule("SAME FRAME, TWO ANSWERS  (both correct)")
# =============================================================================
print(f"format A  samples : {n} ({n_rock} rock)      <- rebalanced sample centers")
print(f"format B  cells   : {n_occ} occupied, {n_rockc} rock / {n_clearc} clear  <- true scene balance")
print(f"shared frame_time : {float(bev['frame_time'])}")
