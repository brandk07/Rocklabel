"""Full-recording sampled baseline accumulation and prediction-age audit.

Native classifier outputs, operational floor band, unfiltered input. No decay,
clearing or confirmation policy is deployed. Age alone does not prove a false
point is safe to clear. Output retains a frame cache for later map ablations.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from rocklabel.config import load_config
from rocklabel.geometry.leveling import pin_level_to_labels, check_level_match, level_record
from rocklabel.labels import load_labels
from rocklabel.dataset.labeling import points_in_rock, inside_arena, label_rocks, LABEL_CLEAR
from rocklabel.recording.pipeline import ScanStream, WindowedScanStream
from rocklabel.train import visual_audit as V
from rocklabel.train.mcapview import _score_balls

REPORT = ROOT/'training/reports/lance-review-2026-09-04'
BASELINE = ROOT/'training/experiments/deploy/cls-stray/trainall/best.pt'
LABELS = ROOT/'labels/archive/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json'
RECORDING = ROOT/'recordings/archive/misc/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.mcap'


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline-utc', type=float, required=True, help='Unix timestamp wall deadline')
    parser.add_argument('--checkpoint', type=Path, default=BASELINE)
    parser.add_argument('--out', type=Path, default=REPORT/'full-map')
    parser.add_argument('--frames-dir', type=Path, default=ROOT/'training/caches/lance-review-full-map')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; no audit started')
    out = args.out
    out.mkdir(exist_ok=False)
    frames_dir = args.frames_dir
    frames_dir.mkdir(exist_ok=False)
    labels = load_labels(str(LABELS))
    loaded = V._load_checkpoint(str(args.checkpoint), device)
    g = loaded['generator']
    cfg = pin_level_to_labels(load_config(None), labels.level)
    stream = ScanStream(str(RECORDING), cfg, stride=1, progress=False)
    check_level_match(labels.level, level_record(stream), str(LABELS))
    floor = getattr(getattr(stream, 'solution', None), 'floor_z', None)
    if floor is None:
        floor = labels.level.get('floor_z') if labels.level else None
    if floor is None:
        raise ValueError('measured floor required')
    settings = dict(recording=str(RECORDING), labels=str(LABELS), checkpoint=str(args.checkpoint),
        threshold=loaded['threshold'], generator=g, stride=10, window_s=.05, floor_band=[-.1,.6],
        floor_z=float(floor), max_range_m=8, cell_m=.1, filtering='none',
        inference_device=args.device, deadline_utc=args.deadline_utc)
    settings['audit_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    settings['visual_audit_source_sha256'] = hashlib.sha256(Path(V.__file__).read_bytes()).hexdigest()
    settings['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    atomic_json(out/'settings.json', settings)
    atomic_json(frames_dir/'settings.json', settings)
    accum = V.PredictionAccumulator(float(g['centers_voxel_m']))
    last_seen, raw_cells = {}, {r.id:set() for r in labels.rocks}
    timeline, start, origin, processed = [], time.monotonic(), None, 0
    status = 'complete'

    def measure(rel):
        cells = accum.reduce(.1)
        keep = inside_arena(cells.positions, labels.arena)
        cells = V.CellMap(cells.positions[keep],cells.probabilities[keep],cells.cell_ids[keep])
        hot = cells.probabilities >= loaded['threshold']
        lab = label_rocks(cells.positions,labels.rocks,float(g.get('boundary_shell_m',.05)))
        footprint_labels = V.projected_rock_labels(cells.positions,labels.rocks,float(g.get('boundary_shell_m',.05)))
        hot_ids = {tuple(k) for k in cells.cell_ids[hot]}
        false = hot & (lab == LABEL_CLEAR)
        # Keep original voxel keys: recomputing from float64 representatives
        # could round a float32 boundary return into a neighboring voxel.
        stamp_by_position = {tuple(value[0]):last_seen[key] for key,value in accum._map.items()}
        ages = np.array([rel-stamp_by_position[tuple(p)] for p in cells.positions])
        row = dict(time_s=rel, processed_frames=processed, native_voxels=len(accum._map),
            ground_cells=len(hot), false_cells=int(false.sum()),
            false_cells_older_10s=int((false & (ages>10)).sum()),
            false_cells_older_60s=int((false & (ages>60)).sum()),
            false_footprint_cells=int((hot & (footprint_labels==LABEL_CLEAR)).sum()),
            mean_false_age_s=float(ages[false].mean()) if false.any() else 0.)
        per_rock = []
        for rock in labels.rocks:
            on = hot & points_in_rock(cells.positions,rock)
            detected = {tuple(k) for k in cells.cell_ids[on]}
            visible = raw_cells[rock.id]
            per_rock.append(dict(rock_id=rock.id, visible_cells=len(visible),
                covered_cells=len(detected & visible),
                coverage=len(detected & visible)/len(visible) if visible else None,
                footprint_coverage=len(hot_ids & visible)/len(visible) if visible else None))
        seen = [r['coverage'] for r in per_rock if r['coverage'] is not None]
        row['audited_rocks'] = len(seen)
        row['macro_coverage'] = float(np.mean(seen)) if seen else None
        row['worst_rock_coverage'] = min(seen) if seen else None
        footprint_seen = [r['footprint_coverage'] for r in per_rock if r['footprint_coverage'] is not None]
        row['macro_footprint_coverage'] = float(np.mean(footprint_seen)) if footprint_seen else None
        row['worst_rock_footprint_coverage'] = min(footprint_seen) if footprint_seen else None
        return row,per_rock,cells,ages

    limits = V._crop_limits([g])
    with (out/'timeline.csv').open('w') as log:
        writer = None
        for k, scan in enumerate(WindowedScanStream(stream,.05)):
            if time.time() >= args.deadline_utc:
                status = 'budget_exhausted';break
            if k % 10:
                continue
            if origin is None:
                origin = float(scan.time_s)
            rel = float(scan.time_s)-origin
            frame = V._crop_frame(scan,limits,float(floor),(-.1,.6),8.)
            if frame is None:
                continue
            frame.time_s = rel
            rng = np.random.default_rng([int(g['seed']),frame.index])
            positions, probabilities = _score_balls(frame.xyz,frame.intensity,g,loaded['model'],device,rng,64)
            accum.update(positions,probabilities)
            for key in np.floor(positions/accum.voxel).astype(np.int64):
                last_seen[tuple(key)] = rel
            raw_rows = []
            for rock in labels.rocks:
                on = frame.xyz[points_in_rock(frame.xyz,rock)]
                cells = V.cell_set(on[:,:2],.1)
                raw_cells[rock.id].update(cells)
                raw_rows.extend((rock.id,*key) for key in cells)
            np.savez_compressed(frames_dir/f'frame-{processed:05d}.npz',
                index=frame.index,time_s=rel,base=frame.base,positions=positions,probabilities=probabilities,
                raw_rock_cells=np.array(raw_rows,dtype=np.int64).reshape(-1,3))
            processed += 1
            if processed%10 == 0:
                row,_,_,_ = measure(rel)
                timeline.append(row)
                if writer is None:
                    writer=csv.DictWriter(log,fieldnames=row);writer.writeheader()
                writer.writerow(row);log.flush()
                progress = row | dict(status='running',wall_seconds=time.monotonic()-start)
                atomic_json(out/'progress.json',progress)
                print(progress,flush=True)
    if not processed:
        atomic_json(out/'summary.json',dict(status=status,processed_frames=0))
        return
    row,per_rock,cells,ages=measure(rel)
    payload = dict(status=status,aggregate=row,per_rock=per_rock,wall_seconds=time.monotonic()-start,
        scope='Sampled full recording, unfiltered inputs, latest per native 3D voxel. '
              'False-cell ages are diagnostic; no visibility-aware clearing or traversability claim.')
    atomic_json(out/'summary.json',payload)
    np.savez_compressed(out/'map.npz',positions=cells.positions,probabilities=cells.probabilities,
                        cell_ids=cells.cell_ids,age_s=ages)
    with (out/'per-rock.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=per_rock[0]);writer.writeheader();writer.writerows(per_rock)
    print(payload,flush=True)


if __name__=='__main__':
    main()
