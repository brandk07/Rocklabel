"""Offline filtering ablations with unfiltered rock geometry as denominator.

These operate on fixed cropped scoring windows, not the live pre-ingest
batches. They are diagnostics and do not change a deployed filter setting.
"""
from pathlib import Path
import csv
from dataclasses import replace
import json
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from rocklabel.train import visual_audit as V
from rocklabel.labels import load_labels
from rocklabel.dataset.labeling import points_in_rock
from rocklabel.live.filters import floating_keep_mask, statistical_outlier_mask

REPORT = ROOT / 'training/reports/lance-review-2026-09-04'
LABELS = ROOT / 'labels/archive/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json'
BASELINE = ROOT / 'training/experiments/deploy/cls-stray/trainall/best.pt'
VARIANTS = ['unfiltered', 'floating-050', 'radius-010-3', 'radius-015-3', 'previous-frame-010']


def keep_points(name, frame, previous):
    xyz = frame.xyz
    if name == 'unfiltered':
        return np.ones(len(xyz), dtype=bool)
    if name == 'floating-050':
        return floating_keep_mask(xyz, cell_size=1., max_height=.5)
    if name.startswith('radius'):
        radius = .1 if name == 'radius-010-3' else .15
        counts = cKDTree(xyz).query_ball_point(xyz, radius, return_length=True, workers=2)
        return counts >= 4  # self plus at least three other raw returns
    if previous is None:
        return np.ones(len(xyz), dtype=bool)  # explicit warm-up passthrough
    distances, _ = cKDTree(previous.xyz).query(xyz, distance_upper_bound=.1, workers=2)
    return np.isfinite(distances)


def main():
    torch.set_num_threads(2)
    labels = load_labels(str(LABELS))
    with (ROOT / 'training/caches/lance-review-intervals.pkl').open('rb') as f:
        saved = pickle.load(f)  # trusted local cache written by pilot_evaluation.py
    intervals = saved['intervals']
    loaded = V._load_checkpoint(str(BASELINE), torch.device('cpu'))
    results = {}
    aggregate = []
    out = REPORT / 'filter-ablation'
    out.mkdir(exist_ok=True)
    for name in VARIANTS:
        print('Filter', name, flush=True)
        results[name] = {}
        observations = []
        retention = {}
        total_raw = total_kept = 0
        for interval in intervals:
            frames = []
            previous = None
            for frame in interval.frames:
                keep = keep_points(name, frame, previous)
                previous = frame
                total_raw += len(keep); total_kept += int(keep.sum())
                frames.append(replace(frame, xyz=frame.xyz[keep], intensity=frame.intensity[keep]))
            filtered = replace(interval, frames=frames)
            result = V._score_interval(filtered, loaded, torch.device('cpu'), 64, .1, labels)
            results[name][interval.sequence] = result
            raw = np.concatenate([f.xyz for f in interval.frames])
            kept = np.concatenate([f.xyz for f in frames])
            for rock in labels.rocks:
                if interval.visible_points.get(rock.id, 0) < 15:
                    continue
                raw_on = raw[points_in_rock(raw, rock)]
                kept_on = kept[points_in_rock(kept, rock)]
                raw_cells = V.cell_set(raw_on[:, :2], .1)
                kept_cells = V.cell_set(kept_on[:, :2], .1)
                retention.setdefault(rock.id, []).append((len(kept_on)/len(raw_on), len(kept_cells & raw_cells)/len(raw_cells)))
                metric = V.rock_metrics(raw, rock, result, .1, .25)
                observations.append(dict(rock_id=rock.id, interval_sequence=interval.sequence,
                                         start_s=interval.start_s,end_s=interval.end_s,
                                         visible_points=interval.visible_points[rock.id],
                                         visible_cells=interval.visible_cells[rock.id],
                                         model_b=metric | dict(false_cells=result.false_cells,false_components=result.false_components)))
        rows = V._summarize_rocks(observations, ['model_b'], .25)
        row = dict(filter=name, raw_return_retention=total_kept/total_raw,
                   macro_rock_return_retention=np.mean([np.mean(v,axis=0)[0] for v in retention.values()]),
                   worst_rock_cell_retention=min(np.min(v,axis=0)[1] for v in retention.values()),
                   macro_median_coverage=np.mean([r['model_b']['median_coverage'] for r in rows]),
                   worst_rock_median_coverage=min(r['model_b']['median_coverage'] for r in rows),
                   persistent_misses=sum(not r['model_b']['detected_any'] for r in rows),
                   mean_false_cells=np.mean([r.false_cells for r in results[name].values()]))
        aggregate.append(row)
        (out / f'{name}.json').write_text(json.dumps(dict(aggregate=row,per_rock=rows,retention=retention,observations=observations),indent=2))
        print(row,flush=True)
        if name != 'unfiltered':
            for observation in observations:
                raw_row = next(r for r in baseline_observations if (r['rock_id'],r['interval_sequence']) == (observation['rock_id'],observation['interval_sequence']))
                observation['model_a'] = raw_row['model_b']
            cases = V.select_cases(observations,['model_a','model_b'],12)
            case_dir=out/name/'cases';case_dir.mkdir(parents=True,exist_ok=True)
            by_seq={i.sequence:i for i in intervals}
            for case in cases:
                it=by_seq[case['interval_sequence']]
                V.render_case(str(case_dir/f"rock-{case['rock_id']:03d}-interval-{it.sequence:04d}.png"),
                              it,labels.get(case['rock_id']),labels,
                              [results['unfiltered'][it.sequence],results[name][it.sequence]],
                              [case['model_a'],case['model_b']],(-.1,.6),.1)
        else:
            baseline_observations=observations
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=aggregate[0]);writer.writeheader();writer.writerows(aggregate)
    print('Wrote',out,flush=True)


if __name__ == '__main__':
    main()
