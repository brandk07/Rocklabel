"""Diagnostic temporal accumulation on unchanged, unfiltered Lance windows.

No model fitting, live setting change, or free-space inference. A rejected
single observation remains unconfirmed, not traversable. Intervals restart
the map, so these results do not measure full-run persistence or latency.
"""
from pathlib import Path
import csv
import hashlib
import json
import pickle
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from rocklabel.train import visual_audit as V
from rocklabel.train.mcapview import _score_balls
from rocklabel.labels import load_labels
from rocklabel.dataset.labeling import inside_arena, label_rocks, LABEL_CLEAR

REPORT = ROOT / 'training/reports/lance-review-2026-09-04'
BASELINE = ROOT / 'training/experiments/deploy/cls-stray/trainall/best.pt'
LABELS = ROOT / 'labels/archive/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json'
VARIANTS = ['latest-native', 'latest-ground-cell', 'two-observations-native',
            'two-observations-ground-cell', 'three-observations-ground-cell',
            'two-observations-near-010', 'two-observations-near-015']


def ground_map(positions, probabilities):
    accum = V.PredictionAccumulator(.05)
    accum.update(positions, probabilities)
    return accum.reduce(.1)


def accumulate(frames, variant):
    if '-near-' in variant:
        accum = V.PredictionAccumulator(.05)
        for positions, probabilities in frames:
            accum.update(positions, probabilities)
        if not accum._map:
            return accum.reduce(.1)
        positions = np.stack([v[0] for v in accum._map.values()])
        current = np.array([v[1] for v in accum._map.values()])
        radius = .1 if variant.endswith('010') else .15
        per_frame = []
        for pos, prob in frames:
            neighbors = cKDTree(pos).query_ball_point(positions, radius, workers=2)
            per_frame.append(np.array([prob[idx].max() if len(idx) else 0. for idx in neighbors]))
        if len(per_frame) < 2:
            return ground_map(positions, np.zeros(len(positions)))
        confidence = np.sort(np.stack(per_frame), axis=0)[-2]
        # Support can suppress a current prediction but cannot promote it.
        return ground_map(positions, np.minimum(current, confidence))
    if variant == 'latest-native':
        accum = V.PredictionAccumulator(.05)
        for positions, probabilities in frames:
            accum.update(positions, probabilities)
        return accum.reduce(.1)
    history = {}
    for positions, probabilities in frames:
        if variant.endswith('ground-cell'):
            cells = ground_map(positions, probabilities)
            ids, pos, prob = cells.cell_ids, cells.positions, cells.probabilities
        else:
            ids = np.floor(positions/.05).astype(np.int64)
            pos, prob = positions, probabilities
        # At most one entry from each frame: native duplicates follow latest.
        unique = {tuple(k): (p, float(v)) for k, p, v in zip(ids, pos, prob)}
        for key, value in unique.items():
            history.setdefault(key, []).append(value)
    positions, probabilities = [], []
    required = 3 if variant.startswith('three') else 2
    for values in history.values():
        if variant == 'latest-ground-cell':
            p, v = values[-1]
        elif len(values) >= required:
            # k-th largest frame confidence requires k distinct observations.
            # This is an evidence score, not a calibrated probability.
            p, v = sorted(values, key=lambda pair: pair[1], reverse=True)[required-1]
        else:
            continue
        positions.append(p); probabilities.append(v)
    if not positions:
        return V.CellMap(np.empty((0, 3)), np.empty(0), np.empty((0, 2), np.int64))
    return ground_map(np.asarray(positions), np.asarray(probabilities))


def result_for(cells, loaded, labels, name):
    keep = inside_arena(cells.positions, labels.arena)
    cells = V.CellMap(cells.positions[keep], cells.probabilities[keep], cells.cell_ids[keep])
    hot = cells.probabilities >= loaded['threshold']
    lab = label_rocks(cells.positions, labels.rocks, .05)
    false = {tuple(k) for k in cells.cell_ids[hot & (lab == LABEL_CLEAR)]}
    return V.ModelResult(loaded['path'], name, 'classify', loaded['threshold'], cells,
                         {tuple(k) for k in cells.cell_ids[hot]}, len(false),
                         V.connected_components(false))


def main():
    torch.set_num_threads(2)
    device = torch.device('cpu')
    labels = load_labels(str(LABELS))
    cache_path = ROOT / 'training/caches/lance-review-intervals.pkl'
    with cache_path.open('rb') as f:
        saved = pickle.load(f)  # trusted cache created in this review
    intervals = saved['intervals']
    loaded = V._load_checkpoint(str(BASELINE), device)
    identity = hashlib.sha256(BASELINE.read_bytes() + saved['identity'].encode()).hexdigest()
    cache = ROOT / 'training/caches/lance-review-native-scores.pkl'
    if cache.exists():
        with cache.open('rb') as f:
            native = pickle.load(f)
        if native['identity'] != identity:
            raise RuntimeError('native-score cache identity changed')
        scores = native['scores']
    else:
        scores = {}
        for interval in intervals:
            print('Score interval', interval.sequence, flush=True)
            scores[interval.sequence] = []
            for frame in interval.frames:
                g = loaded['generator']
                rng = np.random.default_rng([int(g['seed']), frame.index])
                scores[interval.sequence].append(_score_balls(
                    frame.xyz, frame.intensity, g, loaded['model'], device, rng, 64))
        with cache.open('wb') as f:
            pickle.dump(dict(identity=identity, scores=scores), f)
    out = REPORT / 'temporal-ablation'
    out.mkdir(exist_ok=True)
    results, observations, summary = {}, {}, []
    for name in VARIANTS:
        results[name], observations[name] = {}, []
        for interval in intervals:
            result = result_for(accumulate(scores[interval.sequence], name), loaded, labels, name)
            results[name][interval.sequence] = result
            raw = np.concatenate([f.xyz for f in interval.frames])
            for rock in labels.rocks:
                if interval.visible_points.get(rock.id, 0) < 15:
                    continue
                observations[name].append(dict(rock_id=rock.id, interval_sequence=interval.sequence,
                    start_s=interval.start_s, end_s=interval.end_s,
                    visible_points=interval.visible_points[rock.id], visible_cells=interval.visible_cells[rock.id],
                    model_b=V.rock_metrics(raw, rock, result, .1, .25) |
                        dict(false_cells=result.false_cells, false_components=result.false_components)))
        rows = V._summarize_rocks(observations[name], ['model_b'], .25)
        row = dict(accumulation=name,
            macro_median_coverage=float(np.mean([r['model_b']['median_coverage'] for r in rows])),
            worst_rock_median_coverage=min(r['model_b']['median_coverage'] for r in rows),
            persistent_misses=sum(not r['model_b']['detected_any'] for r in rows),
            mean_false_cells=float(np.mean([r.false_cells for r in results[name].values()])))
        summary.append(row)
        (out/f'{name}.json').write_text(json.dumps(dict(aggregate=row, per_rock=rows,
                                                      observations=observations[name]), indent=2))
        print(row, flush=True)
        if name == VARIANTS[0]:
            continue
        for observation, baseline in zip(observations[name], observations[VARIANTS[0]]):
            observation['model_a'] = baseline['model_b']
        cases = V.select_cases(observations[name], ['model_a', 'model_b'], 12)
        case_dir = out/name/'cases'
        case_dir.mkdir(parents=True, exist_ok=True)
        by_seq = {it.sequence: it for it in intervals}
        for case in cases:
            it = by_seq[case['interval_sequence']]
            V.render_case(str(case_dir/f"rock-{case['rock_id']:03d}-interval-{it.sequence:04d}.png"),
                it, labels.get(case['rock_id']), labels,
                [results[VARIANTS[0]][it.sequence], results[name][it.sequence]],
                [case['model_a'], case['model_b']], (-.1, .6), .1)
    with (out/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0]); writer.writeheader(); writer.writerows(summary)


if __name__ == '__main__':
    main()
