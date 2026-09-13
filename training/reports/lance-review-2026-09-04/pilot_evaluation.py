"""CPU-only pilot on fixed Lance intervals; no fitting or threshold promotion.

Caches decoded intervals locally so additional checkpoints and filtering
experiments can reuse exactly the same unfiltered evidence.
"""
from pathlib import Path
import csv
import hashlib
import json
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from rocklabel.config import load_config
from rocklabel.geometry.leveling import pin_level_to_labels
from rocklabel.labels import load_labels
from rocklabel.train import visual_audit as V

REPORT = ROOT / 'training/reports/lance-review-2026-09-04'
RECORDING = ROOT / 'recordings/archive/misc/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.mcap'
LABELS = ROOT / 'labels/archive/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json'
BASELINE = ROOT / 'training/experiments/deploy/cls-stray/trainall/best.pt'
BAND, CELL = (-.1, .6), .1


def main():
    torch.set_num_threads(2)
    labels = load_labels(str(LABELS))
    baseline = V._load_checkpoint(str(BASELINE), torch.device('cpu'))
    cfg = pin_level_to_labels(load_config(None), labels.level)
    settings = dict(start_s=150., end_s=450., stride=10, window_s=.05,
                    accum_seconds=5., candidates_per_rock=1, min_visible_points=15,
                    cell_m=CELL, floor_band=BAND, max_range=8.)
    identity = hashlib.sha256((LABELS.read_bytes() + json.dumps(settings, sort_keys=True).encode()
                               + json.dumps(baseline['generator'], sort_keys=True).encode()
                               + str(RECORDING.stat()).encode())).hexdigest()
    cache = ROOT / 'training/caches/lance-review-intervals.pkl'
    if cache.exists():
        with cache.open('rb') as f:
            saved = pickle.load(f)  # only this script writes this local cache
        if saved['identity'] != identity:
            raise RuntimeError('interval cache metadata changed; use a new cache path')
        intervals, floor = saved['intervals'], saved['floor']
    else:
        intervals, floor = V.collect_intervals(str(RECORDING), labels, str(LABELS), cfg,
                                               [baseline['generator']] * 2, **settings)
        with cache.open('wb') as f:
            pickle.dump(dict(identity=identity, intervals=intervals, floor=floor), f)
    print('Fixed intervals:', len(intervals), flush=True)
    device = torch.device('cpu')
    scores = {it.sequence: [V._score_interval(it, baseline, device, 64, CELL, labels)] for it in intervals}
    candidates = [('legacy-phantom-vb3', ROOT / 'training/experiments/stray/cls-phantom/loro_VolleyBallTest3.reslam/best.pt')]
    for path in sorted((ROOT / 'training/experiments/lance-campaign-v1').glob('*/seed-42/*/test_metrics.json')):
        arm, seed, fold = path.relative_to(ROOT / 'training/experiments/lance-campaign-v1').parts[:3]
        candidates.append((f'{arm}-{fold}', path.with_name('best.pt')))
    for name, path in candidates:
        out = REPORT / 'pilot' / name
        if (out / 'operating-points.csv').exists():
            continue
        print('Evaluate', name, flush=True)
        loaded = V._load_checkpoint(str(path), device)
        observations, pair_scores = [], {}
        for it in intervals:
            result = V._score_interval(it, loaded, device, 64, CELL, labels)
            pair_scores[it.sequence] = [scores[it.sequence][0], result]
            raw = np.concatenate([f.xyz for f in it.frames])
            for rock in labels.rocks:
                if it.visible_points.get(rock.id, 0) < 15:
                    continue
                row = dict(rock_id=rock.id, interval_sequence=it.sequence, start_s=it.start_s,
                           end_s=it.end_s, visible_points=it.visible_points[rock.id],
                           visible_cells=it.visible_cells[rock.id])
                for key, scored in zip(('model_a', 'model_b'), pair_scores[it.sequence]):
                    row[key] = V.rock_metrics(raw, rock, scored, CELL, .25) | dict(false_cells=scored.false_cells,
                                                                                  false_components=scored.false_components)
                observations.append(row)
        keys = ['model_a', 'model_b']
        per_rock = V._summarize_rocks(observations, keys, .25)
        cases = V.select_cases(observations, keys, 12)
        comparison = V.comparison_summary(observations, per_rock, keys, .2)
        by_seq = {it.sequence: it for it in intervals}
        (out / 'cases').mkdir(parents=True, exist_ok=True)
        for case in cases:
            it = by_seq[case['interval_sequence']]
            rock = labels.get(case['rock_id'])
            case['image'] = f'rock-{rock.id:03d}-interval-{it.sequence:04d}.png'
            V.render_case(str(out / 'cases' / case['image']), it, rock, labels, pair_scores[it.sequence],
                          [case[k] for k in keys], BAND, CELL)
        report_settings = settings | dict(recording=str(RECORDING), labels=str(LABELS), out=str(out),
                                           floor_z=floor, min_coverage=.25, material_gap=.2, device='cpu',
                                           filtering='none', unaudited_rock_ids=sorted({r.id for r in labels.rocks}-{r['rock_id'] for r in per_rock}))
        checkpoints = [{k: m[k] for k in ('path','name','task','threshold')} for m in [baseline, loaded]]
        V._write_reports(str(out), report_settings, checkpoints, observations, per_rock, cases, comparison, keys)
        rows = V.operating_point_rows(intervals, pair_scores, labels, CELL, .25, 15)
        with (out / 'operating-points.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        for seq, results in pair_scores.items():
            np.savez_compressed(out / f'cells-{seq:03d}.npz',
                                a_positions=results[0].cells.positions, a_probabilities=results[0].cells.probabilities,
                                b_positions=results[1].cells.positions, b_probabilities=results[1].cells.probabilities)
        with (out / 'summary.md').open('a') as f:
            f.write('\nPilot scope: 150–450 s, unfiltered inputs, visibility-rich intervals. '
                    f'Unaudited rocks: {report_settings["unaudited_rock_ids"]}. '
                    'The operating-point sweep is diagnostic and fitted on these evaluation data.\n')
        print('Wrote', out, flush=True)


if __name__ == '__main__':
    main()
