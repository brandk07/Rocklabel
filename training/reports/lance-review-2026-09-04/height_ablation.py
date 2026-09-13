"""Isolate sensitivity to a few low returns in classifier normalization.

Diagnostic input-contract changes on a frozen checkpoint, not deployment
settings. Original raw rock geometry remains the coverage denominator.
"""
from pathlib import Path
import csv
import json
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rocklabel.train import visual_audit as V
from rocklabel.labels import load_labels
from temporal_ablation import accumulate, result_for, REPORT, BASELINE, LABELS


class RobustHeight(torch.nn.Module):
    def __init__(self, model, fraction, clamp):
        super().__init__()
        self.model, self.fraction, self.clamp = model, fraction, clamp

    def forward(self, points, counts):
        n = points.shape[1]
        real = counts.clamp(1, n).long()
        valid = torch.arange(n, device=points.device)[None] < real[:, None]
        sorted_z = points[..., 2].masked_fill(~valid, torch.inf).sort(dim=1).values
        index = ((real-1)*self.fraction).long()
        reference = sorted_z.gather(1, index[:, None])
        transformed = points.clone()
        z = points[..., 2] - reference
        transformed[..., 2] = z.clamp_min(0) if self.clamp else z
        return self.model(transformed, counts)


def main():
    torch.set_num_threads(2)
    device = torch.device('cpu')
    labels = load_labels(str(LABELS))
    loaded = V._load_checkpoint(str(BASELINE), device)
    with (ROOT/'training/caches/lance-review-intervals.pkl').open('rb') as f:
        intervals = pickle.load(f)['intervals']
    with (ROOT/'training/caches/lance-review-native-scores.pkl').open('rb') as f:
        native = pickle.load(f)['scores']
    baseline = {it.sequence: result_for(accumulate(native[it.sequence], 'latest-native'),
                                       loaded, labels, 'original-minimum') for it in intervals}
    out = REPORT/'height-ablation'
    out.mkdir(exist_ok=True)
    summary = []
    for fraction, clamp in [(0.01, False), (.05, False), (.05, True)]:
        name = f'height-p{fraction*100:02.0f}' + ('-clamped' if clamp else '')
        print('Evaluate', name, flush=True)
        candidate = loaded | dict(model=RobustHeight(loaded['model'], fraction, clamp).eval(), name=name)
        results, observations = {}, []
        for it in intervals:
            result = V._score_interval(it, candidate, device, 64, .1, labels)
            results[it.sequence] = [baseline[it.sequence], result]
            raw = np.concatenate([f.xyz for f in it.frames])
            for rock in labels.rocks:
                if it.visible_points.get(rock.id, 0) < 15:
                    continue
                row = dict(rock_id=rock.id, interval_sequence=it.sequence, start_s=it.start_s,
                           end_s=it.end_s, visible_points=it.visible_points[rock.id],
                           visible_cells=it.visible_cells[rock.id])
                for key, scored in zip(['model_a', 'model_b'], results[it.sequence]):
                    row[key] = V.rock_metrics(raw, rock, scored, .1, .25) | dict(
                        false_cells=scored.false_cells, false_components=scored.false_components)
                observations.append(row)
        rows = V._summarize_rocks(observations, ['model_a', 'model_b'], .25)
        aggregate = dict(normalization=name,
            macro_median_coverage=float(np.mean([r['model_b']['median_coverage'] for r in rows])),
            worst_rock_median_coverage=min(r['model_b']['median_coverage'] for r in rows),
            mean_false_cells=float(np.mean([r[1].false_cells for r in results.values()])))
        summary.append(aggregate)
        (out/f'{name}.json').write_text(json.dumps(dict(aggregate=aggregate, per_rock=rows,
                                                     observations=observations), indent=2))
        print(aggregate, flush=True)
        cases = V.select_cases(observations, ['model_a', 'model_b'], 12)
        case_dir = out/name/'cases'; case_dir.mkdir(parents=True, exist_ok=True)
        by_seq = {it.sequence: it for it in intervals}
        for case in cases:
            it = by_seq[case['interval_sequence']]
            V.render_case(str(case_dir/f"rock-{case['rock_id']:03d}-interval-{it.sequence:04d}.png"),
                it, labels.get(case['rock_id']), labels, results[it.sequence],
                [case['model_a'],case['model_b']], (-.1, .6), .1)
        sweep = V.operating_point_rows(intervals, results, labels, .1, .25, 15)
        with (out/name/'operating-points.csv').open('w') as f:
            writer=csv.DictWriter(f, fieldnames=sweep[0]); writer.writeheader(); writer.writerows(sweep)
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=summary[0]); writer.writeheader(); writer.writerows(summary)


if __name__ == '__main__':
    main()
