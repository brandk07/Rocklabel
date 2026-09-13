"""Descriptive campaign tables; no automatic winner or deployment selection."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import statistics

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
CAMPAIGN = ROOT/'training/experiments/lance-campaign-v1'


def write_csv(path, rows):
    if rows:
        with path.open('w') as f:
            w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)


def main():
    tasks = json.loads((CAMPAIGN/'manifest.json').read_text())['tasks']
    states = json.loads((CAMPAIGN/'campaign-status.json').read_text())
    training, audits = [], []
    for i, task in enumerate(tasks):
        metric_path = Path(task['directory'])/'test_metrics.json'
        if metric_path.exists():
            m = json.loads(metric_path.read_text())
            training.append(dict(task=i, arm=task['arm'], seed=task['seed'], fold=task['fold'],
                **{k:m[k] for k in ('pr_auc','precision','recall','f1','threshold')},
                checkpoint=str(metric_path.with_name('best.pt'))))
        path = CAMPAIGN/f'audit-{i:02d}'
        if i >= len(states) or states[i].get('audit') != 'complete':
            continue
        payload = json.loads((path/'audit.json').read_text())
        rows = list(csv.DictReader((path/'operating-points.csv').open()))
        rows = [{k:(v if k=='model' else float(v)) for k,v in r.items()} for r in rows]
        own = [min((r for r in rows if r['model']==key),
                   key=lambda r:abs(r['threshold']-payload['checkpoints'][j]['threshold']))
               for j,key in enumerate(('model_a','model_b'))]
        assert all(abs(r['threshold']-payload['checkpoints'][j]['threshold'])<1e-7
                   for j,r in enumerate(own)), 'stored threshold missing from sweep'
        budget = own[0]['mean_false_cells']
        allowed = [r for r in rows if r['model']=='model_b' and r['mean_false_cells']<=budget]
        oracle = max(allowed, key=lambda r:(r['macro_median_coverage'],
                     r['worst_rock_median_coverage'],-r['mean_false_cells'])) if allowed else None
        row = dict(task=i,arm=task['arm'],seed=task['seed'],fold=task['fold'],report=str(path))
        for prefix, metrics in [('reference',own[0]),('stored',own[1]),('oracle_at_reference_fp',oracle)]:
            row.update({f'{prefix}_{k}':metrics[k] if metrics else None for k in
                ('threshold','audited_rocks','macro_median_coverage','worst_rock_median_coverage',
                 'persistent_misses','mean_false_cells')})
        audits.append(row)
    write_csv(REPORT/'campaign-training.csv',training)
    write_csv(REPORT/'campaign-audits.csv',audits)
    grouped = defaultdict(list)
    for row in training: grouped[row['arm']].append(row['pr_auc'])
    summary = dict(training_complete=len(training),audits_complete=len(audits),
        volleyball_mean_ap={k:dict(n=len(v),mean=statistics.mean(v)) for k,v in grouped.items()},
        scope='Three volleyball folds and three seeds per arm. Lance audits use independent '
              'five-second accumulated maps selected across the recording, with equal physical-rock '
              'weight and legacy 3D attribution. Oracle thresholds are fitted on evaluation data. '
              'Tables do not certify visual inspection or select a winner.')
    (REPORT/'campaign-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
