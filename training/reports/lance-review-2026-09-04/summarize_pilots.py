"""Refresh a descriptive screening table from completed pilot reports only."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPORT = Path(__file__).resolve().parent


def main():
    summaries = []
    for path in sorted((REPORT/'pilot').glob('*/operating-points.csv')):
        payload = json.loads(path.with_name('audit.json').read_text())
        rows = list(csv.DictReader(path.open()))
        for row in rows:
            for k in row:
                if k != 'model':
                    row[k] = float(row[k])
        own = []
        for i, key in enumerate(['model_a', 'model_b']):
            threshold = payload['checkpoints'][i]['threshold']
            own.append(min([r for r in rows if r['model'] == key],
                           key=lambda r: abs(r['threshold']-threshold)))
        budget = own[0]['mean_false_cells']
        allowed = [r for r in rows if r['model']=='model_b' and r['mean_false_cells'] <= budget]
        oracle = max(allowed, key=lambda r: (r['macro_median_coverage'],
                         r['worst_rock_median_coverage'], -r['mean_false_cells']))
        summary = dict(candidate=path.parent.name, checkpoint=payload['checkpoints'][1]['path'])
        for prefix, row in [('stored', own[1]), ('oracle_at_reference_fp', oracle)]:
            summary.update({f'{prefix}_{k}':row[k] for k in ['threshold','macro_median_coverage',
                  'worst_rock_median_coverage','mean_false_cells','persistent_misses']})
        summaries.append(summary)
    if not summaries:
        return
    with (REPORT/'pilot-screening.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=summaries[0]);writer.writeheader();writer.writerows(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, measure in zip(axes, ['macro_median_coverage', 'worst_rock_median_coverage']):
        ax.scatter([own[0]['mean_false_cells']], [own[0][measure]*100], marker='*',
                   c='black',s=170,label='Reference stray trainall',zorder=5)
        for row in summaries:
            name=row['candidate']
            if name=='legacy-phantom-vb3':
                label='Legacy phantom VB3';color='gray'
            else:
                arm,fold=name.split('-pointnet',1)
                vb=fold.split('VolleyBallTest')[1].split('.')[0]
                label=f'{arm} VB{vb}'
                color={'pointnet-stray':'#0072B2','pointnet-matched':'#009E73',
                       'stats-stray':'#D55E00','stats-matched':'#CC79A7'}[arm]
            ax.scatter(row['stored_mean_false_cells'],row[f'stored_{measure}']*100,
                       c=color,s=35)
            ax.annotate(label,(row['stored_mean_false_cells'],row[f'stored_{measure}']*100),
                        xytext=(4,3),textcoords='offset points',fontsize=6)
        ax.set_xlabel('Mean false-positive 10 cm cells / selected interval')
        ax.set_ylabel(('Mean of per-rock medians' if measure.startswith('macro') else
                       'Worst rock median')+' coverage (%)')
        ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Lance pilot: stored thresholds, 9 rocks, 5 visibility-rich intervals\n'
                 '150–450 s excerpt; full floor band; unfiltered; no deployment selection',fontsize=11)
    fig.savefig(REPORT/'pilot-screening.png',dpi=160)
    fig.savefig(REPORT/'pilot-screening.svg')
    plt.close(fig)
    print(f'Wrote {len(summaries)} completed candidates; oracle columns fit these audit data.')


if __name__ == '__main__':
    main()
