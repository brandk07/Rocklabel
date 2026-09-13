"""Replay cached native scores through diagnostic map-update policies.

All variants use identical model outputs and unfiltered rock-cell denominators.
Confirmation/mean scores are not calibrated probabilities or free-space tests.
"""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from rocklabel.train import visual_audit as V
from rocklabel.labels import load_labels
from rocklabel.dataset.labeling import points_in_rock, inside_arena, label_rocks, LABEL_CLEAR

REPORT=Path(__file__).resolve().parent
CACHE=ROOT/'training/caches/lance-review-full-map'


def summarize(name,positions,probabilities,labels,visible,threshold):
    accum=V.PredictionAccumulator(.05);accum.update(positions,probabilities)
    cells=accum.reduce(.1)
    keep=inside_arena(cells.positions,labels.arena)
    cells=V.CellMap(cells.positions[keep],cells.probabilities[keep],cells.cell_ids[keep])
    hot=cells.probabilities>=threshold
    lab=label_rocks(cells.positions,labels.rocks,.05)
    footprint_labels=V.projected_rock_labels(cells.positions,labels.rocks,.05)
    hot_ids={tuple(k) for k in cells.cell_ids[hot]}
    rows=[]
    for rock in labels.rocks:
        pred={tuple(k) for k in cells.cell_ids[hot & points_in_rock(cells.positions,rock)]}
        covered=pred & visible[rock.id]
        rows.append(dict(rock_id=rock.id,visible_cells=len(visible[rock.id]),covered_cells=len(covered),
                         coverage=len(covered)/max(len(visible[rock.id]),1),fragments=V.connected_components(covered),
                         footprint_coverage=len(hot_ids & visible[rock.id])/max(len(visible[rock.id]),1)))
    summary=dict(policy=name,macro_coverage=float(np.mean([r['coverage'] for r in rows])),
                 worst_rock_coverage=min(r['coverage'] for r in rows),
                 rocks_below_25pct=sum(r['coverage']<.25 for r in rows),
                 false_cells=int((hot & (lab==LABEL_CLEAR)).sum()),
                 macro_footprint_coverage=float(np.mean([r['footprint_coverage'] for r in rows])),
                 worst_rock_footprint_coverage=min(r['footprint_coverage'] for r in rows),
                 false_footprint_cells=int((hot & (footprint_labels==LABEL_CLEAR)).sum()))
    return summary,rows,cells


def main():
    settings=json.loads((CACHE/'settings.json').read_text())
    threshold=settings['threshold'];labels=load_labels(settings['labels'])
    native=V.PredictionAccumulator(.05)
    native_hits={};ground={};hysteresis={n:{} for n in (1,2,3)}
    visible={r.id:set() for r in labels.rocks}
    for i,path in enumerate(sorted(CACHE.glob('frame-*.npz'))):
        with np.load(path) as f:
            pos,prob=f['positions'],f['probabilities']
            for rid,x,y in f['raw_rock_cells']:
                visible[int(rid)].add((int(x),int(y)))
        native.update(pos,prob)
        unique={tuple(key):float(p) for key,p in zip(np.floor(pos/.05).astype(np.int64),prob)}
        for key,p in unique.items():
            native_hits[key]=native_hits.get(key,0)+int(p>=threshold)
        one=V.PredictionAccumulator(.05);one.update(pos,prob);cells=one.reduce(.1)
        for key,xyz,p in zip(map(tuple,cells.cell_ids),cells.positions,cells.probabilities):
            old=ground.get(key)
            if old is None:
                count,total,ema=1,p,p
            else:
                count,total,ema=old[2]+1,old[3]+p,.75*old[4]+.25*p
            ground[key]=(xyz,float(p),count,total,ema)
            for required,mapping in hysteresis.items():
                # Positive votes need not be consecutive; a clear observation
                # sequence resets them. Unobserved cells are left unchanged.
                votes,clear,confirmed,location= mapping.get(key,(0,0,False,xyz))
                if p>=threshold:
                    votes+=1;clear=0;location=xyz
                    confirmed=confirmed or votes>=required
                elif p<.2:
                    clear+=1
                    if clear>=3:
                        votes=0;confirmed=False
                else:
                    clear=0
                mapping[key]=(votes,clear,confirmed,location)
        if i%500==0:print('Cached frames',i,flush=True)
    out=REPORT/'full-map-ablation';out.mkdir(exist_ok=True)
    variants={}
    values=list(native._map.items())
    positions=np.stack([v[0] for _,v in values]);probabilities=np.array([v[1] for _,v in values])
    variants['native-latest']=(positions,probabilities)
    for n in (2,3):
        variants[f'native-{n}-positive-observations']=(positions,np.where(
            np.array([native_hits[k]>=n for k,_ in values]),probabilities,0.))
    values=list(ground.values());positions=np.stack([v[0] for v in values])
    variants['ground-latest']=(positions,np.array([v[1] for v in values]))
    variants['ground-mean']=(positions,np.array([v[3]/v[2] for v in values]))
    variants['ground-ema-025']=(positions,np.array([v[4] for v in values]))
    for n,mapping in hysteresis.items():
        values=list(mapping.values())
        variants[f'ground-hysteresis-{n}-on-3-clear']=(np.stack([v[3] for v in values]),
                                                    np.array([float(v[2]) for v in values]))
    summary=[]
    for name,(pos,prob) in variants.items():
        row,rocks,cells=summarize(name,pos,prob,labels,visible,threshold)
        summary.append(row)
        (out/f'{name}.json').write_text(json.dumps(dict(aggregate=row,per_rock=rocks),indent=2))
        np.savez_compressed(out/f'{name}.npz',positions=cells.positions,probabilities=cells.probabilities,
                            cell_ids=cells.cell_ids)
        print(row,flush=True)
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)


if __name__=='__main__':main()
