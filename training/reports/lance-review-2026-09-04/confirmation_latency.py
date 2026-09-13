"""Retrospective timing of footprint coverage from cached native predictions.

50%/80% are diagnostic milestones against each rock's full observed footprint,
not stopping-distance requirements. Distances are base-to-label-center in XY.
"""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from rocklabel.labels import load_labels

REPORT=Path(__file__).resolve().parent
CACHE=ROOT/'training/caches/lance-review-full-map'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=CACHE)
    parser.add_argument('--out',type=Path,default=REPORT/'confirmation-latency.csv')
    parser.add_argument('--reference',type=Path,default=REPORT/'full-map-ablation/native-latest.json')
    parser.add_argument('--observations',type=int,nargs='+',default=[1,2,3,5,8,12])
    args=parser.parse_args()
    if not args.observations or min(args.observations)<1:parser.error('positive observation counts required')
    settings=json.loads((args.cache/'settings.json').read_text());threshold=settings['threshold']
    voxel=float(settings['generator']['centers_voxel_m']);cell_m=float(settings['cell_m'])
    labels=load_labels(settings['labels']);paths=sorted(args.cache.glob('frame-*.npz'))
    if not paths:raise ValueError('no cached scoring frames')
    footprints={r.id:set() for r in labels.rocks}
    first_visible={}
    for path in paths:
        with np.load(path) as f:
            raw=f['raw_rock_cells'];t=float(f['time_s'])
            for rid in np.unique(raw[:,0]):
                cells={(int(x),int(y)) for _,x,y in raw[raw[:,0]==rid]}
                footprints[int(rid)].update(cells)
                if len(cells)>=3:first_visible.setdefault(int(rid),t)
    required=tuple(sorted(set([1,*args.observations])))
    native={};votes={};ground={n:{} for n in required}
    milestones={n:{} for n in required}
    for i,path in enumerate(paths):
        with np.load(path) as f:
            pos,prob=f['positions'],f['probabilities'];t=float(f['time_s']);base=f['base']
        unique={tuple(key):(p,float(v)) for key,p,v in zip(np.floor(pos/voxel).astype(np.int64),pos,prob)}
        for key,(p,v) in unique.items():
            cell=tuple(np.floor(p[:2].astype(float)/cell_m).astype(np.int64))
            old=native.get(key);old_votes=votes.get(key,0)
            count=old_votes+int(v>=threshold);votes[key]=count
            for n,mapping in ground.items():
                if old is not None and old[1]>=threshold and old_votes>=n:
                    mapping[old[0]]-=1
                if v>=threshold and count>=n:
                    mapping[cell]=mapping.get(cell,0)+1
            native[key]=(cell,v)
        for n,mapping in ground.items():
            for rock in labels.rocks:
                visible=footprints[rock.id]
                coverage=sum(mapping.get(cell,0)>0 for cell in visible)/max(len(visible),1)
                for fraction in (.5,.8):
                    if coverage>=fraction:
                        milestones[n].setdefault((rock.id,fraction),(t,float(np.linalg.norm(base[:2]-rock.center[:2]))))
        if i%500==0:print('Timing frame',i,flush=True)
    rows=[]
    for n,mapping in ground.items():
        for rock in labels.rocks:
            visible=footprints[rock.id]
            row=dict(required_positive_observations=n,rock_id=rock.id,
                first_three_raw_cells_s=first_visible.get(rock.id),
                final_footprint_coverage=sum(mapping.get(cell,0)>0 for cell in visible)/max(len(visible),1))
            for fraction in (.5,.8):
                t,d=milestones[n].get((rock.id,fraction),(None,None))
                base_t=milestones[1].get((rock.id,fraction),(None,None))[0]
                name=str(int(fraction*100))
                row[f'first_{name}pct_s']=t;row[f'base_distance_at_{name}pct_m']=d
                row[f'delay_vs_baseline_{name}pct_s']=t-base_t if t is not None and base_t is not None else None
            rows.append(row)
    # Reproduce independently computed final map measurements before using timing.
    reference=json.loads(args.reference.read_text())
    if reference.get('status','complete')!='complete':raise ValueError('reference map is incomplete')
    expected={r['rock_id']:r['footprint_coverage'] for r in reference['per_rock']}
    assert all(abs(r['final_footprint_coverage']-expected[r['rock_id']])<1e-9 for r in rows if r['required_positive_observations']==1)
    with args.out.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    print(f'Wrote timing for {len(required)} confirmation rules and {len(labels.rocks)} rocks; native map reproduced.',flush=True)


if __name__=='__main__':main()
