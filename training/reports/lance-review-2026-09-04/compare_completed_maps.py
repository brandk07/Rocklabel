"""Compare completed full-recording audits and their cached detection timing.

Read-only with respect to models and deployment. Same-recording development
measurements, never an automatic winner selection. Can watch queued audits.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from rocklabel.labels import load_labels
from rocklabel.train import visual_audit as V

REPORT=Path(__file__).resolve().parent
BASE=REPORT/'full-map'
BASE_CACHE=ROOT/'training/caches/lance-review-full-map'
OUT=REPORT/'completed-map-comparison'


def write_csv(path,rows):
    if rows:
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def geometry(cache):
    digest=hashlib.sha256();footprints={}
    paths=sorted(cache.glob('frame-*.npz'))
    for p in paths:
        with np.load(p) as f:
            for key in ('index','time_s','base','raw_rock_cells'):
                a=f[key];digest.update(key.encode());digest.update(str(a.shape).encode());digest.update(a.tobytes())
            for rid,x,y in f['raw_rock_cells']:
                footprints.setdefault(int(rid),set()).add((int(x),int(y)))
    return dict(sha256=digest.hexdigest(),frames=len(paths)),footprints


def planned():
    for seed in (42,43):
        for arm in ('pointnet-stray','pointnet-matched'):
            name=f'{arm}-{seed}'
            yield name,ROOT/f'training/experiments/lance-finalists-v1/full-map-{name}',ROOT/f'training/caches/lance-finalists-{name}','volleyball_only'
    for arm in ('control','adapt'):
        yield f'target-{arm}',ROOT/f'training/experiments/lance-target-negatives-v1/full-map-{arm}',ROOT/f'training/caches/lance-target-{arm}','same_recording_negative_adaptation_trial'


def operating_points(positions,prob,ids,threshold,labels,footprints,shell,cell):
    lab=V.projected_rock_labels(positions,labels.rocks,shell)
    keys=[tuple(k) for k in ids]
    thresholds=np.unique(np.r_[threshold,np.linspace(.01,.99,99),1-10.**np.linspace(-6,-2,9),10.**np.linspace(-6,-2,9)])
    rows=[]
    for t in thresholds:
        hot=prob>=t;occupied={k for k,h in zip(keys,hot) if h}
        coverage=[len(occupied & footprint)/len(footprint) for footprint in footprints.values()]
        false=int((hot & (lab==0)).sum())
        rows.append(dict(threshold=float(t),macro_footprint_coverage=float(np.mean(coverage)),
            worst_rock_footprint_coverage=min(coverage),false_footprint_cells=false,false_area_m2=false*cell**2))
    stored=next(r for r in rows if r['threshold']==threshold)
    return rows,stored


def native_timing(cache,reference,out):
    if not out.exists():
        cmd=[sys.executable,str(REPORT/'confirmation_latency.py'),'--cache',str(cache),
             '--reference',str(reference),'--out',str(out),'--observations','1']
        with out.with_suffix('.log').open('w') as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    return {int(r['rock_id']):r for r in csv.DictReader(out.open())}


def compare(name,audit,cache,scope,base_geometry,footprints,base_timing):
    output=OUT/name;output.mkdir(exist_ok=True)
    settings=json.loads((audit/'settings.json').read_text())
    reference_settings=json.loads((BASE/'settings.json').read_text())
    for k in ('recording','labels','stride','window_s','floor_band','floor_z','max_range_m','cell_m','filtering'):
        if settings[k]!=reference_settings[k]:raise ValueError(f'{name}: different operational setting {k}')
    identity,own_footprints=geometry(cache)
    if identity!=base_geometry or own_footprints!=footprints:
        raise ValueError(f'{name}: different frames, poses or observed rock geometry')
    labels=load_labels(settings['labels']);cell=float(settings['cell_m']);threshold=float(settings['threshold'])
    summary=json.loads((audit/'summary.json').read_text())
    timing=native_timing(cache,audit/'summary.json',output/'timing.csv')
    reference=json.loads((REPORT/'full-map-ablation/native-latest.json').read_text())
    with np.load(BASE/'map.npz') as f:bp,bv,bi=f['positions'],f['probabilities'],f['cell_ids']
    with np.load(audit/'map.npz') as f:pos,prob,ids=f['positions'],f['probabilities'],f['cell_ids']
    shell=float(settings['generator'].get('boundary_shell_m',.05))
    base_shell=float(reference_settings['generator'].get('boundary_shell_m',.05))
    if shell!=base_shell:raise ValueError('different attribution boundary shells')
    _,base_point=operating_points(bp,bv,bi,float(reference_settings['threshold']),labels,footprints,shell,cell)
    sweep,point=operating_points(pos,prob,ids,threshold,labels,footprints,shell,cell)
    for k in ('macro_footprint_coverage','worst_rock_footprint_coverage','false_footprint_cells'):
        if abs(point[k]-summary['aggregate'][k])>1e-9:raise ValueError(f'independent map metric mismatch: {k}')
    allowed=[r for r in sweep if r['false_footprint_cells']<=base_point['false_footprint_cells']]
    oracle=max(allowed,key=lambda r:(r['macro_footprint_coverage'],r['worst_rock_footprint_coverage'],-r['false_footprint_cells'])) if allowed else None
    write_csv(output/'operating-points.csv',sweep)
    per=[]
    old={r['rock_id']:r for r in reference['per_rock']}
    for r in summary['per_rock']:
        rid=r['rock_id'];row=dict(rock_id=rid,visible_cells=r['visible_cells'],
            reference_3d_coverage=old[rid]['coverage'],candidate_3d_coverage=r['coverage'],
            reference_footprint_coverage=old[rid]['footprint_coverage'],candidate_footprint_coverage=r['footprint_coverage'])
        for milestone in (50,80):
            a,b=base_timing[rid][f'first_{milestone}pct_s'],timing[rid][f'first_{milestone}pct_s']
            row.update({f'reference_first_{milestone}pct_s':float(a) if a else None,
                        f'candidate_first_{milestone}pct_s':float(b) if b else None,
                        f'delay_vs_reference_{milestone}pct_s':float(b)-float(a) if a and b else None})
        per.append(row)
    write_csv(output/'per-rock.csv',per)
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    for ax,title,p,v,t,m in [(axes[0],'Existing stray train-all',bp,bv,reference_settings['threshold'],base_point),
                            (axes[1],name,pos,prob,threshold,point)]:
        lab=V.projected_rock_labels(p,labels.rocks,shell);hot=v>=t
        ax.scatter(bp[:,0],bp[:,1],s=2,c='#dddddd',linewidths=0)
        for code,color in [(0,'#d55e00'),(-1,'#0072b2'),(1,'#009e73')]:
            keep=hot & (lab==code);ax.scatter(p[keep,0],p[keep,1],s=5,c=color,linewidths=0)
        for rock in labels.rocks:
            V._rock_outline(ax,rock,'#222222',.8);ax.text(*rock.center[:2],str(rock.id),fontsize=7,ha='center')
        ax.set_aspect('equal');ax.set(xlabel='World x (m)',ylabel='World y (m)',
            title=f"{title} · threshold {t:.3f}\nFalse area {m['false_area_m2']:.2f} m²; mean/worst footprint {m['macro_footprint_coverage']:.1%}/{m['worst_rock_footprint_coverage']:.1%}")
    fig.suptitle('Final accumulated Lance map · identical 2,599 sampled windows · full floor band\nOrange: positive outside rock footprint; green: inside; blue: shell; grey: other scored cells, not verified free space',fontsize=10)
    fig.savefig(output/'map.png',dpi=160);fig.savefig(output/'map.svg');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained');x=np.arange(len(per))
    for ax,metric,title in [(axes[0],'footprint','Occupied XY footprint'),(axes[1],'3d','3D-attributed rock surface')]:
        ax.bar(x-.18,[r[f'reference_{metric}_coverage']*100 for r in per],.36,label='Reference')
        ax.bar(x+.18,[r[f'candidate_{metric}_coverage']*100 for r in per],.36,label=name)
        ax.set(xticks=x,xticklabels=[r['rock_id'] for r in per],ylim=(0,105),ylabel='Coverage (%)',xlabel='Physical rock ID',title=title);ax.legend(fontsize=8)
    fig.savefig(output/'per-rock.png',dpi=160);plt.close(fig)
    result=dict(candidate=name,scope=scope,checkpoint=settings['checkpoint'],
        checkpoint_sha256=settings['checkpoint_sha256'],geometry=identity,reference=base_point,stored=point,
        oracle_at_reference_fp=oracle,
        note='Oracle threshold is fitted on this evaluation recording. Final footprint occupancy can be supplied by a floating positive above a rock. Review per-rock surface coverage, timing, and images before interpreting the aggregate.',
        comparison_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    base_geometry,footprints=geometry(BASE_CACHE)
    base_timing=native_timing(BASE_CACHE,REPORT/'full-map-ablation/native-latest.json',OUT/'baseline-timing.csv')
    end=datetime(2026,9,5,13,0,tzinfo=timezone.utc).timestamp()
    while True:
        results=[];states=[]
        for name,audit,cache,scope in planned():
            p=audit/'summary.json';output=OUT/name/'comparison.json'
            status=json.loads(p.read_text()).get('status') if p.exists() else 'pending'
            states.append(dict(candidate=name,audit=status))
            if status!='complete':continue
            if output.exists():result=json.loads(output.read_text())
            else:
                print('Compare',name,flush=True)
                result=compare(name,audit,cache,scope,base_geometry,footprints,base_timing)
            results.append(result)
        (OUT/'status.json').write_text(json.dumps(dict(candidates=states,comparisons_complete=len(results)),indent=2)+'\n')
        table=[dict(candidate=r['candidate'],scope=r['scope'],**r['stored']) for r in results]
        write_csv(OUT/'stored-thresholds.csv',table)
        if not args.watch or time.time()>=end or len(results)==6:break
        time.sleep(30)


if __name__=='__main__':main()
