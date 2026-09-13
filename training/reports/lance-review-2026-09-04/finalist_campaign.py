"""Paired full-volleyball fits and whole-recording Lance audits, within the
original 12-hour deadline. Waits for the existing sweep; never stops another
process, changes a deployment checkpoint, or trains on Lance.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from rocklabel.train.data import run_dir_name
from rocklabel.train.lance_campaign import run_child

REPORT = Path(__file__).resolve().parent
SWEEP = ROOT/'training/experiments/lance-campaign-v1'
FINAL = ROOT/'training/experiments/lance-finalists-v1'
DEADLINE = datetime(2026,9,5,12,27,49,tzinfo=timezone.utc).timestamp()


def save(path, data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');os.replace(temp,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    tasks=[]
    for seed in (42,43):
        for arm,phantom in [('pointnet-stray',0.),('pointnet-matched',.2)]:
            parent=FINAL/arm/f'seed-{seed}'
            directory=parent/run_dir_name('pointnet','trainall',['dx','dy','dz'])
            cmd=[sys.executable,'-m','rocklabel.train.cli','train','--model','pointnet',
                '--test-run','all',
                '--features','dx','dy','dz','--cache-dir','training/caches/full-sweep',
                '--runs-root',str(parent),'--epochs','60','--patience','15','--batch','256',
                '--seed',str(seed),'--device','cuda','--aug-stray-frac','.05','--aug-thin-min','.5',
                '--aug-phantom-mode','matched','--aug-phantom-frac',str(phantom)]
            audit=[sys.executable,'-u',str(REPORT/'full_map_audit.py'),'--checkpoint',str(directory/'best.pt'),
                '--out',str(FINAL/f'full-map-{arm}-{seed}'),
                '--frames-dir',str(ROOT/f'training/caches/lance-finalists-{arm}-{seed}'),
                '--device','cuda','--deadline-utc',str(DEADLINE)]
            tasks.append(dict(arm=arm,seed=seed,directory=str(directory),train=cmd,audit=audit))
    if not args.execute:
        print(json.dumps(dict(deadline_utc='2026-09-05T12:27:49Z',tasks=tasks),indent=2));return
    if FINAL.exists():
        raise RuntimeError('output already exists; inspect before starting a duplicate')
    FINAL.mkdir()
    shutil.copytree(SWEEP/'source',FINAL/'source')
    manifest=dict(deadline_utc='2026-09-05T12:27:49Z',lance_training=False,tasks=tasks,
                  source_sha256=json.loads((SWEEP/'manifest.json').read_text())['source_sha256'],
                  audit_script_sha256=hashlib.sha256((REPORT/'full_map_audit.py').read_bytes()).hexdigest())
    save(FINAL/'manifest.json',manifest)
    state=dict(status='waiting_for_sweep',results=[])
    save(FINAL/'status.json',state)
    pid=json.loads((REPORT/'campaign-process.json').read_text())['pid']
    while time.time()<DEADLINE:
        proc=Path(f'/proc/{pid}/cmdline')
        if not proc.exists() or b'rocklabel.train.lance_campaign' not in proc.read_bytes():
            break
        time.sleep(15)
    if time.time()>=DEADLINE:
        state['status']='budget_exhausted';save(FINAL/'status.json',state);return
    previous=json.loads((SWEEP/'campaign-status.json').read_text())
    if len(previous)!=36 or any(r.get('training')!='complete' or r.get('audit')!='complete' for r in previous):
        state['status']='stopped_for_sweep_inspection';save(FINAL/'status.json',state);return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no finalist training started')
    deadline=time.monotonic()+max(0,DEADLINE-time.time())
    state['status']='running';save(FINAL/'status.json',state)
    # Complete and audit a matched pair before spending time on another seed.
    for pair in (tasks[:2],tasks[2:]):
        for task in pair:
            if DEADLINE-time.time()<900:
                state['status']='insufficient_time_for_another_fit';save(FINAL/'status.json',state);return
            active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                                  capture_output=True,text=True,check=True)
            if active.stdout.strip():
                state['status']='gpu_occupied';save(FINAL/'status.json',state);return
            name=f"{task['arm']}-{task['seed']}"
            print('Train',name,flush=True)
            status=run_child(task['train'],FINAL/f'train-{name}.log',deadline,FINAL/'source')
            row=dict(arm=task['arm'],seed=task['seed'],training=status)
            state['results'].append(row);save(FINAL/'status.json',state)
            if status!='complete':
                state['status']=status;save(FINAL/'status.json',state);return
        for task in pair:
            name=f"{task['arm']}-{task['seed']}"
            print('Full-map audit',name,flush=True)
            status=run_child(task['audit'],FINAL/f'audit-{name}.log',deadline)
            row=next(r for r in state['results'] if r['arm']==task['arm'] and r['seed']==task['seed'])
            row['audit']=status;save(FINAL/'status.json',state)
            if status!='complete':
                state['status']=status;save(FINAL/'status.json',state);return
    state['status']='complete';save(FINAL/'status.json',state)


if __name__=='__main__':
    main()
