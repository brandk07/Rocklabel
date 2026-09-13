"""Separate target-domain negative adaptation trial, within the original budget.

All Lance rock neighborhoods are excluded. Western negatives train/validate;
eastern negatives and all Lance rock neighborhoods are evaluation-only.
Same-recording adaptation is not an independent competition transfer test.
"""
from pathlib import Path
import argparse
import copy
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
import subprocess
import sys
import time
import tempfile

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
REPORT=Path(__file__).resolve().parent
OUT=ROOT/'training/experiments/lance-target-negatives-v1'
BASE=ROOT/'training/experiments/deploy/cls-stray/trainall/best.pt'
DEADLINE=datetime(2026,9,5,12,27,49,tzinfo=timezone.utc).timestamp()


def train(arm, smoke=False):
    import numpy as np
    import torch
    from rocklabel.train import engine as E,data as D,metrics as M
    from rocklabel.train.models import build_model_from_config
    from rocklabel.train.visual_audit import projected_rock_labels
    from rocklabel.labels import load_labels
    torch.set_num_threads(4)
    if not smoke and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device('cpu' if smoke else 'cuda');E._seed_all(42)
    ck=torch.load(BASE,map_location='cpu',weights_only=False);cfg=copy.deepcopy(ck['config'])
    meta=D.load_cache_meta(str(ROOT/'training/caches/full-sweep'))
    runs=[D.RunData(str(ROOT/'training/caches/full-sweep'),r) for r in sorted(meta['runs'])]
    masks=[D.block_val_mask(r.frame,cfg['val_frac'],cfg['gap_frames'],
             times=meta['runs'][r.run_id]['frame_times'],gap_seconds=cfg.get('gap_seconds')) for r in runs]
    tr=E.Split(runs,[m[0] for m in masks]);va=E.Split(runs,[m[1] for m in masks])
    split_meta=json.loads((REPORT/'lance-negative-split.json').read_text())
    split=dict(np.load(REPORT/'lance-negative-split.npz'))
    target_cache=split_meta['cache_dir']
    lance=D.RunData(target_cache,split_meta['lance_run'])
    masks={key:np.isin(np.arange(len(lance)),split[key]) for key in ['train','val']}
    labels=load_labels(str(ROOT/'labels/archive/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json'))
    assert np.all(lance.labels[masks['train']]==0)
    assert np.all(projected_rock_labels(lance.centers[masks['train']],labels.rocks,.6)==0)
    assert np.all(lance.centers[masks['train'],0]<-3.6)
    assert not np.intersect1d(lance.frame[masks['train']],lance.frame[masks['val']]).size
    lt,lv=E.Split([lance],[masks['train']]),E.Split([lance],[masks['val']])
    del runs,lance
    if smoke:
        for data in (tr,va):
            idx=torch.cat([torch.where(data.labels==0)[0][:64],torch.where(data.labels==1)[0][:64]])
            data.points,data.labels,data.counts=data.points[idx],data.labels[idx],data.counts[idx]
        for data in (lt,lv):
            data.points,data.labels,data.counts=data.points[:32],data.labels[:32],data.counts[:32]
    model=build_model_from_config(cfg).to(device);model.load_state_dict(ck['model'])
    threshold=float(ck['threshold']);target_weight=0. if arm=='control' else .5
    directory=(Path(tempfile.mkdtemp(prefix='rocklabel-target-smoke-')) if smoke else OUT)/arm
    directory.mkdir(parents=True,exist_ok=False)
    epochs=1 if smoke else 15
    trial=dict(arm=arm,seed=42,epochs=epochs,smoke=smoke,lr=.0001,target_loss_weight=target_weight,
        batch_volleyball=128,batch_target=32,frozen_batchnorm_statistics=True,
        target_half_matched_to_volleyball_positive_counts=True,threshold_fixed=threshold,
        selection='Minimize western validation negative FP fraction, then mean probability, '
                  'subject to volleyball validation AP >= initial AP - .01 and rock recall '
                  '>= .98 * initial recall at the fixed threshold. Initial weights eligible.',
        source_checkpoint=str(BASE),source_checkpoint_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        split=split_meta,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        split_sha256=hashlib.sha256((REPORT/'lance-negative-split.npz').read_bytes()).hexdigest())
    (directory/'adaptation.json').write_text(json.dumps(trial,indent=2)+'\n')
    (directory/'config.json').write_text(json.dumps(cfg|{'target_adaptation':trial},indent=2)+'\n')
    y=va.labels.numpy()
    def evaluate():
        p=E.predict(model,va,device,256);neg=E.predict(model,lv,device,256)
        return dict(volleyball_val_ap=M.average_precision(y,p),
                    volleyball_val_recall=M.confusion(y,p,threshold)['recall'],
                    western_val_false_fraction=float((neg>=threshold).mean()),
                    western_val_mean_probability=float(neg.mean()))
    initial=evaluate();best=initial;history=[dict(epoch=-1,**initial,eligible=True,selected=True)]
    def publish(epoch,metrics):
        payload=dict(ck,model=model.state_dict(),threshold=threshold,target_adaptation=trial,
                     adaptation_epoch=epoch,adaptation_validation=metrics)
        E._save_checkpoint(payload,str(directory/'best.pt'))
    publish(-1,initial)
    print('Initial',initial,flush=True)
    gen=torch.Generator(device=device).manual_seed(42)
    opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=cfg['weight_decay'])
    positive_counts=tr.counts[tr.labels==1].clamp(8,256).to(device)
    pos_weight=float((tr.labels==0).sum()/max(int((tr.labels==1).sum()),1))
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    def match_counts(pts,cnt):
        b,n,_=pts.shape;real=cnt.clamp(max=n)
        desired=positive_counts[torch.randint(len(positive_counts),(b,),device=device,generator=gen)]
        keep=torch.where(torch.rand(b,device=device,generator=gen)<.5,torch.minimum(real,desired),real)
        rank=torch.rand(b,n,device=device,generator=gen).masked_fill(
            torch.arange(n,device=device)[None]>=real[:,None],2.).argsort(dim=1)
        positions=torch.arange(n,device=device)[None].expand(b,n)%keep[:,None]
        take=rank.gather(1,positions)
        return pts.gather(1,take[...,None].expand(-1,-1,4)),keep
    for epoch in range(epochs):
        if time.time()>=DEADLINE:break
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        order=torch.randperm(len(tr));losses=[]
        for start in range(0,len(tr)-127,128):
            if time.time()>=DEADLINE:break
            idx=order[start:start+128];li=torch.randint(len(lt),(32,))
            vp,vc,_=E._augment(tr.points[idx].to(device),tr.counts[idx].to(device),cfg,gen)
            lp,lc=match_counts(lt.points[li].to(device),lt.counts[li].to(device))
            lp,lc,_=E._augment(lp,lc,cfg,gen)
            logits=model(torch.cat([vp,lp]),torch.cat([vc,lc]))
            loss=loss_fn(logits[:128],tr.labels[idx].to(device))
            loss=loss+target_weight*torch.nn.functional.binary_cross_entropy_with_logits(logits[128:],torch.zeros(32,device=device))
            loss=loss+cfg['tnet_reg']*model.pop_regularizer()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite adaptation loss')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
        if time.time()>=DEADLINE:break
        metrics=evaluate()
        eligible=(metrics['volleyball_val_ap']>=initial['volleyball_val_ap']-.01 and
                  metrics['volleyball_val_recall']>=.98*initial['volleyball_val_recall'])
        selected=eligible and (metrics['western_val_false_fraction'],metrics['western_val_mean_probability']) < (best['western_val_false_fraction'],best['western_val_mean_probability'])
        if selected:best=metrics;publish(epoch,metrics)
        history.append(dict(epoch=epoch,**metrics,eligible=eligible,selected=selected))
        with (directory/'history.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=history[0]);writer.writeheader();writer.writerows(history)
        print(history[-1],flush=True)
    selected_ck=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    changed=any(not torch.equal(v.cpu(),ck['model'][k].cpu()) for k,v in selected_ck['model'].items())
    (directory/'validation.json').write_text(json.dumps(dict(initial=initial,selected=best,
        selected_epoch=selected_ck['adaptation_epoch'],weights_changed=changed,
        completed_epochs=len(history)-1),indent=2)+'\n')
    if not smoke and time.time()<DEADLINE:
        # Eastern negatives are opened only after selection is finished.
        lance=D.RunData(target_cache,split_meta['lance_run'])
        test=E.Split([lance],[np.isin(np.arange(len(lance)),split['test'])])
        model.load_state_dict(selected_ck['model']);p=E.predict(model,test,device,256)
        model.load_state_dict(ck['model']);reference=E.predict(model,test,device,256)
        (directory/'eastern-negative-test.json').write_text(json.dumps(dict(n=len(test),
            reference_false_fraction=float((reference>=threshold).mean()),
            selected_false_fraction=float((p>=threshold).mean()),threshold=threshold,
            scope='Reserved eastern clear neighborhoods in the same recording; no rock recall measure.'),indent=2)+'\n')
    print('Output',directory,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--train-arm',choices=['control','adapt'])
    parser.add_argument('--smoke',action='store_true',help='one tiny CPU training/checkpoint round trip in /tmp')
    args=parser.parse_args()
    if args.train_arm:return train(args.train_arm,args.smoke)
    if not args.execute:
        print('Dry: paired control/adaptation on existing selected western negatives; '
              'wait for sweep and finalist campaigns; original deadline 2026-09-05 12:27:49 UTC.');return
    from rocklabel.train.lance_campaign import run_child
    if OUT.exists():raise RuntimeError('output already exists')
    OUT.mkdir()
    def status(value):
        (OUT/'status.json').write_text(json.dumps(value,indent=2)+'\n')
    status(dict(status='waiting_for_finalists'))
    pid=json.loads((REPORT/'finalist-process.json').read_text())['pid']
    while time.time()<DEADLINE:
        proc=Path(f'/proc/{pid}/cmdline')
        if not proc.exists() or b'finalist_campaign.py' not in proc.read_bytes():break
        time.sleep(15)
    if DEADLINE-time.time()<1200:
        status(dict(status='insufficient_remaining_budget'));return
    prior=json.loads((ROOT/'training/experiments/lance-finalists-v1/status.json').read_text())
    if prior['status']!='complete':
        status(dict(status='stopped_for_finalist_inspection',finalist_status=prior['status']));return
    deadline=time.monotonic()+max(0,DEADLINE-time.time());results=[]
    for arm in ['control','adapt']:
        active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        if active.stdout.strip():status(dict(status='gpu_occupied',results=results));return
        print('Train',arm,flush=True)
        state=run_child([sys.executable,'-u',str(Path(__file__).resolve()),'--train-arm',arm],OUT/f'train-{arm}.log',deadline)
        results.append(dict(arm=arm,training=state));status(dict(status='running',results=results))
        if state!='complete':status(dict(status=state,results=results));return
    for row in results:
        arm=row['arm'];print('Audit',arm,flush=True)
        cmd=[sys.executable,'-u',str(REPORT/'full_map_audit.py'),'--checkpoint',str(OUT/arm/'best.pt'),
             '--out',str(OUT/f'full-map-{arm}'),'--frames-dir',str(ROOT/f'training/caches/lance-target-{arm}'),
             '--device','cuda','--deadline-utc',str(DEADLINE)]
        row['audit']=run_child(cmd,OUT/f'audit-{arm}.log',deadline)
        status(dict(status='running',results=results))
        if row['audit']!='complete':status(dict(status=row['audit'],results=results));return
    status(dict(status='complete',results=results))


if __name__=='__main__':main()
