"""Prepare target negatives through the operational classifier input path.

The historical Lance cache used a labeler's different height band. This cache
uses the measured floor, full operational band, 8 m range and native builder.
Only neighborhoods outside every rock's expanded XY footprint are retained.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from rocklabel.config import load_config
from rocklabel.labels import load_labels
from rocklabel.geometry.leveling import pin_level_to_labels,check_level_match,level_record
from rocklabel.recording.pipeline import ScanStream,WindowedScanStream
from rocklabel.dataset.neighborhoods import build_inference_samples
from rocklabel.dataset.labeling import inside_arena
from rocklabel.train import visual_audit as V

REPORT=Path(__file__).resolve().parent
CACHE=ROOT/'training/caches/lance-operational-negatives'
RUN='lance_operational_negatives'
RECORDING=ROOT/'recordings/archive/misc/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.mcap'
LABELS=ROOT/'labels/archive/lance_raw_data_2026_05_20-12_50_53_0.lidar.noselfhits.labels.json'


def main():
    if CACHE.exists():raise RuntimeError('cache already exists; preserve previous samples')
    g=json.loads((REPORT/'full-map/settings.json').read_text())['generator']
    labels=load_labels(str(LABELS));cfg=pin_level_to_labels(load_config(None),labels.level)
    stream=ScanStream(str(RECORDING),cfg,stride=1,progress=False)
    check_level_match(labels.level,level_record(stream),str(LABELS))
    floor=getattr(getattr(stream,'solution',None),'floor_z',None)
    if floor is None:floor=labels.level.get('floor_z') if labels.level else None
    if floor is None:raise ValueError('measured floor required')
    limits=V._crop_limits([g]);arrays={k:[] for k in ['points','counts','centers','frame']}
    times={};n=0;origin=None
    for k,scan in enumerate(WindowedScanStream(stream,.05)):
        if k%240:continue
        if origin is None:origin=float(scan.time_s)
        frame=V._crop_frame(scan,limits,float(floor),(-.1,.6),8.)
        if frame is None:continue
        rng=np.random.default_rng([int(g['seed']),frame.index])
        samples=build_inference_samples(frame.xyz,frame.intensity,g,rng)
        if samples is None:continue
        centers=samples['centers_odom']
        keep=(V.projected_rock_labels(centers,labels.rocks,.6)==0) & inside_arena(centers,labels.arena)
        if not keep.any():continue
        arrays['points'].append(samples['neighborhoods'][keep])
        arrays['counts'].append(samples['true_counts'][keep])
        arrays['centers'].append(centers[keep])
        arrays['frame'].append(np.full(int(keep.sum()),frame.index,np.int32))
        times[str(frame.index)]=float(scan.time_s)-origin;n+=int(keep.sum())
        if len(times)%10==0:print('Prepared frames',len(times),'negative samples',n,flush=True)
    arrays={k:np.concatenate(v) for k,v in arrays.items()};arrays['labels']=np.zeros(n,np.int8)
    directory=CACHE/RUN;directory.mkdir(parents=True)
    for key,value in arrays.items():np.save(directory/f'{key}.npy',value)
    settings=dict(generator=g,recording=str(RECORDING),labels=str(LABELS),floor_z=float(floor),
        floor_band=[-.1,.6],range_max=8.,stride=240,window_s=.05,exclusion_m=.6,
        label_height_band_used=False,builder='build_inference_samples',
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    identity=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()+LABELS.read_bytes()).hexdigest()
    meta=dict(config_hash=identity,profile='lance-operational-negatives',generator=g,
        settings=settings,runs={RUN:dict(n=n,rock=0,clear=n,frames=len(times),frame_times=times)})
    (CACHE/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    frames,centers=arrays['frame'],arrays['centers']
    west=np.flatnonzero(centers[:,0]<-3.6);east=np.flatnonzero(centers[:,0]>-2.4)
    uniq,count=np.unique(frames[west],return_counts=True)
    i=min(np.searchsorted(np.cumsum(count),count.sum()*.8),len(uniq)-1)
    cut,gap=uniq[i],uniq[max(0,i-2)]
    train=west[frames[west]<gap];val=west[frames[west]>=cut]
    assert len(train)>1000 and len(val)>500 and len(east)>1000
    split=dict(lance_run=RUN,cache_dir=str(CACHE),cache_hash=identity,
        train_x_max=-3.6,test_x_min=-2.4,rock_footprint_exclusion_m=.6,
        val_start_frame=int(cut),gap_start_frame=int(gap),training_rocks=0,
        train_count=len(train),validation_count=len(val),reserved_east_count=len(east),
        scope='Negative-only target adaptation with operational inputs; spatial/temporal holdouts '
              'in the same recording are not an independent competition test.',
        val_rule='Tail block with at least 20% of western negative samples; two selected frames '
                 'before validation excluded from training.',settings=settings)
    np.savez_compressed(REPORT/'lance-negative-split.npz',train=train,val=val,test=east,
                        far_from_rocks=np.arange(n))
    (REPORT/'lance-negative-split.json').write_text(json.dumps(split,indent=2)+'\n')
    print(split,flush=True)


if __name__=='__main__':main()
