"""Bounded desktop replay measurement; writes timings and closes its own window."""
from pathlib import Path
import argparse
import csv
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--seconds',type=float,default=165)
parser.add_argument('--baseline',action='store_true')
parser.add_argument('--compare',action='store_true')
args=parser.parse_args();rows=[];start=None;closing=False

import rocklabel.live.viz.app as module
if args.baseline:
    source=Path(__file__).with_name('baseline_app.py')
    exec(compile(source.read_text(),str(source),'exec'),module.__dict__)
    import rocklabel.live.viz
    rocklabel.live.viz.VizApp=module.VizApp
VizApp=module.VizApp
from rocklabel.cli import main

start_ticking=VizApp._start_ticking
def ticking(self):
    self.set_color_mode('model')
    if not self._secondary:
        def probes():
            while not self._tick_stop.wait(1.):
                if start is None:continue
                t=time.perf_counter()
                def delivered(t=t):
                    rows.append(dict(wall_s=t-start,recording_s=self._engine.source.position_sec,
                        operation='_input_latency',seconds=time.perf_counter()-t,points=0))
                self.post(delivered)
        threading.Thread(target=probes,daemon=True).start()
    return start_ticking(self)
VizApp._start_ticking=ticking

for name in ('_set_point_geometry','_model_colors','_rebuild_rocks','_update_scene'):
    original=getattr(VizApp,name)
    def wrap(self,*a,_name=name,_original=original,**kw):
        global start,closing
        t=time.perf_counter()
        if start is None:start=t
        try:return _original(self,*a,**kw)
        finally:
            elapsed=time.perf_counter()-t
            rows.append(dict(wall_s=t-start,recording_s=self._engine.source.position_sec,
                             operation=_name,seconds=elapsed,
                             points=len(a[1]) if _name=='_set_point_geometry' else 0))
            if _name=='_update_scene' and not self._secondary and t-start>=args.seconds and not closing:
                closing=True;self.post(self.close_window)
    setattr(VizApp,name,wrap)

try:
    main(['live','--play','recordings/archive/misc/trimmed_lance.mcap','--speed','1',
          '--source','udp','--model','training/experiments/lance-campaign-v1/pointnet-matched/seed-44/pointnet_loro_VolleyBallTest3.reslam_dx-dy-dz/best.pt',
          '--max-range','8','--floor-band','-.10','.60','--web-ui','--no-browser']+
         (['--compare-model','training/experiments/deploy/cls-stray/trainall/best.pt'] if args.compare else []))
finally:
    with args.out.open('w') as f:
        w=csv.DictWriter(f,fieldnames=['wall_s','recording_s','operation','seconds','points'])
        w.writeheader();w.writerows(rows)
    print('Wrote',args.out,flush=True)
