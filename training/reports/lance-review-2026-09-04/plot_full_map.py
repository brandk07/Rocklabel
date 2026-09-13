"""Standalone figures for the full-recording map diagnostics."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from rocklabel.train import visual_audit as V
from rocklabel.labels import load_labels


def main():
    report=Path(__file__).resolve().parent
    settings=json.loads((report/'full-map/settings.json').read_text())
    labels=load_labels(settings['labels'])
    names=['native-latest','native-3-positive-observations',
           'ground-hysteresis-1-on-3-clear','ground-hysteresis-3-on-3-clear']
    short=['Current native map','Three positive observations per native voxel',
           'Ground-cell hysteresis: one positive / three clear',
           'Ground-cell hysteresis: three positive / three clear']
    base=np.load(report/'full-map-ablation/native-latest.npz')
    background=base['positions']
    fig,axes=plt.subplots(2,2,figsize=(13,11),layout='constrained')
    for ax,name,title in zip(axes.flat,names,short):
        with np.load(report/f'full-map-ablation/{name}.npz') as f:
            pos,p=f['positions'],f['probabilities']
        metrics=json.loads((report/f'full-map-ablation/{name}.json').read_text())['aggregate']
        hot=p>=settings['threshold'];lab=V.projected_rock_labels(pos,labels.rocks,.05)
        ax.scatter(background[:,0],background[:,1],s=2,c='#dddddd',linewidths=0)
        ax.scatter(pos[hot & (lab==0),0],pos[hot & (lab==0),1],s=4,c='#d55e00',linewidths=0)
        ax.scatter(pos[hot & (lab==-1),0],pos[hot & (lab==-1),1],s=4,c='#0072b2',linewidths=0)
        ax.scatter(pos[hot & (lab==1),0],pos[hot & (lab==1),1],s=5,c='#009e73',linewidths=0)
        for rock in labels.rocks:
            V._rock_outline(ax,rock,'#222222',.8)
            ax.text(rock.center[0],rock.center[1],str(rock.id),fontsize=7,ha='center',va='center')
        ax.set_title(f"{title}\nFalse footprint {metrics['false_footprint_cells']*.01:.2f} m²; "
                     f"mean/worst coverage {metrics['macro_footprint_coverage']:.1%} / "
                     f"{metrics['worst_rock_footprint_coverage']:.1%}",fontsize=10)
        ax.set_aspect('equal');ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)')
    fig.suptitle('Lance: final accumulated map over 35.5 minutes, all 12 labelled rocks\n'
                 'Identical unfiltered classifier predictions; full floor band; diagnostic update rules',fontsize=12)
    handles=[Line2D([],[],marker='o',linestyle='',color=c,label=n) for c,n in
             [('#d55e00','Positive outside footprint'),('#009e73','Positive inside footprint'),
              ('#0072b2','Boundary shell'),('#dddddd','Other scored cells (not evidence of free space)')]]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,fontsize=8)
    fig.savefig(report/'full-map-comparison.png',dpi=160)
    fig.savefig(report/'full-map-comparison.svg');plt.close(fig)
    rows=list(csv.DictReader((report/'full-map/timeline.csv').open()))
    t=np.array([float(r['time_s']) for r in rows])/60
    total=np.array([float(r['false_cells']) for r in rows])*.01
    stale=np.array([float(r['false_cells_older_60s']) for r in rows])*.01
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.plot(t,total,label='All false cells',color='#d55e00')
    ax.plot(t,stale,label='Representative last refreshed >60 s ago',color='#0072b2')
    ax.set(xlabel='Recording time (minutes)',ylabel='False ground-cell area (m²)',
           title='Baseline map persistence: 3D label attribution, unfiltered input')
    ax.grid(alpha=.2);ax.legend();fig.savefig(report/'full-map-persistence.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
