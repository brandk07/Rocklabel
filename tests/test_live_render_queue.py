"""Slow redraws must not bury input events in Open3D's unbounded queue."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip('open3d.visualization.gui')
from rocklabel.live.viz.app import VizApp


def viewer():
    app=object.__new__(VizApp)
    app._update_pending_lock=threading.Lock()
    app._update_pending=False
    app._tick_stop=threading.Event()
    queue=[]
    app.post=lambda fn: (queue.append(fn) or True)
    return app,queue


def test_slow_render_coalesces_ticks_and_leaves_controls_reachable():
    app,queue=viewer();seen=[];state={'frame':0}
    def render():
        seen.append(state['frame'])
        # Ticks can arrive from the worker while the main thread is busy.
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(lambda _:app._post_update(),range(100)))
    app._update_scene=render
    for i in range(2400):
        state['frame']=i
        app._post_update()
    assert len(queue)==1  # two minutes of 20 Hz ticks, one current redraw
    app.post(lambda:seen.append('control'))
    queue.pop(0)()
    assert seen==[2399]
    assert len(queue)==1
    queue.pop(0)()
    assert seen==[2399,'control']
    app._post_update()
    assert len(queue)==1


def test_failed_render_does_not_disable_future_updates():
    app,queue=viewer()
    def fail():raise RuntimeError('render failed')
    app._update_scene=fail;app._post_update()
    with pytest.raises(RuntimeError):queue.pop(0)()
    app._update_scene=lambda:None;app._post_update()
    assert len(queue)==1


def test_no_window_and_shutdown_do_not_leave_or_execute_pending_work():
    app,queue=viewer();post=app.post;app.post=lambda fn:False
    app._post_update();assert not app._update_pending
    app.post=post;app._update_scene=lambda:pytest.fail('rendered after close')
    app._post_update();app._tick_stop.set();queue.pop(0)()
    app._post_update();assert not queue


def test_failed_post_can_be_retried():
    app,queue=viewer();post=app.post
    def fail(fn):raise RuntimeError('window unavailable')
    app.post=fail
    with pytest.raises(RuntimeError):app._post_update()
    app.post=post;app._post_update();assert len(queue)==1
