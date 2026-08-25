/* rocklabel live control panel.
 *
 * One page, no build step, no network dependency. The server owns the control
 * catalog, so this file knows how to render a *kind* of control (bool, float,
 * int, enum, action, readout) and nothing about which controls exist. Adding a
 * knob is a line in spec.py.
 *
 * The hard part is not rendering, it is the two-way sync: the same settings can
 * be changed from the Open3D window's keyboard shortcuts while you are dragging
 * a slider here. So the poll mirrors state onto every control EXCEPT ones the
 * user is currently touching, and except for a short window after a write —
 * otherwise a poll in flight when you let go snaps the control back to the old
 * value. See fresh() / isBusy().
 */
'use strict';

const S = {
  schema: null,
  values: {},
  status: {},
  flags: {},
  transport: null,
  /* control id -> timestamp until which our own value wins over the server's */
  pending: {},
  /* control id -> node handles, so the poll can write without re-rendering */
  nodes: {},
  scrubbing: false,
  /* id of the control with a pointer currently held down on it, if any */
  dragging: null,
  /* the latest /api/scene payload, and the view toggles over it */
  scene: null,
  histLog: true,
  mapTable: false,
  /* last prediction display mode seen, so a switch can refresh the map now */
  lastDisplay: null,
};

/* Writes settle within a poll or two; this is how long we trust ours over the
   server's. Long enough to cover the post + the next poll already in flight. */
const SETTLE_MS = 700;
const POLL_MS = 250;
/* The scene payload is tens of kilobytes against a few hundred for state, and
 * a fused surface does not change meaningfully in 250 ms. Its own clock. */
const SCENE_POLL_MS = 1000;

const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function fresh(id) { return (S.pending[id] || 0) > Date.now(); }
function isBusy(id) {
  const n = S.nodes[id];
  if (!n) return false;
  // Mid-drag, mid-type, or inside the settle window after our own write.
  if (S.dragging === id || fresh(id)) return true;
  const active = document.activeElement;
  return (!!n.input && active === n.input) || (!!n.readback && active === n.readback);
}
function touch(id) { S.pending[id] = Date.now() + SETTLE_MS; }

/* A range input reports its value continuously but never "I am being dragged",
   and a mirror landing mid-drag is exactly the jump this page has to avoid. */
function trackDrag(id, el) {
  el.addEventListener('pointerdown', () => { S.dragging = id; });
  el.addEventListener('pointerup', () => { S.dragging = null; });
  el.addEventListener('pointercancel', () => { S.dragging = null; });
}

/* --------------------------------------------------- collapse memory ---- */
/* Which terminal panels are folded, per session. localStorage, guarded: a
   storage-less context must cost a closed panel nothing. */
function readCollapse() {
  try { return JSON.parse(localStorage.getItem('live-sections') || '{}'); }
  catch (e) { return {}; }
}
function sectionOpen(id) { return readCollapse()[id] !== false; }
function writeCollapse(id, open) {
  try {
    const st = readCollapse();
    st[id] = !!open;
    localStorage.setItem('live-sections', JSON.stringify(st));
  } catch (e) { /* remembering is a nicety, never a requirement */ }
}

/* ------------------------------------------------------------------ api */
async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).error || msg; } catch (e) { /* not JSON */ }
    throw new Error(msg);
  }
  return res.json();
}

async function setValue(id, value) {
  touch(id);
  try {
    apply(await api('/api/set', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: id, value }),
    }));
  } catch (e) {
    console.error(`set ${id}:`, e.message);
  }
}

async function runAction(name, args) {
  try {
    apply(await api('/api/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, args: args || {} }),
    }));
  } catch (e) {
    console.error(`action ${name}:`, e.message);
  }
}

/* --------------------------------------------------------------- render */
function labelFor(c) {
  const wrap = h('div', 'ctl-label');
  wrap.appendChild(h('span', 'ctl-name', c.label));
  if (c.unit) wrap.appendChild(h('span', 'ctl-unit', c.unit));
  if (c.key) wrap.appendChild(h('kbd', 'ctl-key', c.key));
  return wrap;
}

/* The help text lives behind a "?" per control: showing all of it at once
   turns a control panel into a manual, hiding it entirely loses the only
   record of what these numbers mean. */
function helpToggle(c, row) {
  if (!c.help) return null;
  const btn = h('button', 'ctl-help-btn', '?');
  btn.type = 'button';
  btn.setAttribute('aria-label', `What does "${c.label}" do?`);
  const body = h('div', 'ctl-help', c.help);
  body.hidden = true;
  btn.onclick = () => {
    body.hidden = !body.hidden;
    btn.classList.toggle('is-open', !body.hidden);
  };
  row.appendChild(body);
  return btn;
}

function renderControl(c) {
  const row = h('div', 'ctl');
  const label = labelFor(c);
  const kind = c.kind;
  // Where the "?" hangs: beside the label on value rows, and on the control
  // itself for the full-width ones, which have no label element.
  let helpHost = label;

  if (kind === 'bool' || kind === 'action') row.classList.add('ctl-wide');

  if (kind === 'readout') {
    const out = h('div', 'ctl-out', '—');
    row.appendChild(label);
    row.appendChild(out);
    S.nodes[c.id] = { out, control: c };
  } else if (kind === 'bool') {
    const wrap = h('label', 'ctl-check');
    const box = h('input');
    box.type = 'checkbox';
    box.onchange = () => setValue(c.id, box.checked);
    wrap.appendChild(box);
    wrap.appendChild(h('span', null, c.label));
    if (c.key) wrap.appendChild(h('kbd', 'ctl-key', c.key));
    row.appendChild(wrap);
    helpHost = wrap;
    S.nodes[c.id] = { input: box, control: c };
  } else if (kind === 'action') {
    const bar = h('div', 'ctl-actions');
    const btn = h('button', `btn${c.style ? ` btn-${c.style}` : ''}`, c.label);
    btn.onclick = () => runAction(c.id);
    bar.appendChild(btn);
    if (c.key) bar.appendChild(h('kbd', 'ctl-key', c.key));
    row.appendChild(bar);
    helpHost = bar;
    S.nodes[c.id] = { button: btn, control: c };
  } else if (kind === 'enum') {
    const sel = h('select', 'input');
    // Choices may carry a group (the checkpoint pickers offer one entry per
    // trained fold, dozens of them): an <optgroup> per group turns a wall of
    // options into a list you can actually find a run in.
    const groups = new Map();
    for (const ch of c.choices || []) {
      const opt = h('option', null, ch.label);
      opt.value = String(ch.value);
      if (!ch.group) { sel.appendChild(opt); continue; }
      let box = groups.get(ch.group);
      if (!box) {
        box = document.createElement('optgroup');
        box.label = ch.group;
        groups.set(ch.group, box);
        sel.appendChild(box);
      }
      box.appendChild(opt);
    }
    sel.onchange = () => {
      const raw = sel.value;
      const choice = (c.choices || []).find((x) => String(x.value) === raw);
      setValue(c.id, choice ? choice.value : raw);
    };
    row.appendChild(label);
    const cell = h('div', 'ctl-value');
    cell.appendChild(sel);
    row.appendChild(cell);
    S.nodes[c.id] = { input: sel, control: c };
  } else {
    // int / float. Bounded-and-explored values get a slider with a readback;
    // values you arrive with (the region bounds, the caps) get a number box you
    // can type an exact -1.50 into, which is what a slider can never do.
    const cell = h('div', 'ctl-value');
    const wide = (c.max != null && c.min != null) && (c.max - c.min) > 1000;
    const typed = wide || c.id.startsWith('region.') || c.id.startsWith('crop.')
      || c.id === 'view.accum_frames' || c.id === 'view.accum_max_points';
    let input;
    let readback = null;
    if (typed) {
      input = h('input', 'input ctl-num');
      input.type = 'number';
      if (c.min != null) input.min = c.min;
      if (c.max != null) input.max = c.max;
      if (c.step != null) input.step = c.step;
      input.onchange = () => setValue(c.id, input.value);
    } else {
      // Slider AND number box, not one or the other: you find a threshold by
      // dragging, then you type the exact 0.982 you settled on. Both write the
      // same control, and the poll mirrors onto both.
      input = h('input', 'ctl-slider');
      input.type = 'range';
      input.min = c.min != null ? c.min : 0;
      input.max = c.max != null ? c.max : 1;
      input.step = c.step != null ? c.step : 0.01;
      readback = h('input', 'ctl-readback');
      readback.type = 'number';
      if (c.min != null) readback.min = c.min;
      if (c.max != null) readback.max = c.max;
      if (c.step != null) readback.step = c.step;
      readback.setAttribute('aria-label', `${c.label} value`);
      trackDrag(c.id, input);
      input.oninput = () => {
        touch(c.id);
        readback.value = fmt(c, input.value);
      };
      input.onchange = () => setValue(c.id, input.value);
      readback.onchange = () => {
        input.value = readback.value;      // keep the thumb with the number
        setValue(c.id, readback.value);
      };
    }
    cell.appendChild(input);
    if (readback) cell.appendChild(readback);
    row.appendChild(label);
    row.appendChild(cell);
    S.nodes[c.id] = { input, readback, control: c };
  }

  const help = helpToggle(c, row);
  if (help) helpHost.appendChild(help);
  return row;
}

function fmt(c, value) {
  const n = Number(value);
  if (!isFinite(n)) return '—';
  if (c.kind === 'int') return n.toLocaleString();
  const step = c.step || 0.01;
  const places = step >= 1 ? 0 : String(step).split('.')[1].length;
  return n.toFixed(places);
}

function renderSchema(schema) {
  S.schema = schema;
  S.nodes = {};
  const host = $('#panels');
  host.textContent = '';
  for (const sec of schema.sections) {
    // The replay section IS the transport bar above the stage, not a card.
    if (sec.id === 'replay') continue;
    const card = h('details', 'card sec');
    card.open = sectionOpen(sec.id);
    card.addEventListener('toggle', () => writeCollapse(sec.id, card.open));

    const head = h('summary', 'sec-head');
    head.appendChild(h('span', 'sec-caret', '▸'));
    head.appendChild(h('span', 'sec-title',
      sec.id === 'model' && schema.model ? `${sec.title} · ${schema.model}` : sec.title));
    head.appendChild(h('span', 'sec-rule'));
    card.appendChild(head);

    const body = h('div', 'sec-body');
    for (const c of sec.controls) body.appendChild(renderControl(c));
    if (sec.id === 'model') {
      const warn = h('div', 'ctl-warning');
      warn.hidden = true;
      body.appendChild(warn);
      S.nodes['model.warning'] = { out: warn };
    }
    card.appendChild(body);
    host.appendChild(card);
  }

  const hasReplay = schema.sections.some((s) => s.id === 'replay');
  $('#transport').hidden = !hasReplay;
  if (hasReplay) initTransport(schema);
  const badge = $('#modeBadge');
  badge.textContent = schema.mode.toUpperCase();
  badge.classList.toggle('is-replay', schema.mode !== 'live');
  $('#subtitle').textContent = schema.subtitle;
  document.title = `rocklabel live · ${schema.subtitle}`;
}

/* ------------------------------------------------------------ transport */
/* Scrubber clocks are mm:ss.d — a recording is navigated by its timeline, and
   83.4 s reads worse than 01:23.4 once it passes a minute. */
function fmtClock(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return `${String(m).padStart(2, '0')}:${rest.toFixed(1).padStart(4, '0')}`;
}

function initTransport(schema) {
  const range = $('#transportRange');
  range.max = Math.max(0.001, schema.duration_sec);
  range.step = 0.05;
  // Registered like any other control so the drag/settle guards in apply()
  // cover the scrubber too — it is the control most likely to be mid-drag.
  const sec = schema.sections.find((s) => s.id === 'replay');
  const control = sec && sec.controls.find((c) => c.id === 'replay.position');
  if (control) S.nodes['replay.position'] = { input: range, control };
  $('#playBtn').onclick = () => runAction('replay.play_pause');
  $('#restartBtn').onclick = () => runAction('replay.restart');
  trackDrag('replay.position', range);

  // Speed lives in the bar rather than in a card: it is a transport control,
  // and it is reached for while watching the scene, not while reading a form.
  const speed = sec && sec.controls.find((c) => c.id === 'replay.speed');
  const sel = $('#speedSelect');
  if (speed && sel) {
    sel.textContent = '';
    for (const ch of speed.choices || []) {
      const opt = h('option', null, ch.label);
      opt.value = String(ch.value);
      sel.appendChild(opt);
    }
    sel.onchange = () => {
      const choice = (speed.choices || []).find((x) => String(x.value) === sel.value);
      setValue('replay.speed', choice ? choice.value : sel.value);
    };
    S.nodes['replay.speed'] = { input: sel, control: speed };
  }

  // Scrubbing is committed on release, not per pixel: a backward seek rewinds
  // the recording and re-fuses the whole map, so one seek per drag, not fifty.
  range.oninput = () => {
    S.scrubbing = true;
    $('#transportTime').textContent =
      `${fmtClock(range.value)} / ${fmtClock(schema.duration_sec)}`;
  };
  range.onchange = () => {
    S.scrubbing = false;
    touch('replay.position');
    setValue('replay.position', range.value);
  };
}

/* ---------------------------------------------------------------- apply */
function apply(snap) {
  S.values = snap.values || {};
  S.status = snap.status || {};
  S.flags = snap.flags || {};
  S.transport = snap.transport || null;

  for (const [id, node] of Object.entries(S.nodes)) {
    const c = node.control;
    if (!c) continue;
    if (c.kind === 'readout') {
      const text = S.status[id];
      node.out.textContent = text == null ? '—' : text;
      node.out.classList.toggle('is-warn', !!(c.warn_flag && S.flags[c.warn_flag]));
      node.out.classList.toggle('is-rec', id === 'record.state' && !!S.flags.recording);
      continue;
    }
    if (c.kind === 'action') continue;
    if (!(id in S.values) || isBusy(id)) continue;

    const v = S.values[id];
    if (c.kind === 'bool') {
      node.input.checked = !!v;
    } else if (c.kind === 'enum') {
      node.input.value = String(v);
    } else {
      node.input.value = v;
      if (node.readback) node.readback.value = fmt(c, v);
    }
  }

  const warnNode = S.nodes['model.warning'];
  if (warnNode) {
    const text = S.status['model.warning'] || '';
    warnNode.out.textContent = text;
    warnNode.out.hidden = !text;
  }

  // Switching the prediction view changes what the overhead map draws, and the
  // map has its own slower clock — so ask for a fresh payload right away
  // instead of leaving the picture a second behind the dropdown.
  const disp = S.values['model.display'];
  if (disp !== undefined && disp !== S.lastDisplay) {
    const first = S.lastDisplay === null;
    S.lastDisplay = disp;
    if (!first) pollScene();
  }

  renderStrip();
  renderAlerts();
  renderDiag();
  if (S.transport) renderTransport();
}

/* The four numbers you glance at without scrolling, always in the top bar. */
function renderStrip() {
  const strip = $('#statusStrip');
  const cells = [
    ['throughput', 'status.rate', 'rate_low'],
    ['cells', 'status.cells', null],
    ['accumulated', 'status.accum', 'accum_capped'],
    ['pose', 'status.pose', null],
  ];
  strip.textContent = '';
  for (const [name, id, warn] of cells) {
    if (!(id in S.status)) continue;
    const cell = h('div', 'ro-cell');
    cell.appendChild(h('span', 'k', name));
    const v = h('span', 'v', S.status[id]);
    v.title = S.status[id];               // pose strings especially get cut off
    if (warn && S.flags[warn]) v.classList.add('is-warn');
    cell.appendChild(v);
    strip.appendChild(cell);
  }
}

/* Every warning the pipeline raises, one line, or nothing at all. Recording
   leads because it changes what touching this page does to disk. */
function renderAlerts() {
  const host = $('#alerts');
  const items = [];
  if (S.flags.recording) items.push(['alert-item is-rec', 'REC — writing to disk']);
  if (S.flags.paused) items.push(['alert-item', 'PAUSED']);
  if (S.flags.rate_low) items.push(['alert-item', 'THROUGHPUT LOW']);
  if (S.flags.accum_capped) items.push(['alert-item', 'ACCUMULATION CAPPED']);
  if (S.flags.region_empty) items.push(['alert-item', 'SCORING REGION EMPTY']);
  // A second model is a second GPU load and a second window; saying so up here
  // means the state is never something you have to scroll to find.
  if (S.flags.comparing) items.push(['alert-item is-compare', 'COMPARING 2 MODELS']);
  if (S.flags.compare_error) items.push(['alert-item is-rec', 'COMPARE FAILED']);
  if (S.flags.compare_caveat) items.push(['alert-item', 'MODELS NOT DIRECTLY COMPARABLE']);
  host.textContent = '';
  host.hidden = !items.length;
  if (!items.length) return;
  const row = h('div', 'alerts-row');
  for (const [cls, text] of items) row.appendChild(h('span', cls, text));
  host.appendChild(row);
}

/* Server-computed diagnostics the pipeline never used to surface. Absent keys
   drop out; no diagnostics at all hides the whole panel. */
const DIAGNOSTICS = [
  ['status.pose_xyz', 'pose'],
  ['status.level_residual', 'level residual'],
  ['status.latency', 'ingest latency'],
  ['status.drops', 'udp drops'],
  ['status.slam_detail', 'slam'],
];

function renderDiag() {
  const host = $('#diagnostics');
  const card = $('#diagCard');
  host.textContent = '';
  let shown = 0;
  for (const [id, name] of DIAGNOSTICS) {
    if (!(id in S.status)) continue;
    const row = h('div', 'diag-row');
    row.appendChild(h('span', 'diag-k', name));
    row.appendChild(h('span', 'diag-v', S.status[id]));
    host.appendChild(row);
    shown++;
  }
  card.hidden = shown === 0;
}

function renderTransport() {
  const t = S.transport;
  $('#playBtn').textContent = t.playing ? '❚❚ Pause' : '▶ Play';
  const range = $('#transportRange');
  if (!S.scrubbing && !fresh('replay.position') && !t.seeking) {
    range.value = t.position_sec;
    $('#transportTime').textContent =
      `${fmtClock(t.position_sec)} / ${fmtClock(t.duration_sec)}`;
  }
}

/* ================================================================ VIEWS ==== */
/* The overhead map, the confidence histogram and the trend sparks. All three
 * read one payload (/api/scene) and are redrawn wholesale — none of them holds
 * user state that a redraw could interrupt, unlike the controls. */

function renderViews() {
  const scene = S.scene;
  if (!scene) return;
  renderMap(scene);
  renderHistogram(scene);
  renderTrends(scene);
}

/* -- overhead ------------------------------------------------------------- */
let mapMarks = [];

function renderMap(scene) {
  const canvas = $('#mapCanvas');
  const out = window.Views.overhead(canvas, scene);
  mapMarks = out.marks;

  const det = scene.detections || { total: 0, shown: 0 };
  const rocks = scene.rocks || { rows: [], total: 0, points: 0, noise_points: 0 };
  const outlined = scene.display === 2;
  const bev = scene.bev;
  const parts = [];
  if (outlined) {
    // In the outline view the count that matters is rocks, not points — and
    // what the noise gate dropped, because that is the number the Min points
    // knob moves.
    parts.push(rocks.total
      ? `${rocks.total.toLocaleString()} rock${rocks.total === 1 ? '' : 's'}`
        + ` from ${rocks.points.toLocaleString()} detections`
        + (rocks.shown < rocks.total ? ` (${rocks.shown.toLocaleString()} drawn)` : '')
      : 'No rocks — no clump is big enough yet');
    if (rocks.noise_points) {
      parts.push(`${rocks.noise_points.toLocaleString()} dropped as noise`);
    }
  } else {
    parts.push(det.total
      ? `${det.total.toLocaleString()} detection${det.total === 1 ? '' : 's'}`
        + (det.shown < det.total ? ` (${det.shown.toLocaleString()} drawn)` : '')
      : 'No detections yet');
  }
  if (bev) parts.push(`height ${bev.z_min.toFixed(2)} … ${bev.z_max.toFixed(2)} m`);
  $('#mapSub').textContent = parts.join(' · ');

  renderMapLegend(bev, outlined);
  if (S.mapTable) {
    if (outlined) renderRockTable(rocks); else renderMapTable(det);
  }
}

/* Identity is never colour-alone: the ramp gets its ends labelled and each
 * mark class gets a named swatch. Overlaid on the map's corner. */
function renderMapLegend(bev, outlined) {
  const host = $('#mapLegend');
  host.textContent = '';
  if (bev) {
    const ramp = h('div', 'legend-ramp');
    ramp.appendChild(h('span', 'legend-ramp-end', `${bev.z_min.toFixed(2)} m`));
    ramp.appendChild(h('span', 'legend-ramp-bar'));
    ramp.appendChild(h('span', 'legend-ramp-end', `${bev.z_max.toFixed(2)} m`));
    ramp.appendChild(h('span', 'legend-ramp-name', 'height'));
    host.appendChild(ramp);
  }
  const marks = outlined
    ? [['outline', 'rock outline'], ['sensor', 'sensor']]
    : [['rock', 'rock ≥ threshold'], ['sensor', 'sensor']];
  for (const [cls, label] of marks) {
    const item = h('span', 'legend-item');
    item.appendChild(h('span', `legend-swatch swatch-${cls}`));
    item.appendChild(document.createTextNode(label));
    host.appendChild(item);
  }
}

function renderMapTable(det) {
  const host = $('#mapTableView');
  host.textContent = '';
  const rows = [...det.rows].sort((a, b) => b[3] - a[3]).slice(0, 25)
    .map((d) => [d[0].toFixed(2), d[1].toFixed(2), d[2].toFixed(2), d[3].toFixed(3)]);
  host.appendChild(window.Charts.tableView(
    ['x (m)', 'y (m)', 'z (m)', 'probability'], rows,
    rows.length ? `Strongest ${rows.length} of ${det.total} detections`
                : 'No detections yet'));
}

/* The outline view's table is one row per rock, biggest first — the list you
 * read to decide whether the noise gate is set right. */
function renderRockTable(rocks) {
  const host = $('#mapTableView');
  host.textContent = '';
  const rows = rocks.rows.slice(0, 25).map((r) => [
    r.area.toFixed(2), String(r.n), r.prob.toFixed(3),
    `${r.x.toFixed(2)}, ${r.y.toFixed(2)}`,
  ]);
  host.appendChild(window.Charts.tableView(
    ['area (m²)', 'points', 'mean probability', 'x, y (m)'], rows,
    rows.length ? `Largest ${rows.length} of ${rocks.total} rocks`
                : 'No clump is big enough to be a rock yet'));
}

/* Dense marks get a nearest-point hit layer rather than pinpoint targets. */
function wireMapHover() {
  const canvas = $('#mapCanvas');
  canvas.addEventListener('mousemove', (e) => {
    if (!mapMarks.length) return;
    const box = canvas.getBoundingClientRect();
    const mx = e.clientX - box.left;
    const my = e.clientY - box.top;
    let best = null;
    let bestD = 24;                       // generous hit radius, not the mark size
    for (const m of mapMarks) {
      const d = Math.hypot(m.x - mx, m.y - my);
      if (d < bestD) { bestD = d; best = m; }
    }
    if (!best) { window.Charts.hideTip(); return; }
    if (best.rock) {
      const r = best.rock;
      window.Charts.showTip(e, 'Rock outline', [
        { label: 'footprint', value: `${r.area.toFixed(2)} m²`,
          color: window.Charts.cssVar('--accent') },
        { label: 'detections', value: r.n.toLocaleString() },
        { label: 'probability', value: `${r.prob.toFixed(3)} mean · `
                                     + `${r.prob_max.toFixed(3)} peak` },
        { label: 'x, y', value: `${r.x.toFixed(2)}, ${r.y.toFixed(2)} m` },
      ]);
      return;
    }
    window.Charts.showTip(e, 'Detection', [
      { label: 'probability', value: best.d[3].toFixed(3),
        color: window.Charts.cssVar('--accent') },
      { label: 'x, y', value: `${best.d[0].toFixed(2)}, ${best.d[1].toFixed(2)} m` },
      { label: 'z', value: `${best.d[2].toFixed(2)} m` },
    ]);
  });
  canvas.addEventListener('mouseleave', () => window.Charts.hideTip());

  $('#mapTable').onclick = () => {
    S.mapTable = !S.mapTable;
    $('#mapTable').setAttribute('aria-pressed', String(S.mapTable));
    $('#mapTableView').hidden = !S.mapTable;
    if (S.scene) renderMap(S.scene);
  };
}

/* -- histogram ------------------------------------------------------------ */
function renderHistogram(scene) {
  const card = $('#histCard');
  const hist = scene.histogram;
  card.hidden = !hist;
  if (!hist) return;

  const plot = $('#histPlot');
  plot.textContent = '';
  const chart = h('div', 'chart');   // live.css sizes .chart svg to its container
  chart.appendChild(window.Views.histogram(hist, S.histLog));
  plot.appendChild(chart);
  $('#histSub').textContent =
    `${hist.total.toLocaleString()} scored centers · `
    + `${hist.above.toLocaleString()} at or above ${hist.threshold.toFixed(2)}`;

  const legend = $('#histLegend');
  legend.textContent = '';
  const row = h('div', 'legend-row');
  for (const [cls, label] of [['rock', 'counted as rock'],
                              ['quiet', 'below threshold']]) {
    const item = h('span', 'legend-item');
    item.appendChild(h('span', `legend-swatch swatch-${cls}`));
    item.appendChild(document.createTextNode(label));
    row.appendChild(item);
  }
  legend.appendChild(row);
  // The counts are reachable without reading the bars.
  legend.appendChild(window.Charts.tableView(
    ['probability', 'centers'],
    hist.counts.map((c, i) => [
      `${hist.edges[i].toFixed(2)} – ${hist.edges[i + 1].toFixed(2)}`,
      c.toLocaleString(),
    ]),
  ));
}

/* -- trends --------------------------------------------------------------- */
/* Compact rail sparks rather than full plots: one measure per spark, latest
 * value printed right in the header, drawn at their true pixel size so the
 * type stays readable at rail width (a 720-unit viewBox scaled into ~300px
 * turns 11px labels into 5px ones). */
function renderTrends(scene) {
  const hist = scene.history;
  const host = $('#trends');
  const card = $('#trendCard');
  if (!hist || hist.t.length < 2) { card.hidden = true; return; }
  card.hidden = false;
  host.textContent = '';
  for (const s of hist.series) {
    if (!s.values.some((v) => v > 0)) continue;
    host.appendChild(window.Views.spark(s));
  }
  if (!host.children.length) card.hidden = true;
}

/* ----------------------------------------------------------------- poll */
function setLink(up) {
  $('#linkState').classList.toggle('is-down', !up);
  $('#linkState').title = up
    ? 'Connected to the live process'
    : 'The live process is not answering — its window may have been closed';
  document.body.classList.toggle('is-down', !up);
}

async function poll() {
  try {
    apply(await api('/api/state'));
    setLink(true);
  } catch (e) {
    setLink(false);
  }
}

async function pollScene() {
  try {
    S.scene = await api('/api/scene');
    renderViews();
  } catch (e) { /* the state poll owns the connection indicator */ }
}

/* ---------------------------------------------------------------- theme */
function initTheme() {
  const saved = localStorage.getItem('rocklabel-theme');
  const theme = saved === 'light' ? 'light' : 'dark';   // dark is the default
  document.documentElement.dataset.theme = theme;
  $('#themeToggle').onclick = () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('rocklabel-theme', next);
    // SVG reads the tokens live; a canvas baked them in at draw time.
    if (S.scene) renderViews();
  };
}

async function main() {
  initTheme();
  try {
    renderSchema(await api('/api/schema'));
  } catch (e) {
    setLink(false);
    console.error('schema:', e.message);
    return;
  }
  wireMapHover();
  $('#histScale').onclick = () => {
    S.histLog = !S.histLog;
    $('#histScale').textContent = S.histLog ? 'Log counts' : 'Linear counts';
    $('#histScale').setAttribute('aria-pressed', String(S.histLog));
    if (S.scene) renderHistogram(S.scene);
  };
  // Redraw the canvas views when the theme flips or the window resizes: both
  // change colors or pixel geometry the canvas baked in at draw time.
  addEventListener('resize', () => { if (S.scene) renderViews(); });

  await poll();
  await pollScene();
  setInterval(poll, POLL_MS);
  setInterval(pollScene, SCENE_POLL_MS);
}

if (typeof document !== 'undefined' && document.getElementById('panels')) {
  main();
}

/* Exposed for tests/frontend/run_webui.mjs, which boots this file against a
   minimal DOM. Harmless in a browser. */
if (typeof globalThis !== 'undefined') {
  globalThis.__live = { S, renderSchema, apply, renderControl, fmt, main,
                        renderViews, renderMap, renderHistogram, renderTrends };
}
