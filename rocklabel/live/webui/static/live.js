/* rocklabel live control panel.
 *
 * One page, no build step, no network dependency — same rules as the dashboard.
 * The server owns the control catalog, so this file knows how to render a
 * *kind* of control (bool, float, int, enum, action, readout) and nothing about
 * which controls exist. Adding a knob is a line in spec.py.
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
  return !!n.input && document.activeElement === n.input;
}
function touch(id) { S.pending[id] = Date.now() + SETTLE_MS; }

/* A range input reports its value continuously but never "I am being dragged",
   and a mirror landing mid-drag is exactly the jump this page has to avoid. */
function trackDrag(id, el) {
  el.addEventListener('pointerdown', () => { S.dragging = id; });
  el.addEventListener('pointerup', () => { S.dragging = null; });
  el.addEventListener('pointercancel', () => { S.dragging = null; });
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
    for (const ch of c.choices || []) {
      const opt = h('option', null, ch.label);
      opt.value = String(ch.value);
      sel.appendChild(opt);
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
      input = h('input', 'ctl-slider');
      input.type = 'range';
      input.min = c.min != null ? c.min : 0;
      input.max = c.max != null ? c.max : 1;
      input.step = c.step != null ? c.step : 0.01;
      readback = h('span', 'ctl-readback', '—');
      trackDrag(c.id, input);
      input.oninput = () => {
        touch(c.id);
        readback.textContent = fmt(c, input.value);
      };
      input.onchange = () => setValue(c.id, input.value);
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
    // The transport lives in its own sticky bar above the cards.
    if (sec.id === 'replay') continue;
    const card = h('section', 'card');
    const head = h('div', 'card-head');
    const title = h('h2', null, sec.id === 'model' && schema.model
      ? `${sec.title} · ${schema.model}` : sec.title);
    head.appendChild(title);
    if (sec.blurb) head.appendChild(h('div', 'card-sub', sec.blurb));
    card.appendChild(head);
    for (const c of sec.controls) card.appendChild(renderControl(c));
    if (sec.id === 'model') {
      const warn = h('div', 'ctl-warning');
      warn.hidden = true;
      card.appendChild(warn);
      S.nodes['model.warning'] = { out: warn };
    }
    host.appendChild(card);
  }

  const hasReplay = schema.sections.some((s) => s.id === 'replay');
  $('#transport').hidden = !hasReplay;
  if (hasReplay) initTransport(schema);
  $('#modeBadge').textContent = schema.mode.toUpperCase();
  $('#subtitle').textContent = schema.subtitle;
  document.title = `rocklabel live · ${schema.subtitle}`;
}

/* ------------------------------------------------------------ transport */
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

  // Scrubbing is committed on release, not per pixel: a backward seek rewinds
  // the recording and re-fuses the whole map, so one seek per drag, not fifty.
  range.oninput = () => {
    S.scrubbing = true;
    $('#transportTime').textContent =
      `${Number(range.value).toFixed(1)} / ${schema.duration_sec.toFixed(1)} s`;
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
      if (node.readback) node.readback.textContent = fmt(c, v);
    }
  }

  const warnNode = S.nodes['model.warning'];
  if (warnNode) {
    const text = S.status['model.warning'] || '';
    warnNode.out.textContent = text;
    warnNode.out.hidden = !text;
  }

  renderStrip();
  if (S.transport) renderTransport();
}

function renderStrip() {
  const strip = $('#statusStrip');
  const cells = [
    ['rate', 'status.rate', 'rate_low'],
    ['cells', 'status.cells', null],
    ['accum', 'status.accum', 'accum_capped'],
    ['pose', 'status.pose', null],
    ['state', 'status.state', 'paused'],
  ];
  strip.textContent = '';
  for (const [name, id, warn] of cells) {
    if (!(id in S.status)) continue;
    const cell = h('div', 'live-strip-cell');
    cell.appendChild(h('span', 'k', name));
    const v = h('span', 'v', S.status[id]);
    if (warn && S.flags[warn]) v.classList.add('is-warn');
    cell.appendChild(v);
    strip.appendChild(cell);
  }
}

function renderTransport() {
  const t = S.transport;
  $('#playBtn').textContent = t.playing ? 'Pause' : 'Play';
  const range = $('#transportRange');
  if (!S.scrubbing && !fresh('replay.position') && !t.seeking) {
    range.value = t.position_sec;
    $('#transportTime').textContent =
      `${t.position_sec.toFixed(1)} / ${t.duration_sec.toFixed(1)} s`;
  }
}

/* ================================================================ VIEWS ==== */
/* The overhead map, the confidence histogram and the trend plots. All three
 * read one payload (/api/scene) and are redrawn wholesale — none of them holds
 * user state that a redraw could interrupt, unlike the controls. */

function renderViews() {
  const scene = S.scene;
  if (!scene) return;
  $('#views').hidden = false;
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
  const bev = scene.bev;
  const parts = [];
  parts.push(det.total
    ? `${det.total.toLocaleString()} detection${det.total === 1 ? '' : 's'}`
      + (det.shown < det.total ? ` (${det.shown.toLocaleString()} drawn)` : '')
    : 'No detections yet');
  if (bev) parts.push(`height ${bev.z_min.toFixed(2)} … ${bev.z_max.toFixed(2)} m`);
  $('#mapSub').textContent = parts.join(' · ');

  renderMapLegend(bev);
  if (S.mapTable) renderMapTable(det);
}

/* Identity is never colour-alone: the ramp gets its ends labelled and each
 * mark class gets a named swatch. */
function renderMapLegend(bev) {
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
  for (const [cls, label] of [['rock', 'rock ≥ threshold'], ['sensor', 'sensor']]) {
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
    window.Charts.showTip(e, 'Detection', [
      { label: 'probability', value: best.d[3].toFixed(3),
        color: window.Charts.cssVar('--series-2') },
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
  const chart = h('div', 'chart');   // app.css sizes .chart svg to its container
  chart.appendChild(window.Views.histogram(hist, S.histLog));
  plot.appendChild(chart);
  $('#histSub').textContent =
    `${hist.total.toLocaleString()} scored centers · `
    + `${hist.above.toLocaleString()} at or above ${hist.threshold.toFixed(2)}`;

  const legend = $('#histLegend');
  legend.textContent = '';
  for (const [cls, label] of [['rock', 'counted as rock'],
                              ['quiet', 'below threshold']]) {
    const item = h('span', 'legend-item');
    item.appendChild(h('span', `legend-swatch swatch-${cls}`));
    item.appendChild(document.createTextNode(label));
    legend.appendChild(item);
  }
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
function renderTrends(scene) {
  const hist = scene.history;
  const host = $('#trends');
  const card = $('#trendCard');
  if (!hist || hist.t.length < 2) { card.hidden = true; return; }
  card.hidden = false;
  host.textContent = '';
  // Elapsed seconds, thinned to whole numbers: the x axis is "how long ago",
  // and a wall clock would imply a precision the 1 Hz sampling does not have.
  const x = hist.t.map((t) => Math.round(t));
  for (const s of hist.series) {
    if (!s.values.some((v) => v > 0)) continue;
    const box = h('div', 'trend');
    box.appendChild(window.Charts.lineChart({
      x,
      xLabel: 't (s)',
      series: [{ name: s.label, values: s.values }],
      format: (v) => window.Views.compact(v),
      caption: `${s.label} — ${s.unit}`,
    }));
    host.appendChild(box);
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
  const theme = saved || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
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
  addEventListener('resize', () => { if (S.scene) renderMap(S.scene); });

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
