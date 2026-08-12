/* Execute the live control panel's render path in node against real payloads.
 *
 * Usage: node run_webui.mjs <webui-dir> <fixtures.json>
 *
 * Same arrangement as run_dashboard.mjs: no browser, so live.html + live.js are
 * loaded into the minimal DOM in minidom.mjs, fetch is stubbed with responses
 * recorded from the real Flask app, and then every control is exercised —
 * toggled, dragged, typed into, clicked. Any exception fails the run.
 *
 * The two behaviours worth pinning down here are the ones that are invisible in
 * a screenshot and infuriating in use:
 *
 *   1. every control the schema advertises actually renders, and
 *   2. a poll arriving mid-drag does not yank the control out from under you.
 */
import fs from 'node:fs';
import { El, parse } from './minidom.mjs';

const [ROOT, FIXTURES] = process.argv.slice(2);
const errors = [];
const fail = (where, e) => errors.push(`${where}: ${(e && e.stack) || e}`);

// --------------------------------------------------------------------- DOM
const html = fs.readFileSync(`${ROOT}/templates/live.html`, 'utf8')
  .replace(/\{\{[^}]*\}\}/g, 'x');          // strip Jinja placeholders
const body = parse(html);

// A 2D context that records what was asked of it. The map is a canvas, so the
// only way to assert it drew anything is to count the drawing calls — and a
// count is exactly what tells us the detections reached the picture.
function fakeCtx() {
  const calls = { arc: 0, fill: 0, stroke: 0, drawImage: 0, fillText: 0, putImageData: 0 };
  const ctx = {
    calls,
    canvas: null,
    createImageData: (w, hh) => ({ width: w, height: hh, data: new Uint8ClampedArray(w * hh * 4) }),
    putImageData() { calls.putImageData++; },
    drawImage() { calls.drawImage++; },
    arc() { calls.arc++; },
    fill() { calls.fill++; },
    stroke() { calls.stroke++; },
    fillText() { calls.fillText++; },
  };
  for (const noop of ['setTransform', 'clearRect', 'save', 'restore', 'beginPath',
                      'moveTo', 'lineTo', 'closePath', 'translate', 'rotate',
                      'setLineDash', 'rect']) ctx[noop] = () => {};
  return ctx;
}
const canvasCtx = new Map();
function makeEl(tag) {
  const n = new El(tag);
  if (String(tag).toLowerCase() === 'canvas') {
    n.getContext = () => {
      if (!canvasCtx.has(n)) canvasCtx.set(n, fakeCtx());
      return canvasCtx.get(n);
    };
  }
  return n;
}
// Elements parsed out of the HTML are plain Els; give the map canvas a context.
for (const c of body.querySelectorAll('canvas')) {
  c.getContext = () => {
    if (!canvasCtx.has(c)) canvasCtx.set(c, fakeCtx());
    return canvasCtx.get(c);
  };
}

globalThis.document = {
  body,
  documentElement: Object.assign(new El('html'), { dataset: { theme: 'dark' } }),
  createElement: makeEl,
  createElementNS: (_ns, t) => new El(t),
  createTextNode: (t) => Object.assign(new El('#text'), { _text: String(t) }),
  createDocumentFragment: () => new El('#fragment'),
  querySelector: (s) => body.querySelector(s),
  querySelectorAll: (s) => body.querySelectorAll(s),
  getElementById: (id) => body.querySelector('#' + id),
  addEventListener() {},
  activeElement: null,
  title: '',
};
globalThis.window = { innerWidth: 1400, innerHeight: 900, devicePixelRatio: 2 };
// views.js reads the design tokens off :root; charts.js does the same.
const TOKENS = {
  '--surface-1': '#1a1a19', '--surface-2': '#232322', '--surface-3': '#2c2c2a',
  '--plane': '#0d0d0d', '--text-primary': '#ffffff', '--text-secondary': '#c3c2b7',
  '--text-muted': '#898781', '--gridline': '#2c2c2a', '--baseline': '#383835',
  '--series-1': '#3987e5', '--series-2': '#d95926', '--series-3': '#199e70',
  '--series-4': '#c98500',
};
globalThis.getComputedStyle = () => ({
  getPropertyValue: (name) => TOKENS[name] || '#888888',
});
globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.matchMedia = () => ({ matches: false });
globalThis.addEventListener = () => {};
globalThis.setInterval = () => 0;          // the harness polls by hand
globalThis.setTimeout = (fn) => { try { fn(); } catch (e) { fail('setTimeout', e); } return 0; };
globalThis.clearTimeout = () => {};

// ----------------------------------------------------------------- network
const fixtures = JSON.parse(fs.readFileSync(FIXTURES, 'utf8'));
const posted = [];
globalThis.fetch = async (path, options) => {
  if (options && options.method === 'POST') {
    posted.push({ path, body: JSON.parse(options.body) });
    // The real endpoints echo a fresh snapshot back; so does this.
    return { ok: true, status: 200, json: async () => fixtures['/api/state'] };
  }
  const data = fixtures[path.split('?')[0]];
  if (data === undefined) fail('fetch', `no fixture for ${path}`);
  return { ok: true, status: 200, statusText: 'OK', json: async () => data ?? {} };
};

// ------------------------------------------------------------------- boot
// charts.js lives with the dashboard and is borrowed by this page; views.js
// holds the map and the histogram. live.js drives both.
new Function(fs.readFileSync(`${ROOT}/../../dashboard/static/charts.js`, 'utf8'))();
new Function(fs.readFileSync(`${ROOT}/static/views.js`, 'utf8'))();
new Function(fs.readFileSync(`${ROOT}/static/live.js`, 'utf8'))();
const live = globalThis.__live;
if (!live) { console.error('live.js did not expose its test hooks'); process.exit(1); }
try { await live.main(); } catch (e) { fail('boot', e); }
const settle = async () => { for (let i = 0; i < 4; i++) await new Promise((r) => setImmediate(r)); };
await settle();

// -------------------------------------------------------------- drive all
const all = (sel) => body.querySelectorAll(sel);
const el0 = (id) => body.querySelector('#' + id);
const click = (n, what) => {
  try { n && n.onclick && n.onclick({ preventDefault() {}, stopPropagation() {} }); }
  catch (e) { fail(what, e); }
};
const schema = fixtures['/api/schema'];
const controls = schema.sections.flatMap((s) => s.controls);
const settable = controls.filter((c) => c.kind !== 'readout' && c.kind !== 'action');
// The replay section is rendered into the sticky transport bar, not a card, so
// its controls are driven by hand below and carry no per-control help button.
const carded = schema.sections.filter((s) => s.id !== 'replay').flatMap((s) => s.controls);

// Every help toggle opens and closes.
const helps = all('.ctl-help-btn');
helps.forEach((b) => { try { b.onclick(); b.onclick(); } catch (e) { fail('help', e); } });

// Every action button fires — the cards' and the transport bar's.
const actions = all('.ctl-actions .btn');
actions.forEach((b) => { try { b.onclick(); } catch (e) { fail('action', e); } });
const transportBtns = [el0('playBtn'), el0('restartBtn')].filter(Boolean);
transportBtns.forEach((b) => { try { b.onclick(); } catch (e) { fail('transport', e); } });
await settle();

// Every setting is written: checkboxes flipped, selects moved, numbers typed,
// sliders dragged and released.
let wrote = 0;
for (const c of settable) {
  const node = live.S.nodes[c.id];
  if (!node || !node.input) { errors.push(`no input rendered for ${c.id}`); continue; }
  const el = node.input;
  try {
    if (c.kind === 'bool') {
      el.checked = !el.checked;
      el.onchange();
    } else if (c.kind === 'enum') {
      const last = c.choices[c.choices.length - 1];
      el.value = String(last.value);
      el.onchange();
    } else {
      el.value = String(c.min != null ? c.min : 0);
      if (el.oninput) el.oninput();
      el.onchange();
    }
    wrote++;
  } catch (e) { fail(`set ${c.id}`, e); }
}
await settle();

// A poll landing mid-drag must not move the control. This is the bug that makes
// a browser panel unusable: you drag the threshold, a 4 Hz poll arrives, and the
// slider snaps back to where the server last saw it.
let dragHeld = true;
const slider = settable.find((c) => c.kind === 'float' && live.S.nodes[c.id]
                                    && live.S.nodes[c.id].readback);
if (!slider) {
  errors.push('no slider rendered — nothing to test the mid-drag guard with');
} else {
  const node = live.S.nodes[slider.id];
  live.S.pending[slider.id] = 0;          // clear the post-write settle window
  live.S.dragging = slider.id;            // ...so only the drag guard is left
  node.input.value = String(slider.max);
  live.apply(fixtures['/api/state']);
  dragHeld = String(node.input.value) === String(slider.max);
  live.S.dragging = null;
}

// Released, the same poll must take effect — otherwise the panel would go deaf
// to the Open3D window's keyboard shortcuts.
live.S.pending = {};
live.apply(fixtures['/api/state']);
const mirrored = settable.every((c) => {
  const node = live.S.nodes[c.id];
  if (!node || !node.input || !(c.id in fixtures['/api/state'].values)) return true;
  const want = fixtures['/api/state'].values[c.id];
  return c.kind === 'bool'
    ? node.input.checked === !!want
    : String(node.input.value) === String(want);
});

// ------------------------------------------------------------- extra views
// The map, the histogram and the trends all render off one /api/scene payload.
const scene = fixtures['/api/scene'];
let viewErr = null;
try { live.renderViews(); } catch (e) { viewErr = (e && e.stack) || String(e); }
if (viewErr) fail('renderViews', viewErr);

const mapCtx = canvasCtx.get(body.querySelector('#mapCanvas'));
const nDet = scene.detections.rows.length;
// One arc per detection, plus the scoring-region ring.
const drewEveryDetection = !!mapCtx && mapCtx.calls.arc >= nDet;
const drewTerrain = !!mapCtx && mapCtx.calls.drawImage > 0;
const drewScaleBar = !!mapCtx && mapCtx.calls.fillText > 0;

// The height ramp must be labelled at both ends — a value scale readable only
// as colour is the thing the accessibility pass exists to catch.
const rampEnds = body.querySelectorAll('.legend-ramp-end').length;
const markSwatches = body.querySelectorAll('#mapLegend .legend-swatch').length;

// Toggling to the table view must produce real rows, not just reveal a box.
click(el0('mapTable'), 'map table toggle');
const mapTableRows = body.querySelectorAll('#mapTableView tbody tr').length;

const histBars = body.querySelectorAll('#histPlot path').length;
const histSwatches = body.querySelectorAll('#histLegend .legend-swatch').length;
const histTableRows = body.querySelectorAll('#histLegend tbody tr').length;
click(el0('histScale'), 'histogram scale toggle');       // log -> linear
const histBarsLinear = body.querySelectorAll('#histPlot path').length;
click(el0('histScale'), 'histogram scale toggle back');

// One plot per series that has data — never two measures on one axis.
const trendPlots = body.querySelectorAll('#trends .chart-figure').length;
const trendsWithData = scene.history.series.filter(
  (s) => s.values.some((v) => v > 0)).length;
// A single series carries no legend box; its caption names it.
const trendLegends = body.querySelectorAll('#trends .legend').length;

// ------------------------------------------------------------------ checks
const el = el0;
const cards = all('#panels .card');       // the control cards, not the view cards
const readouts = controls.filter((c) => c.kind === 'readout');
const checks = [
  ['a card per non-transport section', cards.length === schema.sections.length - 1],
  ['every settable control rendered', wrote === settable.length],
  ['every carded control with help got a "?"', helps.length === carded.filter((c) => c.help).length],
  ['every readout filled in', readouts.every((c) => {
    const n = live.S.nodes[c.id];
    return n && n.out && n.out.textContent && n.out.textContent !== '—';
  })],
  ['status strip populated', el('statusStrip').children.length > 0],
  ['transport shown for a replay', el('transport').hidden === false],
  ['transport time rendered', /\d/.test(el('transportTime').textContent)],
  ['posts reached the server', posted.length >= wrote + actions.length + transportBtns.length],
  ['a mid-drag poll does not move the slider', dragHeld],
  ['a released control mirrors the server', mirrored],
  // -- the extra views --
  ['overhead drew the fused terrain', drewTerrain],
  ['overhead drew every detection', drewEveryDetection],
  ['overhead drew a scale bar', drewScaleBar],
  ['the height ramp is labelled at both ends', rampEnds === 2],
  ['rock and sensor both have named swatches', markSwatches === 2],
  ['the map has a table view with rows', mapTableRows > 0],
  ['histogram drew a bar per bin', histBars >= scene.histogram.counts.length],
  ['histogram survives the log/linear toggle', histBarsLinear >= scene.histogram.counts.length],
  ['histogram names both bar classes', histSwatches === 2],
  ['histogram has a table view with a row per bin',
    histTableRows === scene.histogram.counts.length],
  ['one trend plot per series with data', trendPlots === trendsWithData && trendPlots > 0],
  ['single-series trends carry no legend box', trendLegends === 0],
];
checks.forEach(([name, ok]) => { if (!ok) errors.push(`check failed: ${name}`); });

if (errors.length) {
  console.error(`FAILURES (${errors.length}):\n` + errors.slice(0, 6).join('\n'));
  process.exit(1);
}
console.log(`ok — ${cards.length} cards, ${wrote} settings, ${actions.length} actions, `
          + `${readouts.length} readouts, ${helps.length} help toggles, `
          + `${posted.length} posts, ${nDet} map marks, ${histBars} hist bars, `
          + `${trendPlots} trend plots`);
