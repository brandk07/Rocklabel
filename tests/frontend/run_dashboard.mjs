/* Execute the dashboard's whole render path in node against real API payloads.
 *
 * Usage: node run_dashboard.mjs <dashboard-dir> <fixtures.json>
 *
 * There is no browser in CI (and none that can be driven on a Wayland desktop),
 * so instead of screenshotting the page this loads index.html into a minimal
 * DOM, stubs fetch with recorded API responses, and then *clicks everything*:
 * every nav item, every data tab, every command drawer, every row expander,
 * every preset and help toggle, and the Run button. Any exception thrown while
 * rendering fails the run.
 *
 * It checks that the page renders, not that it looks right — the visual pass is
 * a human opening it. What it reliably catches is the class of bug that costs
 * the most: a renamed field, a missing element id, a chart fed a NaN.
 */
import fs from 'node:fs';
import { El, parse } from './minidom.mjs';

const [ROOT, FIXTURES] = process.argv.slice(2);
const errors = [];
const fail = (where, e) => errors.push(`${where}: ${(e && e.stack) || e}`);

// --------------------------------------------------------------------- DOM
const html = fs.readFileSync(`${ROOT}/templates/index.html`, 'utf8')
  .replace(/\{\{[^}]*\}\}/g, 'x');          // strip Jinja placeholders
const body = parse(html);

globalThis.document = {
  body,
  documentElement: Object.assign(new El('html'), { dataset: { theme: 'dark' } }),
  createElement: (t) => new El(t),
  createElementNS: (_ns, t) => new El(t),
  createTextNode: (t) => Object.assign(new El('#text'), { _text: String(t) }),
  createDocumentFragment: () => new El('#fragment'),
  querySelector: (s) => body.querySelector(s),
  querySelectorAll: (s) => body.querySelectorAll(s),
  addEventListener() {},
  hidden: false,
};
globalThis.window = { innerWidth: 1400, innerHeight: 900, scrollTo() {}, open() {} };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '#3987e5' });
globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.matchMedia = () => ({ matches: false });
globalThis.navigator = { clipboard: { writeText: async () => {} } };
globalThis.setInterval = () => 0;          // no polling in the harness
globalThis.clearTimeout = () => {};
globalThis.setTimeout = (fn) => {          // run debounced work immediately
  try { fn(); } catch (e) { fail('setTimeout', e); }
  return 0;
};

// ----------------------------------------------------------------- network
const fixtures = JSON.parse(fs.readFileSync(FIXTURES, 'utf8'));
globalThis.fetch = async (path) => {
  const key = path.split('?')[0];
  const data = fixtures[key] ?? fixtures[key.replace(/\/j\d+\b/, '/JOB')];
  if (data === undefined) fail('fetch', `no fixture for ${key}`);
  return { ok: true, status: 200, statusText: 'OK', json: async () => data ?? {} };
};

// ------------------------------------------------------------------- boot
new Function(fs.readFileSync(`${ROOT}/static/charts.js`, 'utf8'))();
new Function(fs.readFileSync(`${ROOT}/static/app.js`, 'utf8'))();
for (let i = 0; i < 4; i++) await new Promise((r) => setImmediate(r));

// -------------------------------------------------------------- click all
const click = (n, what) => {
  try { n.onclick && n.onclick({ preventDefault() {}, stopPropagation() {} }); }
  catch (e) { fail(what, e); }
};
const all = (sel) => body.querySelectorAll(sel);

const views = all('.nav-item');
views.forEach((b) => click(b, `view ${b.dataset.view}`));

body.querySelector('[data-view="commands"]').onclick();
const cards = all('.cmd-card');
cards.forEach((c) => click(c, 'command drawer'));
all('.chip').forEach((c) => click(c, 'stage chip'));

body.querySelector('[data-view="data"]').onclick();
let expanders = 0;
for (const t of all('#dataTabs .tab')) {
  click(t, `data tab ${t.dataset.tab}`);
  for (const e of all('.expander')) { click(e, 'expander'); expanders++; }
}

// Rename and delete are the only controls that write to the project without a
// job, and all three data tabs carry them. Drive both round trips on each tab:
// open the panel, fill it in, commit. Posting then re-renders from /api/state,
// so settle between phases — a broken round trip has to fail here, not after the
// checks below have already run.
const settle = async () => {
  for (let i = 0; i < 4; i++) await new Promise((r) => setImmediate(r));
};
let renames = 0;
let deletes = 0;
for (const tab of ['recordings', 'labels', 'datasets']) {
  const open = () => body.querySelector(`[data-tab="${tab}"]`).onclick();
  open();
  const renameBtns = all('.rename-btn');
  renames += renameBtns.length;
  renameBtns.forEach((b) => click(b, `rename ${tab}`));
  all('.rename-input').forEach((i) => { i.value = `renamed_${tab}`; });
  all('.rename .btn-primary').forEach((b) => click(b, `rename save ${tab}`));
  await settle();

  open();
  const deleteBtns = all('.delete-btn');
  deletes += deleteBtns.length;
  deleteBtns.forEach((b) => click(b, `delete ${tab}`));
  const confirms = all('.confirm .btn-danger');
  if (confirms.length !== deleteBtns.length) {
    errors.push(`${tab}: delete did not ask before doing it`);
  }
  confirms.forEach((b) => click(b, `delete confirm ${tab}`));
  all('.rename .btn-ghost').forEach((b) => click(b, `cancel ${tab}`));
  await settle();
}

body.querySelector('[data-view="models"]').onclick();
for (const e of all('.expander')) { click(e, 'run expander'); expanders++; }
all('.chip').forEach((c) => click(c, 'metric chip'));
const charts = all('.chart-figure').length;

body.querySelector('[data-view="live"]').onclick();
// The running viewer's control panel is embedded here. It must be framed once
// and then left alone: re-setting src on every 5 s poll would reload the page
// inside the frame and throw away whatever the user was doing in it.
const frame = body.querySelector('#panelFrame');
const panelShown = body.querySelector('#panelCard').hidden === false;
const panelSrc = frame.src;
let panelReframed = false;
frame.__srcWrites = 0;
Object.defineProperty(frame, 'src', {
  get() { return this._src; },
  set(v) { this._src = v; this.__srcWrites++; },
  configurable: true,
});
frame._src = panelSrc;
body.querySelector('[data-view="live"]').onclick();   // re-render, same job
panelReframed = frame.__srcWrites > 0;
click(body.querySelector('#panelReload'), 'panel reload');
click(body.querySelector('#panelPop'), 'panel pop-out');
all('.quick-btn').forEach((b) => click(b, 'quick action'));
all('.preset-btn').forEach((b) => click(b, 'preset'));
const helps = all('.help-btn').length;
all('.help-btn').forEach((b) => click(b, 'help toggle'));
click(body.querySelector('#runBtn'), 'run');
await settle();

// The history list is the session's record, and a finished row can be launched
// again from it. Rerun shares its slot with Stop: offered once a job is over,
// withheld while it is still running.
body.querySelector('[data-view="jobs"]').onclick();
const history = all('#jobList li');
click(history[0], 'select the running job');
const rerunOnRunning = body.querySelector('#jobRerun').hidden;
click(history[1], 'select the finished job');
await settle();
const rerunOnFinished = body.querySelector('#jobRerun').hidden === false;
click(body.querySelector('#jobRerun'), 'rerun');
await settle();

// ------------------------------------------------------------------ checks
const el = (id) => body.querySelector('#' + id);
const checks = [
  ['6 views wired', views.length === 6],
  ['every command got a card', cards.length === fixtures['/api/catalog'].commands.length],
  ['hero figure filled', el('heroF1').textContent !== '—'],
  ['4 stat tiles', el('tiles').children.length === 4],
  // Read the count off the catalog rather than hardcoding it: adding a stage
  // is a normal thing to do, and a hardcoded 6 turns that into a test failure
  // that says nothing about what broke.
  ['every pipeline stage drawn',
    el('flow').children.length === fixtures['/api/catalog'].stages.length],
  ['activity list populated', el('activity').children.length > 0],
  ['rows expanded', expanders > 0],
  ['running viewer got its control panel embedded', panelShown],
  ['the panel is framed at the job\'s announced url',
    panelSrc === fixtures['/api/jobs'].jobs[0].panel_url],
  ['a re-render does not reload the framed panel', !panelReframed],
  ['rename and delete on all three data tabs', renames >= 3 && deletes === renames],
  ['models view drew charts', charts > 0],
  ['help buttons exist', helps > 0],
  ['run form has fields', all('#runForm [data-param]').length > 0],
  ['job history lists past jobs', history.length === 2],
  ['rerun offered on a finished job', rerunOnFinished],
  ['rerun withheld while the job runs', rerunOnRunning],
];
checks.forEach(([name, ok]) => { if (!ok) errors.push(`check failed: ${name}`); });

if (errors.length) {
  console.error(`FAILURES (${errors.length}):\n` + errors.slice(0, 6).join('\n'));
  process.exit(1);
}
console.log(`ok — ${views.length} views, ${cards.length} command drawers, `
          + `${expanders} expanders, ${charts} charts, ${helps} help buttons, `
          + `${renames} renames, ${deletes} deletes`);
