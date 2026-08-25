/* Execute the dashboard's whole render path in node against real API payloads.
 *
 * Usage: node run_dashboard.mjs <dashboard-dir> <fixtures.json>
 *
 * There is no browser in CI (and none that can be driven on a Wayland desktop),
 * so instead of screenshotting the page this loads index.html into a minimal
 * DOM, stubs fetch with recorded API responses, and then *clicks everything*:
 * every nav item, every data tab, every command drawer, the runs board's rows,
 * chips and note editor, every preset and help toggle, and the Run button. Any
 * exception thrown while rendering fails the run.
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
globalThis.fetch = async (path, opts) => {
  const key = path.split('?')[0];
  const data = fixtures[key] ?? fixtures[key.replace(/\/j\d+\b/, '/JOB')];
  if (data === undefined) fail('fetch', `no fixture for ${key}`);
  if (opts && opts.method && opts.method !== 'GET' && !/notes|run|rename|delete|stop|rerun|preview/.test(key)) {
    fail('fetch', `unexpected ${opts.method} ${key}`);
  }
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
const settle = async () => {
  for (let i = 0; i < 6; i++) await new Promise((r) => setImmediate(r));
};

// ---- pipeline view (the initial one): cards for the main loop, flow, tiles
let expanders = 0;
let charts = 0;
const pipeCards = all('#pipelineGroups .cmd-card').length;
all('.tile').forEach((b) => click(b, 'stat tile'));
// back from wherever the tiles jumped to; every handler goes through click()
// so a render throw becomes a FAILURES entry rather than a dead process
click(body.querySelector('[data-view="pipeline"]'), 'nav pipeline');
all('.flow-stage').forEach((b) => click(b, 'flow stage'));  // jumps to tools
click(all('#stageChips .chip[data-stage="all"]')[0], 'tools chip all');

// ---- tools view: cards for everything else, stage filter, search
click(body.querySelector('[data-view="tools"]'), 'nav tools');
const toolCards = all('#commandGroups .cmd-card').length;
all('.chip', body.querySelector('#stageChips')).forEach((c) => click(c, 'stage chip'));
click(all('#stageChips .chip[data-stage="all"]')[0], 'stage chip all');
const search = body.querySelector('#cmdSearch');
search.value = 'inspect';
search.dispatchEvent({ type: 'input', target: search });   // debounce stub runs it now
click(body.querySelector('[data-view="tools"]'), 're-render tools');
const searchedCards = all('#commandGroups .cmd-card').length;
search.value = '';
search.dispatchEvent({ type: 'input', target: search });

// ---- drawers: open every command card on both views, then drive one form
[...all('#pipelineGroups .cmd-card'), ...all('#commandGroups .cmd-card')]
  .forEach((c) => click(c, 'command drawer'));
all('.chip').forEach((c) => click(c, 'loose chip'));   // fold/status chips too
const helps = all('.help-btn').length;
all('.help-btn').forEach((b) => click(b, 'help toggle'));
all('.preset-btn').forEach((b) => click(b, 'preset'));
all('.checkline').forEach((l) => click(l, 'archived toggle'));
click(body.querySelector('#copyCmd'), 'copy command');
click(body.querySelector('#runBtn'), 'run');
await settle();
click(body.querySelector('#drawerClose'), 'drawer close');
click(body.querySelector('#scrim'), 'scrim close');

// ---- runs board: suites, ranking rows, folds, note editing
click(body.querySelector('[data-view="runs"]'), 'nav runs');
await settle();
const suiteChips = all('#suiteChips .chip').length;
const boardRows = all('.board-row-main').length;
if (boardRows) {
  click(all('.board-row-main')[0], 'board row expand');
}
const foldChips = all('.fold-chip').length;
charts += all('.chart-figure').length;
// The retired flat compare experiments are still on disk in this fixture;
// they must not surface anywhere on the board.
const boardText = body.querySelector('#boardBody').textContent.toLowerCase();
if (boardText.includes('compare')) errors.push('retired compare runs still show on Runs');
click(body.querySelector('#boardRefresh'), 'board refresh');
await settle();

// ---- training view: whatever sweep is on the GPU right now
click(body.querySelector('[data-view="training"]'), 'nav training');
await settle();
const trainCards = all('.train-run').length;
const foldTiles = all('.fold-tile').length;
const epochCharts = all('#trainingBody .chart-figure').length;
const trainText = (body.querySelector('#trainingBody').textContent || '').toLowerCase();
if (trainText.includes('compare')) errors.push('retired compare runs still show on Training');

// Note round trip against the recorded PUT response: open the editor, type,
// save, and expect the row's note text to become the server's answer.
let noteSaved = null;
const editBtns = all('.note-edit-btn');
if (editBtns.length) {
  click(editBtns[0], 'note edit open');
  const ta = body.querySelector('.note-editor textarea');
  if (!ta) errors.push('note editor did not open');
  else {
    ta.value = 'hand-typed note';
    const save = body.querySelector('.note-editor .btn-primary');
    click(save, 'note save');
    await settle();
    const texts = all('.board-note .note-text').map((n) => n.textContent);
    noteSaved = texts.includes(fixtures['/api/board/notes'].note);
  }
}

// ---- data view: tabs, row expansion, rename + delete round trips
click(body.querySelector('[data-view="data"]'), 'nav data');
for (const t of all('#dataTabs .tab')) {
  click(t, `data tab ${t.dataset.tab}`);
  for (const e of all('.expander')) { click(e, 'expander'); expanders++; }
}

// Rename and delete are the only controls that write to the project without a
// job, and all three data tabs carry them. Drive both round trips on each tab:
// open the panel, fill it in, commit. Posting then re-renders from /api/state,
// so settle between phases — a broken round trip has to fail here, not after the
// checks below have already run.
let renames = 0;
let deletes = 0;
for (const tab of ['recordings', 'labels', 'datasets']) {
  // Scope to the active panel: switching tabs leaves the previous tab's DOM
  // in place, so a page-wide selector would count stale rows from other tabs.
  const panel = () => body.querySelector(`#tab-${tab}`);
  const open = () => click(body.querySelector(`[data-tab="${tab}"]`), `open ${tab}`);
  open();
  const renameBtns = all(`#tab-${tab} .rename-btn`);
  renames += renameBtns.length;
  renameBtns.forEach((b) => click(b, `rename ${tab}`));
  all(`#tab-${tab} .rename-input`).forEach((i) => { i.value = `renamed_${tab}`; });
  all(`#tab-${tab} .rename .btn-primary`).forEach((b) => click(b, `rename save ${tab}`));
  await settle();

  open();
  const deleteBtns = all(`#tab-${tab} .delete-btn`);
  deletes += deleteBtns.length;
  deleteBtns.forEach((b) => click(b, `delete ${tab}`));
  const confirms = all(`#tab-${tab} .confirm .btn-danger`);
  if (confirms.length !== deleteBtns.length) {
    errors.push(`${tab}: delete did not ask before doing it`);
  }
  confirms.forEach((b) => click(b, `delete confirm ${tab}`));
  all(`#tab-${tab} .rename .btn-ghost`).forEach((b) => click(b, `cancel ${tab}`));
  await settle();
}

// ---- live view: sensor block, launch buttons, and the panel-embed latch
click(body.querySelector('[data-view="live"]'), 'nav live');
await settle();
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
click(body.querySelector('[data-view="live"]'), 're-render, same job');   // re-render, same job
panelReframed = frame.__srcWrites > 0;
click(body.querySelector('#panelReload'), 'panel reload');
click(body.querySelector('#panelPop'), 'panel pop-out');
all('.quick-btn').forEach((b) => click(b, 'quick action'));
await settle();

// ---- jobs view: history selection and the rerun slot
click(body.querySelector('[data-view="jobs"]'), 'nav jobs');
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
const catalog = fixtures['/api/catalog'];
const checks = [
  ['7 views wired', all('.nav-item').length === 7],
  ['every command got a card', pipeCards + toolCards === catalog.commands.length],
  ['pipeline leads with the main loop',
    pipeCards === catalog.commands.filter((c) => c.tier !== 'tool').length],
  ['hero figure filled', el('heroF1').textContent !== '—'],
  ['4 stat tiles', el('tiles').children.length === 4],
  // Read the count off the catalog rather than hardcoding it: adding a stage
  // is a normal thing to do, and a hardcoded 7 turns that into a test failure
  // that says nothing about what broke.
  ['every pipeline stage drawn',
    el('flow').children.length === catalog.stages.length],
  ['activity list populated', el('activity').children.length > 0],
  ['command search filters', searchedCards > 0 && searchedCards <= toolCards],
  ['rows expanded', expanders > 0],
  ['board loaded rows', boardRows > 0],
  ['suite chips match the board',
    suiteChips === 1 + fixtures['/api/board'].suites.length],
  ['fold detail opened', boardRows === 0 || foldChips > 0],
  ['board note round trip', fixtures['/api/board'].rows.length === 0 || noteSaved === true],
  ['running viewer got its control panel embedded', panelShown],
  ['the panel is framed at the job\'s announced url',
    panelSrc === fixtures['/api/jobs'].jobs[0].panel_url],
  ['a re-render does not reload the framed panel', !panelReframed],
  ['rename and delete on all three data tabs', renames >= 3 && deletes === renames],
  ['training view shows the live sweep', trainCards >= 1 && foldTiles >= 3],
  ['training view drew the epoch curve', epochCharts >= 1],
  ['runs view drew charts', charts > 0],
  ['help buttons exist', helps > 0],
  ['run form has fields', all('#runForm [data-param]').length > 0],
  ['job history lists past jobs', history.length === 2],
  ['rerun offered on a finished job', rerunOnFinished],
  ['rerun withheld while the job runs', rerunOnRunning],
];
checks.forEach(([name, ok]) => { if (!ok) errors.push(`check failed: ${name}`); });

if (errors.length) {
  console.error(`FAILURES (${errors.length}):\n` + errors.slice(0, 8).join('\n'));
  process.exit(1);
}
console.log(`ok — 7 views, ${pipeCards}+${toolCards} command drawers, `
          + `${boardRows} board rows, ${expanders} expanders, ${charts} charts, `
          + `${helps} help buttons, ${renames} renames, ${deletes} deletes`);
