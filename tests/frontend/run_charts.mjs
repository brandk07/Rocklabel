/* Smoke-test charts.js on the data shapes the dashboard actually feeds it.
 *
 * Usage: node run_charts.mjs <dashboard-dir>
 *
 * The cases that matter are the awkward ones: a fold that has not been
 * evaluated yet (NaN), a zero-valued segment, and single-point series.
 */
import fs from 'node:fs';
import { El } from './minidom.mjs';

const ROOT = process.argv[2];
const errors = [];

globalThis.document = {
  body: new El('body'),
  createElement: (t) => new El(t),
  createElementNS: (_ns, t) => new El(t),
  createTextNode: (t) => Object.assign(new El('#text'), { _text: String(t) }),
};
globalThis.window = { innerWidth: 1400, innerHeight: 900 };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '#3987e5' });

new Function(fs.readFileSync(`${ROOT}/static/charts.js`, 'utf8'))();
const C = globalThis.window.Charts;
const count = (n, tag) => [...n.walk()].filter((x) => x.tagName === tag).length;
const check = (name, ok) => { if (!ok) errors.push(name); };

// Grouped columns: 2 models x 4 leave-one-run-out folds, one not yet evaluated.
const g = C.groupedColumns({
  categories: ['myroom1', 'myroom2', 'myroom3', 'myroom4'],
  series: [
    { name: 'pointnet', values: [0.58, 0.75, 0.87, 0.87] },
    { name: 'pointnet2', values: [0.61, NaN, 0.83, 0.85] },
  ],
  max: 1,
  caption: 'F1 on the held-out run.',
});
check('NaN fold must not draw a bar', count(g, 'path') === 7);
check('grouped columns need a table view', count(g, 'table') === 1);
check('grouped columns need a caption', count(g, 'figcaption') === 1);

// Lines: a 30-epoch history.csv.
const epochs = [...Array(30).keys()];
const l = C.lineChart({
  x: epochs,
  xLabel: 'epoch',
  series: [
    { name: 'train loss', values: epochs.map((e) => 0.5 / (e + 1)) },
    { name: 'val loss', values: epochs.map((e) => 0.4 / (e + 1)) },
    { name: 'val PR-AUC', values: epochs.map(() => 0.96) },
  ],
});
check('one end marker per line', count(l, 'circle') === 3);
check('line chart needs a table view', count(l, 'table') === 1);

// Proportion bar including an empty class.
const p = C.proportionBar([
  { label: 'rock samples', value: 5172, color: '#d95926' },
  { label: 'clear samples', value: 16645, color: '#3987e5' },
  { label: 'ignored', value: 0, color: '#199e70' },
]);
check('proportion bar renders bar + legend', p.children.length >= 2);

// Degenerate inputs must not throw.
try {
  C.groupedColumns({ categories: ['a'], series: [{ name: 's', values: [0] }], max: 1 });
  C.lineChart({ x: [0], series: [{ name: 's', values: [0] }] });
  C.proportionBar([]);
} catch (e) {
  errors.push(`degenerate input threw: ${e.message}`);
}

if (errors.length) {
  console.error('FAILURES:\n' + errors.join('\n'));
  process.exit(1);
}
console.log('ok — grouped columns, lines and proportion bar all render');
