/* Minimal SVG chart primitives for the rocklabel dashboard.
 *
 * Deliberately hand-rolled rather than pulled from a CDN: the dashboard is a
 * localhost tool that must work with no network. Three forms cover everything
 * here — grouped columns (metric per fold per model), lines (training curves),
 * and a proportion bar (dataset class balance).
 *
 * House rules baked in, not left to the caller:
 *   - columns capped at 24px with a 4px rounded cap and a square baseline;
 *   - 2px lines, >=8px end markers with a 2px surface ring;
 *   - solid hairline gridlines, one step off the surface;
 *   - a legend whenever there are two or more series;
 *   - a hover tooltip on every mark, and a table-view twin so no value is
 *     reachable only by hovering.
 */
'use strict';

const SVGNS = 'http://www.w3.org/2000/svg';

/** Series colors resolve from CSS custom properties so the theme toggle just works. */
const SERIES_VARS = ['--series-1', '--series-2', '--series-3', '--series-4'];

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function seriesColor(i) { return cssVar(SERIES_VARS[i % SERIES_VARS.length]); }

function el(tag, attrs = {}, text) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}
function h(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

/** Column path: square at the baseline, `r` rounded at the data end. */
function columnPath(x, y, w, hgt, r) {
  const rr = Math.max(0, Math.min(r, w / 2, hgt));
  if (hgt <= 0) return '';
  return `M${x},${y + hgt} L${x},${y + rr} Q${x},${y} ${x + rr},${y}` +
         ` L${x + w - rr},${y} Q${x + w},${y} ${x + w},${y + rr}` +
         ` L${x + w},${y + hgt} Z`;
}

function niceTicks(max, count = 4) {
  if (!(max > 0)) return [0];
  const raw = max / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].find((s) => s * mag >= raw) * mag;
  const ticks = [];
  for (let v = 0; v <= max + step * 1e-9; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

/* ------------------------------------------------------------------ tooltip */
let tipNode = null;
function tooltip() {
  if (!tipNode) {
    tipNode = h('div', 'tooltip');
    tipNode.hidden = true;
    document.body.appendChild(tipNode);
  }
  return tipNode;
}
function showTip(evt, title, rows) {
  const tip = tooltip();
  tip.innerHTML = '';
  tip.appendChild(h('div', 'tooltip-title', title));
  for (const row of rows) {
    const line = h('div', 'tooltip-row');
    const k = h('span', 'k');
    if (row.color) {
      const sw = h('span', 'legend-swatch');
      sw.style.background = row.color;
      k.appendChild(sw);
    }
    k.appendChild(document.createTextNode(row.label));
    line.appendChild(k);
    line.appendChild(h('span', 'v', row.value));
    tip.appendChild(line);
  }
  tip.hidden = false;
  const pad = 14;
  const rect = tip.getBoundingClientRect();
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + rect.width > window.innerWidth - 8) x = evt.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = evt.clientY - rect.height - pad;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}
function hideTip() { if (tipNode) tipNode.hidden = true; }

/* ------------------------------------------------------------------- legend */
function legend(names, colors, kind = 'swatch') {
  const box = h('div', 'legend');
  names.forEach((name, i) => {
    const item = h('div', 'legend-item');
    const sw = h('span', kind === 'line' ? 'legend-swatch line' : 'legend-swatch');
    sw.style.background = colors[i];
    item.appendChild(sw);
    item.appendChild(document.createTextNode(name));
    box.appendChild(item);
  });
  return box;
}

/** The table-view twin every chart ships with (WCAG-clean, never gated). */
function tableView(headers, rows, caption) {
  const details = h('details', 'chart-table');
  details.appendChild(h('summary', null, caption || 'Table view'));
  const wrap = h('div', 'table-wrap');
  wrap.style.marginTop = '10px';
  const table = h('table');
  const thead = h('thead');
  const trh = h('tr');
  headers.forEach((head, i) => {
    const th = h('th', i === 0 ? '' : 'num', head);
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  table.appendChild(thead);
  const tbody = h('tbody');
  rows.forEach((row) => {
    const tr = h('tr');
    row.forEach((cell, i) => tr.appendChild(h('td', i === 0 ? '' : 'num', cell)));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  details.appendChild(wrap);
  return details;
}

/* =========================================================== grouped columns */
/**
 * @param {object} o
 * @param {string[]} o.categories   x-axis groups (e.g. held-out runs)
 * @param {{name:string, values:number[]}[]} o.series
 * @param {string} [o.caption]      figcaption under the chart
 * @param {(v:number)=>string} [o.format]
 * @param {number} [o.max]          y max (defaults to nice-ticked data max)
 */
function groupedColumns(o) {
  const fmt = o.format || ((v) => v.toFixed(3));
  const colors = o.series.map((_, i) => seriesColor(i));
  const fig = h('figure', 'chart-figure');

  if (o.series.length >= 2) fig.appendChild(legend(o.series.map((s) => s.name), colors));

  const W = 720, H = 260;
  const M = { top: 12, right: 12, bottom: 34, left: 44 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const dataMax = Math.max(0, ...o.series.flatMap((s) => s.values.filter(Number.isFinite)));
  const ticks = niceTicks(o.max != null ? o.max : dataMax || 1);
  const yMax = ticks[ticks.length - 1] || 1;
  const yOf = (v) => M.top + plotH - (v / yMax) * plotH;

  const scroll = h('div', 'chart-scroll');
  const chart = h('div', 'chart');
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
                          'aria-label': o.caption || 'Grouped column chart' });

  // gridlines + y ticks — solid hairlines, recessive
  ticks.forEach((t) => {
    svg.appendChild(el('line', {
      x1: M.left, x2: W - M.right, y1: yOf(t), y2: yOf(t),
      stroke: cssVar('--gridline'), 'stroke-width': 1, 'shape-rendering': 'crispEdges',
    }));
    svg.appendChild(el('text', {
      x: M.left - 8, y: yOf(t) + 4, 'text-anchor': 'end',
      fill: cssVar('--text-muted'), 'font-size': 11,
      style: 'font-variant-numeric: tabular-nums',
    }, String(t)));
  });
  svg.appendChild(el('line', {
    x1: M.left, x2: W - M.right, y1: yOf(0), y2: yOf(0),
    stroke: cssVar('--baseline'), 'stroke-width': 1, 'shape-rendering': 'crispEdges',
  }));

  const bandW = plotW / o.categories.length;
  const nSeries = o.series.length;
  const gap = 2;                                   // the surface gap, not a stroke
  const barW = Math.min(24, (bandW * 0.62 - gap * (nSeries - 1)) / nSeries);
  const groupW = barW * nSeries + gap * (nSeries - 1);

  o.categories.forEach((cat, ci) => {
    const cx = M.left + bandW * ci + bandW / 2;
    o.series.forEach((s, si) => {
      const v = s.values[ci];
      if (!Number.isFinite(v)) return;
      const x = cx - groupW / 2 + si * (barW + gap);
      const y = yOf(v);
      const hgt = yOf(0) - y;
      const path = el('path', {
        d: columnPath(x, y, barW, hgt, 4), fill: colors[si],
        tabindex: 0, role: 'graphics-symbol',
        'aria-label': `${s.name}, ${cat}: ${fmt(v)}`,
      });
      const rows = [{ label: s.name, value: fmt(v), color: colors[si] }];
      const enter = (e) => showTip(e, cat, rows);
      path.addEventListener('mousemove', enter);
      path.addEventListener('mouseenter', enter);
      path.addEventListener('mouseleave', hideTip);
      path.addEventListener('focus', (e) => {
        const r = path.getBoundingClientRect();
        showTip({ clientX: r.left + r.width / 2, clientY: r.top }, cat, rows);
      });
      path.addEventListener('blur', hideTip);
      svg.appendChild(path);
    });
    svg.appendChild(el('text', {
      x: cx, y: H - 12, 'text-anchor': 'middle',
      fill: cssVar('--text-secondary'), 'font-size': 11.5,
    }, cat));
  });

  chart.appendChild(svg);
  scroll.appendChild(chart);
  fig.appendChild(scroll);
  if (o.caption) fig.appendChild(h('figcaption', null, o.caption));
  fig.appendChild(tableView(
    ['', ...o.categories],
    o.series.map((s) => [s.name, ...s.values.map((v) => (Number.isFinite(v) ? fmt(v) : '—'))]),
  ));
  return fig;
}

/* ==================================================================== lines */
/**
 * @param {object} o
 * @param {number[]} o.x
 * @param {{name:string, values:number[]}[]} o.series
 * @param {string} [o.xLabel] @param {string} [o.caption]
 */
function lineChart(o) {
  const fmt = o.format || ((v) => v.toFixed(4));
  const colors = o.series.map((_, i) => seriesColor(i));
  const fig = h('figure', 'chart-figure');
  if (o.series.length >= 2) fig.appendChild(legend(o.series.map((s) => s.name), colors, 'line'));

  const W = 720, H = 240;
  const M = { top: 12, right: 54, bottom: 32, left: 48 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const all = o.series.flatMap((s) => s.values.filter(Number.isFinite));
  const ticks = niceTicks(Math.max(...all, 0) || 1);
  const yMax = ticks[ticks.length - 1] || 1;
  const xMax = Math.max(1, o.x.length - 1);
  const xOf = (i) => M.left + (i / xMax) * plotW;
  const yOf = (v) => M.top + plotH - (v / yMax) * plotH;

  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
                          'aria-label': o.caption || 'Line chart' });

  ticks.forEach((t) => {
    svg.appendChild(el('line', {
      x1: M.left, x2: W - M.right, y1: yOf(t), y2: yOf(t),
      stroke: cssVar('--gridline'), 'stroke-width': 1, 'shape-rendering': 'crispEdges',
    }));
    svg.appendChild(el('text', {
      x: M.left - 8, y: yOf(t) + 4, 'text-anchor': 'end',
      fill: cssVar('--text-muted'), 'font-size': 11,
      style: 'font-variant-numeric: tabular-nums',
    }, String(t)));
  });

  o.series.forEach((s, si) => {
    const pts = s.values.map((v, i) => (Number.isFinite(v) ? [xOf(i), yOf(v)] : null))
                        .filter(Boolean);
    if (!pts.length) return;
    svg.appendChild(el('path', {
      d: 'M' + pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' L'),
      fill: 'none', stroke: colors[si], 'stroke-width': 2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round',
    }));
    // End marker: >=8px with the 2px surface ring, plus the one direct label.
    const last = pts[pts.length - 1];
    svg.appendChild(el('circle', {
      cx: last[0], cy: last[1], r: 4.5, fill: colors[si],
      stroke: cssVar('--surface-1'), 'stroke-width': 2,
    }));
    svg.appendChild(el('text', {
      x: last[0] + 9, y: last[1] + 4, fill: cssVar('--text-secondary'), 'font-size': 11,
      style: 'font-variant-numeric: tabular-nums',
    }, fmt(s.values[s.values.length - 1])));
  });

  // x ticks: at most 6, always including the last epoch
  const stride = Math.max(1, Math.ceil(o.x.length / 6));
  o.x.forEach((xv, i) => {
    if (i % stride !== 0 && i !== o.x.length - 1) return;
    svg.appendChild(el('text', {
      x: xOf(i), y: H - 10, 'text-anchor': 'middle',
      fill: cssVar('--text-muted'), 'font-size': 11,
      style: 'font-variant-numeric: tabular-nums',
    }, String(xv)));
  });

  // Crosshair + tooltip over the whole plot (nearest x), not per-point targets.
  const crosshair = el('line', {
    y1: M.top, y2: M.top + plotH, stroke: cssVar('--baseline'),
    'stroke-width': 1, opacity: 0, 'shape-rendering': 'crispEdges',
  });
  svg.appendChild(crosshair);
  const hit = el('rect', { x: M.left, y: M.top, width: plotW, height: plotH,
                           fill: 'transparent', style: 'cursor: crosshair' });
  hit.addEventListener('mousemove', (e) => {
    const box = svg.getBoundingClientRect();
    const rel = ((e.clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(o.x.length - 1,
      Math.round(((rel - M.left) / plotW) * xMax)));
    crosshair.setAttribute('x1', xOf(i));
    crosshair.setAttribute('x2', xOf(i));
    crosshair.setAttribute('opacity', 1);
    showTip(e, `${o.xLabel || 'x'} ${o.x[i]}`, o.series.map((s, si) => ({
      label: s.name,
      value: Number.isFinite(s.values[i]) ? fmt(s.values[i]) : '—',
      color: colors[si],
    })));
  });
  hit.addEventListener('mouseleave', () => { crosshair.setAttribute('opacity', 0); hideTip(); });
  svg.appendChild(hit);

  const chart = h('div', 'chart');
  chart.appendChild(svg);
  fig.appendChild(chart);
  if (o.caption) fig.appendChild(h('figcaption', null, o.caption));
  fig.appendChild(tableView(
    [o.xLabel || 'x', ...o.series.map((s) => s.name)],
    o.x.map((xv, i) => [String(xv), ...o.series.map(
      (s) => (Number.isFinite(s.values[i]) ? fmt(s.values[i]) : '—'))]),
  ));
  return fig;
}

/* ========================================================== proportion bar */
/**
 * A single 100%-wide stacked bar. Segments are separated by a 2px surface gap,
 * never a stroke. Labels ride outside; the table view carries every number.
 * @param {{label:string, value:number, color:string}[]} parts
 */
function proportionBar(parts, opts = {}) {
  const total = parts.reduce((a, p) => a + p.value, 0) || 1;
  const fig = h('figure', 'chart-figure');
  const box = h('div');
  box.style.cssText = 'display:flex; height:14px; border-radius:7px; overflow:hidden; gap:2px;';
  parts.forEach((p) => {
    if (p.value <= 0) return;
    const seg = h('div');
    const pct = (p.value / total) * 100;
    seg.style.cssText = `flex: ${pct} 0 0; background:${p.color}; min-width:2px;`;
    seg.title = `${p.label}: ${p.value.toLocaleString()} (${pct.toFixed(1)}%)`;
    seg.addEventListener('mousemove', (e) => showTip(e, p.label, [
      { label: 'count', value: p.value.toLocaleString(), color: p.color },
      { label: 'share', value: `${pct.toFixed(1)}%` },
    ]));
    seg.addEventListener('mouseleave', hideTip);
    box.appendChild(seg);
  });
  fig.appendChild(box);

  const lg = h('div', 'legend');
  lg.style.marginTop = '10px';
  parts.forEach((p) => {
    const item = h('div', 'legend-item');
    const sw = h('span', 'legend-swatch');
    sw.style.background = p.color;
    item.appendChild(sw);
    item.appendChild(document.createTextNode(
      `${p.label} · ${p.value.toLocaleString()} (${((p.value / total) * 100).toFixed(1)}%)`));
    lg.appendChild(item);
  });
  fig.appendChild(lg);
  if (opts.caption) fig.appendChild(h('figcaption', null, opts.caption));
  return fig;
}

// showTip travels with hideTip: the live panel's own charts (views.js) reuse
// this tooltip rather than growing a second one that drifts out of step.
window.Charts = { groupedColumns, lineChart, proportionBar, seriesColor, cssVar,
                  tableView, showTip, hideTip };
