/* The extra views: an overhead map of the room, and the confidence histogram.
 *
 * The trend plots are not here — those are Charts.lineChart from the dashboard,
 * reused as-is. What lives in this file is the two things it has no equivalent
 * of: a spatial map, and a distribution with a live threshold on it.
 *
 * Encoding, decided once and applied in both:
 *
 *   terrain / context  neutral sequential ramp, --surface-3 → --text-secondary.
 *                      Those two tokens swap ends between light and dark mode,
 *                      so the ramp re-anchors with the theme for free, and
 *                      being neutral it never competes with the thing you are
 *                      actually looking for.
 *   rock               --series-2 (orange), the ONE chromatic hue on the map,
 *                      and the same hue on the histogram's above-threshold
 *                      bars. Orange means "the model calls this a rock" in both
 *                      places, which is what makes the two charts one view.
 *   sensor             --series-1 (blue). Distinct from rock at a glance and
 *                      under CVD; it is a different kind of thing, not a
 *                      different amount of one.
 *
 * Probability is deliberately NOT a second color ramp on the map. At a rock's
 * on-screen size a lightness ramp is unreadable, and spending the channel there
 * would leave the histogram — where the distribution is legible and actionable
 * — with nothing to say.
 */
'use strict';

const V = (name) => getComputedStyle(document.documentElement)
  .getPropertyValue(name).trim();

/* --------------------------------------------------------------- color ---- */
function parseHex(hex) {
  const s = hex.replace('#', '').trim();
  const full = s.length === 3 ? s.split('').map((c) => c + c).join('') : s;
  const n = parseInt(full, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function lerpRgb(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/* ============================================================== OVERHEAD === */
/**
 * Draw the room from above: fused terrain, detections, sensor, scoring region.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {object} scene  the /api/scene payload
 * @returns {{toWorld: Function, marks: Array}} hit-testing handles for hover
 */
function overhead(canvas, scene) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 480;
  const cssH = canvas.clientHeight || 360;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const bev = scene.bev;
  const dets = (scene.detections && scene.detections.rows) || [];
  const sensor = scene.sensor || { x: 0, y: 0, yaw_deg: 0 };

  // World bounds: the terrain, plus anything drawn on top of it. A detection
  // just outside the fused area must not be silently cropped away.
  let minX = sensor.x - 1, maxX = sensor.x + 1;
  let minY = sensor.y - 1, maxY = sensor.y + 1;
  if (bev) {
    minX = Math.min(minX, bev.x0); maxX = Math.max(maxX, bev.x0 + bev.w * bev.cell);
    minY = Math.min(minY, bev.y0); maxY = Math.max(maxY, bev.y0 + bev.h * bev.cell);
  }
  for (const d of dets) {
    minX = Math.min(minX, d[0]); maxX = Math.max(maxX, d[0]);
    minY = Math.min(minY, d[1]); maxY = Math.max(maxY, d[1]);
  }
  const pad = 0.4;
  minX -= pad; maxX += pad; minY -= pad; maxY += pad;

  // One scale for both axes: a map with different x and y scales is a lie about
  // the shape of the room.
  const scale = Math.min(cssW / (maxX - minX), cssH / (maxY - minY));
  const offX = (cssW - (maxX - minX) * scale) / 2;
  const offY = (cssH - (maxY - minY) * scale) / 2;
  // World +y is up; canvas +y is down.
  const sx = (x) => offX + (x - minX) * scale;
  const sy = (y) => cssH - offY - (y - minY) * scale;

  // -- terrain ------------------------------------------------------------- //
  if (bev) {
    const low = parseHex(V('--surface-3'));
    const high = parseHex(V('--text-secondary'));
    const bytes = atob(bev.data);
    const img = ctx.createImageData(bev.w, bev.h);
    for (let j = 0; j < bev.h; j++) {
      // Row 0 of the raster is the low-y edge, i.e. the BOTTOM of the picture.
      const src = (bev.h - 1 - j) * bev.w;
      for (let i = 0; i < bev.w; i++) {
        const level = bytes.charCodeAt(src + i);
        const o = (j * bev.w + i) * 4;
        if (level === 0) { img.data[o + 3] = 0; continue; }  // never measured
        const [r, g, b] = lerpRgb(low, high, (level - 1) / 254);
        img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b;
        img.data[o + 3] = 255;
      }
    }
    const off = document.createElement('canvas');
    off.width = bev.w; off.height = bev.h;
    off.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, sx(bev.x0), sy(bev.y0 + bev.h * bev.cell),
                  bev.w * bev.cell * scale, bev.h * bev.cell * scale);
  }

  // -- scoring region ------------------------------------------------------ //
  const region = scene.region;
  if (region && region.range_max > 0) {
    ctx.save();
    ctx.strokeStyle = V('--baseline');
    ctx.setLineDash([]);            // solid: a dashed ring reads as "projected"
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(sx(sensor.x), sy(sensor.y), region.range_max * scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // -- detections ---------------------------------------------------------- //
  // Sized in world units so the marks mean "this much ground", with a floor so
  // they stay visible zoomed out. One hue: presence is the message.
  const rock = V('--series-2');
  const markPx = Math.max(3, Math.min(9, 0.12 * scale));
  ctx.fillStyle = rock;
  for (const d of dets) {
    ctx.beginPath();
    ctx.arc(sx(d[0]), sy(d[1]), markPx / 2, 0, Math.PI * 2);
    ctx.fill();
  }

  // -- sensor -------------------------------------------------------------- //
  const px = sx(sensor.x);
  const py = sy(sensor.y);
  const yaw = (sensor.yaw_deg || 0) * Math.PI / 180;
  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(-yaw);               // canvas y is flipped, so the angle is too
  ctx.beginPath();                // a triangle: position AND which way it faces
  ctx.moveTo(9, 0); ctx.lineTo(-6, 6); ctx.lineTo(-6, -6); ctx.closePath();
  ctx.fillStyle = V('--series-1');
  ctx.strokeStyle = V('--surface-1');
  ctx.lineWidth = 2;              // 2px surface ring, not a border
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  // -- scale bar ----------------------------------------------------------- //
  const metres = niceMetres((maxX - minX) / 4);
  ctx.save();
  ctx.strokeStyle = V('--text-muted');
  ctx.fillStyle = V('--text-muted');
  ctx.lineWidth = 1;
  const barY = cssH - 14;
  ctx.beginPath();
  ctx.moveTo(12, barY); ctx.lineTo(12 + metres * scale, barY);
  ctx.stroke();
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillText(`${metres} m`, 12, barY - 5);
  ctx.restore();

  return {
    marks: dets.map((d) => ({ x: sx(d[0]), y: sy(d[1]), d })),
    scale,
  };
}

function niceMetres(target) {
  const steps = [0.5, 1, 2, 5, 10, 20];
  return steps.find((s) => s >= target) || 20;
}

/* ============================================================= HISTOGRAM === */
/**
 * Confidence distribution of every scored center, with the decision threshold
 * drawn on it. Bars at or above the threshold take the same orange the map
 * paints rocks with, so "how many marks am I about to see" is readable here.
 *
 * @param {object} hist  {counts, edges, threshold, total, above}
 * @param {boolean} log  log count axis — rocks are rare, and on a linear axis
 *                       the clear lobe flattens them into the baseline
 */
function histogram(hist, log) {
  // The bottom margin has to hold BOTH the tick row and the axis title, with
  // room under the last baseline for descenders — "rock probability" has two.
  // Sizing the box to the plot and letting the axis hang out is how a chart
  // ends up with its own little scrollbar, or a cropped label.
  const W = 560, H = 204;
  const M = { top: 12, right: 12, bottom: 46, left: 46 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;
  const NS = 'http://www.w3.org/2000/svg';
  const mk = (tag, attrs, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  };

  const counts = hist.counts;
  const maxCount = Math.max(1, ...counts);
  const val = (c) => (log ? Math.log10(c + 1) : c);
  const top = val(maxCount);
  const yOf = (c) => M.top + plotH - (val(c) / top) * plotH;
  const xOf = (p) => M.left + p * plotW;

  const svg = mk('svg', {
    viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': `Prediction confidence: ${hist.total} centers, `
                + `${hist.above} at or above the ${hist.threshold.toFixed(2)} threshold`,
  });

  // y grid — solid hairlines, one shade off the surface.
  const ticks = log
    ? [0, 1, 2, 3, 4, 5].filter((e) => e <= top).map((e) => 10 ** e - 1)
    : niceTicks(maxCount);
  for (const t of ticks) {
    svg.appendChild(mk('line', {
      x1: M.left, x2: W - M.right, y1: yOf(t), y2: yOf(t),
      stroke: V('--gridline'), 'stroke-width': 1, 'shape-rendering': 'crispEdges',
    }));
    svg.appendChild(mk('text', {
      x: M.left - 8, y: yOf(t) + 4, 'text-anchor': 'end',
      fill: V('--text-muted'), 'font-size': 11,
      style: 'font-variant-numeric: tabular-nums',
    }, compact(t)));
  }

  // bars — 2px surface gap, 4px rounded data-end anchored to the baseline
  const slot = plotW / counts.length;
  const barW = Math.max(1, slot - 2);
  const rock = V('--series-2');
  const quiet = V('--baseline');
  counts.forEach((c, i) => {
    const p0 = hist.edges[i];
    const p1 = hist.edges[i + 1];
    const above = p1 > hist.threshold;
    const y = yOf(c);
    const hgt = M.top + plotH - y;
    const bar = mk('path', {
      d: roundedTop(xOf(p0) + 1, y, barW, hgt, 4),
      fill: above ? rock : quiet,
    });
    const hit = mk('rect', {
      x: xOf(p0), y: M.top, width: slot, height: plotH,
      fill: 'transparent', style: 'cursor: pointer',
    });
    hit.addEventListener('mousemove', (e) => window.Charts.showTip(
      e, `${p0.toFixed(2)} – ${p1.toFixed(2)}`, [
        { label: 'centers', value: c.toLocaleString(), color: above ? rock : quiet },
        { label: 'verdict', value: above ? 'rock' : 'below threshold' },
      ]));
    hit.addEventListener('mouseleave', () => window.Charts.hideTip());
    svg.appendChild(bar);
    svg.appendChild(hit);
  });

  // threshold rule — solid, labelled; this is the number the slider moves
  const tx = xOf(hist.threshold);
  svg.appendChild(mk('line', {
    x1: tx, x2: tx, y1: M.top - 4, y2: M.top + plotH,
    stroke: V('--text-primary'), 'stroke-width': 2,
  }));
  const anchor = hist.threshold > 0.7 ? 'end' : 'start';
  svg.appendChild(mk('text', {
    x: tx + (anchor === 'end' ? -6 : 6), y: M.top + 6,
    'text-anchor': anchor, fill: V('--text-primary'), 'font-size': 11,
    style: 'font-variant-numeric: tabular-nums',
  }, `threshold ${hist.threshold.toFixed(2)}`));

  // x axis: tick row, then the title, both clear of the bottom edge
  for (const p of [0, 0.25, 0.5, 0.75, 1]) {
    svg.appendChild(mk('text', {
      x: xOf(p), y: H - 26, 'text-anchor': 'middle',
      fill: V('--text-muted'), 'font-size': 11,
      style: 'font-variant-numeric: tabular-nums',
    }, p.toFixed(2)));
  }
  svg.appendChild(mk('text', {
    x: M.left + plotW / 2, y: H - 7, 'text-anchor': 'middle',
    fill: V('--text-muted'), 'font-size': 11,
  }, `rock probability · ${log ? 'log' : 'linear'} counts`));

  return svg;
}

function roundedTop(x, y, w, h, r) {
  const rr = Math.min(r, w / 2, Math.max(0, h));
  const bottom = y + h;
  return `M${x},${bottom} L${x},${y + rr} Q${x},${y} ${x + rr},${y} `
       + `L${x + w - rr},${y} Q${x + w},${y} ${x + w},${y + rr} L${x + w},${bottom} Z`;
}

function niceTicks(max, count = 4) {
  if (max <= 0) return [0];
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;
  const out = [];
  for (let v = 0; v <= max + step * 0.001; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}

function compact(n) {
  if (n < 1000) return String(Math.round(n));
  if (n < 1e6) return `${(n / 1e3).toFixed(n < 1e4 ? 1 : 0)}k`;
  return `${(n / 1e6).toFixed(1)}M`;
}

window.Views = { overhead, histogram, niceTicks, compact };
