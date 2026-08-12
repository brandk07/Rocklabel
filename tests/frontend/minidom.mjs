/* A tiny DOM + HTML parser: enough to execute the dashboard's render path in
 * node and catch runtime errors / missing element ids. Not a browser. */
const VOID = new Set(['meta','link','br','hr','img','input','source']);

class ClassList {
  constructor(el) { this.el = el; }
  get _s() { return (this.el.className || '').split(/\s+/).filter(Boolean); }
  _set(a) { this.el.className = a.join(' '); }
  add(...c) { const s = this._s; c.forEach(x => !s.includes(x) && s.push(x)); this._set(s); }
  remove(...c) { this._set(this._s.filter(x => !c.includes(x))); }
  contains(c) { return this._s.includes(c); }
  toggle(c, force) {
    const on = force === undefined ? !this.contains(c) : !!force;
    on ? this.add(c) : this.remove(c); return on;
  }
}

export class El {
  constructor(tag) {
    this.tagName = (tag || 'div').toLowerCase();
    this.children = []; this.parentNode = null; this.attrs = {};
    this.dataset = {}; this.className = ''; this._text = '';
    this.hidden = false; this.style = { setProperty() {} };
    this.classList = new ClassList(this);
    this._listeners = {};
    this.scrollTop = 0; this.scrollHeight = 100; this.clientHeight = 50;
    this.value = ''; this.checked = false; this.disabled = false;
  }
  get id() { return this.attrs.id || ''; }
  set id(v) { this.attrs.id = v; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  removeChild(c) { this.children = this.children.filter(x => x !== c); }
  remove() { this.parentNode && this.parentNode.removeChild(this); }
  setAttribute(k, v) {
    this.attrs[k] = String(v);
    if (k === 'class') this.className = String(v);
    if (k.startsWith('data-')) this.dataset[k.slice(5).replace(/-(\w)/g, (_, c) => c.toUpperCase())] = String(v);
  }
  getAttribute(k) { return this.attrs[k] ?? null; }
  removeAttribute(k) { delete this.attrs[k]; if (k === 'class') this.className = ''; }
  hasAttribute(k) { return k in this.attrs; }
  focus() {}
  scrollIntoView() {}
  blur() {}
  click() { this.onclick && this.onclick({ preventDefault() {}, stopPropagation() {} }); }
  addEventListener(t, fn) { (this._listeners[t] ||= []).push(fn); }
  getBoundingClientRect() { return { left: 0, top: 0, width: 200, height: 40 }; }
  cloneNode() { const n = new El(this.tagName); n.className = this.className; n.attrs = {...this.attrs}; return n; }
  get firstElementChild() { return this.children.find(c => c.tagName !== '#text') || null; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text || this.children.map(c => c.textContent).join(''); }
  set innerHTML(v) { this._html = v; this.children = []; }
  get innerHTML() { return this._html || ''; }
  *walk() { yield this; for (const c of this.children) if (c.walk) yield* c.walk(); }
  matches(sel) {
    for (const part of sel.trim().split(/(?=[.#\[])/)) {
      if (!part) continue;
      if (part[0] === '#') { if (this.id !== part.slice(1)) return false; }
      else if (part[0] === '.') { if (!this.classList.contains(part.slice(1))) return false; }
      else if (part[0] === '[') {
        const m = part.match(/\[([\w-]+)(?:=["']?([^\]"']*)["']?)?\]/);
        if (!m) return false;
        const key = m[1].startsWith('data-')
          ? this.dataset[m[1].slice(5).replace(/-(\w)/g, (_, c) => c.toUpperCase())]
          : this.attrs[m[1]];
        if (key === undefined) return false;
        if (m[2] !== undefined && key !== m[2]) return false;
      } else if (this.tagName !== part.toLowerCase()) return false;
    }
    return true;
  }
  querySelectorAll(sel) {
    const steps = sel.trim().split(/\s+(?![^\[]*\])/);
    let pool = [...this.walk()].slice(1);
    for (let i = 0; i < steps.length; i++) {
      const hits = pool.filter(n => n.matches(steps[i]));
      if (i === steps.length - 1) return hits;
      pool = hits.flatMap(n => [...n.walk()].slice(1));
    }
    return [];
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

/* Regex HTML parser — the dashboard template is hand-written and well-formed. */
export function parse(html) {
  const root = new El('body');
  const stack = [root];
  const re = /<!--[\s\S]*?-->|<(\/?)([\w-]+)((?:\s+[\w:-]+(?:=(?:"[^"]*"|'[^']*'|[^\s>]+))?)*)\s*(\/?)>|([^<]+)/g;
  let m;
  while ((m = re.exec(html))) {
    const [full, close, tag, attrs, selfClose, text] = m;
    if (full.startsWith('<!--')) continue;
    if (text != null) { if (text.trim()) stack[stack.length - 1]._text ||= text.trim(); continue; }
    if (close) { if (stack.length > 1) stack.pop(); continue; }
    const el = new El(tag);
    for (const a of attrs.matchAll(/([\w:-]+)(?:=(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g)) {
      el.setAttribute(a[1], a[2] ?? a[3] ?? a[4] ?? '');
    }
    if (tag === 'template') el.content = new El('div');
    stack[stack.length - 1].appendChild(el);
    if (!selfClose && !VOID.has(tag.toLowerCase())) stack.push(el);
  }
  // Real browsers park a <template>'s children in .content, not as children.
  for (const n of root.walk()) {
    if (n.tagName === 'template') { n.content.children = n.children; n.children = []; }
  }
  return root;
}
