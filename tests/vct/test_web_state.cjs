// Offline frontend regression: actual inline script, controlled DOM/fetch/Plotly.
// Run with Node's built-in test runner; no npm dependencies or real HTTP/model.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const html = fs.readFileSync(path.join(__dirname, '../../vct/web/index.html'), 'utf8');
const historyHtml = fs.readFileSync(path.join(__dirname, '../../vct/web/history.html'), 'utf8');
const script = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].at(-1)[1]
  .replace(/\ninit\(\);\s*$/, '');
const tick = () => new Promise(resolve => setImmediate(resolve));
const data = (ds, n = 3) => ({dataset: ds, ds_key: ds, n_cells: n, genes: ['A', 'B'],
  cell_types: Array(n).fill(ds), umap: Array.from({length: n}, (_, i) => [i, i])});

test('encoder exploration is collapsed by default and clearly scoped', () => {
  const panel = html.match(/<details\b([^>]*id="encoderExploration"[^>]*)>([\s\S]*?)<\/details>/);
  assert.ok(panel, 'native details panel must exist');
  assert.doesNotMatch(panel[1], /\bopen\b/);
  assert.match(panel[2], /编码器敏感性探索/);
  assert.match(panel[2], /实验性/);
  assert.match(panel[2], /不代表真实细胞响应/);
  for (const id of ['goBtn', 'exprSlider', 'multiValue']) assert.ok(panel[2].includes(`id="${id}"`));
  assert.doesNotMatch(script, /细胞类型重判|细胞身份归属|身份重判改变/);
});

function harness() {
  const nodes = {}, pending = [], tabs = [], plots = [], timers = new Map();
  let nextTimer = 0;
  function element(id = '') {
    let html = '';
    const node = {id, value: '', textContent: '', style: {}, disabled: false, children: [],
      classList: {add() {}, remove() {}, contains() { return false; }, toggle() {}},
      appendChild(child) { this.children.push(child); if (id === 'geneSel' && !this.value) this.value = child.value; },
      insertBefore(child) { this.children.unshift(child); }, on() {}, remove() {},
      get options() { return this.children; },
      get innerHTML() { return html; },
      set innerHTML(value) { html = value; this.children = []; if (id === 'geneSel') this.value = ''; },
    };
    return node;
  }
  const get = id => nodes[id] ||= element(id);
  const ctx = vm.createContext({URLSearchParams, console, fixture: data('pbmc'),
    document: {getElementById: get, createElement: () => element(),
      querySelectorAll: selector => selector === '.multiBtn' ? [get('multi1'), get('multi2')] : []},
    history: {replaceState() {}}, location: {protocol: 'http:', search: ''},
    window: {addEventListener() {}},
    setTimeout(fn) { timers.set(++nextTimer, fn); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
    Plotly: {newPlot(box, traces, layout) { plots.push({box, traces, layout}); },
      react() { return Promise.resolve(); }, relayout() {}, Plots: {resize() {}}},
    fetch(url) { return new Promise((resolve, reject) => pending.push({url, reject,
      reply(body, status = 200) { resolve({ok: status < 400, status, json: async () => body}); }})); },
  });
  vm.runInContext(script, ctx);
  ctx.recordTab = (key, title, render) => { tabs.push({key, title, render}); render('box'); };
  vm.runInContext('openTab = recordTab; drawUmap = () => {}; DATA = fixture; fillGenes(DATA.genes);', ctx);
  const run = code => vm.runInContext(code, ctx);
  async function flushTimers() {
    const callbacks = [...timers.values()]; timers.clear();
    for (const callback of callbacks) callback();
    await tick();
  }
  return {run, ctx, pending, tabs, plots, get, flushTimers};
}

test('history opens without a dataset or inference and keeps a fixed source', () => {
  const h = harness();
  h.run('DATA = null; DS = "ms"; openHistory()');
  assert.equal(h.pending.length, 0);
  assert.equal(h.tabs[0].key, 'history');
  const frame = h.get('box').children.at(-1);
  assert.equal(frame.src, '/web/history.html');
  assert.equal(frame.title, '历史实验与方法迭代（只读）');
  h.run('DS = "ws"');
  h.tabs[0].render('again');
  assert.equal(h.get('again').children.at(-1).src, frame.src);
  assert.match(html, /id="historyBtn"/);
  assert.match(html, /id="historyFromEncoder"/);
});

function historyHarness(protocol = 'http:') {
  const h = harness();
  h.get('archiveState').dataset = {};
  const imageStates = ['ms_figure', 'gears_figure', 'cipher_figure'].map(id => h.get(id));
  const query = h.ctx.document.querySelectorAll;
  h.ctx.document.querySelectorAll = selector => selector === '[data-image] p' ? imageStates : query(selector);
  const source = historyHtml
    .match(/<script>([\s\S]*?)<\/script>/)[1].replace(/loadHistory\(\);\s*$/, '');
  const ctx = vm.createContext({document: h.ctx.document, fetch: h.ctx.fetch,
    location: {protocol}});
  vm.runInContext(source, ctx);
  return {...h, imageStates, run: code => vm.runInContext(code, ctx)};
}

test('history local-file mode explains missing images without requesting the API', async () => {
  const h = historyHarness('file:');
  for (let attempt = 0; attempt < 2; attempt++) {
    const done = h.run('loadHistory()');
    assert.equal(h.pending.length, 0, 'file mode must not fetch a local /api/history path');
    await done;
    assert.equal(h.get('fileHelp').hidden, false);
    assert.match(h.get('archiveState').textContent, /本地文件.*不代表图片丢失/);
    assert.doesNotMatch(h.get('archiveState').textContent, /重启|旧版/);
    assert.equal(h.get('archiveState').dataset.error, 'true');
    assert.equal(h.get('reload').disabled, true);
    for (const id of ['wsMetric', 'encoderMetric', 'gearsMetric', 'tier0Metric']) {
      assert.match(h.get(id).textContent, /暂无可核对指标/);
    }
    for (const state of h.imageStates) assert.match(state.textContent, /图片尚未核对.*服务入口/);
  }
});

test('history local-file help links to the default service and explains custom ports', () => {
  const notice = historyHtml.match(/<div\b([^>]*id="fileHelp"[^>]*)>([\s\S]*?)<\/div>/);
  assert.ok(notice);
  assert.match(notice[1], /\bhidden\b/);
  assert.match(notice[2], /href="http:\/\/127\.0\.0\.1:8377\/web\/history\.html"/);
  assert.match(notice[2], /默认端口/);
  assert.match(notice[2], /VCT_PORT/);
  assert.match(notice[2], /启动VirtualCellTool\.command/);
});

test('history service mode stays on the current HTTP or HTTPS origin', async () => {
  for (const protocol of ['http:', 'https:']) {
    const h = historyHarness(protocol);
    const done = h.run('loadHistory()');
    assert.equal(h.pending.length, 1);
    assert.equal(h.pending[0].url, '/api/history');
    h.pending[0].reply({}, 404);
    await done;
    assert.equal(h.get('fileHelp').hidden, true);
    assert.match(h.get('archiveState').textContent, /HTTP 404/);
    assert.equal(h.get('reload').disabled, false);
  }
});

test('history shows missing/invalid reports honestly while valid metrics remain visible', async () => {
  const h = historyHarness();
  const done = h.run('loadHistory()');
  assert.equal(h.pending[0].url, '/api/history');
  assert.equal(h.get('reload').disabled, true);
  h.pending[0].reply({read_only: true, dataset_independent: true, artifacts: {}, results: {
    ws_result: {status: 'available', values: {n_genes: 3016, ko_spearman: -0.0456, oe_spearman: 0.0719}},
    encoder_result: {status: 'missing'}, gears_result: {status: 'invalid'}, tier0_result: {status: 'unreadable'},
  }});
  await done;
  assert.match(h.get('wsMetric').textContent, /-0\.0456/);
  assert.match(h.get('encoderMetric').textContent, /材料缺失/);
  assert.match(h.get('gearsMetric').textContent, /校验失败/);
  assert.match(h.get('tier0Metric').textContent, /校验失败/);
  assert.equal(h.get('reload').disabled, false);
  const retry = h.run('loadHistory()');
  assert.doesNotMatch(h.get('wsMetric').textContent, /-0\.0456/);
  h.pending[1].reply({}, 404); await retry;
  assert.match(h.get('archiveState').textContent, /HTTP 404/);
  assert.equal(h.get('archiveState').dataset.error, 'true');
  assert.match(h.get('wsMetric').textContent, /暂无可核对指标/);
  assert.equal(h.get('reload').disabled, false);
});

test('history rejects a wrong endpoint identity and recovers from network failure', async () => {
  const h = historyHarness();
  const wrong = h.run('loadHistory()');
  h.pending[0].reply({read_only: false, dataset_independent: true}); await wrong;
  assert.match(h.get('archiveState').textContent, /身份不符/);
  const network = h.run('loadHistory()');
  h.pending[1].reject(new Error('offline')); await network;
  assert.match(h.get('archiveState').textContent, /offline/);
  assert.equal(h.get('reload').disabled, false);
});

test('slow old dataset does not replace the new dataset (including ABA)', async () => {
  const h = harness();
  const old = h.run('loadDataset("pbmc")');
  const middle = h.run('loadDataset("ms")');
  const recent = h.run('loadDataset("pbmc")');
  h.pending[2].reply(data('pbmc', 4)); await recent;
  h.pending[1].reply(data('ms', 2)); await middle;
  h.pending[0].reply(data('pbmc', 3)); await old;
  assert.equal(h.run('DATA.n_cells'), 4);
  assert.equal(h.run('DS'), 'pbmc');
});

test('stale dataset error cannot overwrite success; loading/failure blocks old-data actions', async () => {
  const h = harness();
  const old = h.run('loadDataset("pbmc")');
  const recent = h.run('loadDataset("ms")');
  assert.equal(h.run('DATA'), null);
  const before = h.pending.length;
  await h.run('selectCell(0)'); await h.run('doExprMap()');
  assert.equal(h.pending.length, before);
  h.pending[1].reply(data('ms')); await recent;
  const state = h.get('dsState').innerHTML;
  h.pending[0].reject(new Error('old failure')); await old;
  assert.equal(h.get('dsState').innerHTML, state);
  const fail = h.run('loadDataset("ws")');
  h.pending.at(-1).reject(new Error('new failure')); await fail;
  assert.equal(h.run('DATA'), null);
  assert.equal(h.get('geneSel').disabled, true);
});

test('slow cell attribution cannot overwrite current selection', async () => {
  const h = harness();
  const old = h.run('selectCell(0)'); const recent = h.run('selectCell(1)');
  const response = gene => ({top_genes: [{gene, attribution: 1, expr: 2}], cell_type: 'T', elapsed: 0});
  h.pending[1].reply(response('B')); await recent;
  h.pending[0].reply(response('A')); await old;
  assert.equal(h.run('selectedCell'), 1);
  assert.equal(h.run('currentTop[0]'), 'B');
});

test('old gene coverage cannot enable a newer uncovered gene', async () => {
  const h = harness();
  const old = h.run('checkCipher("A")');
  h.get('geneSel').value = 'B'; const recent = h.run('checkCipher("B")');
  h.pending[0].reply({fitted: true}); await tick();
  h.pending.find(p => p.url.includes('cipher_status') && p !== h.pending[0]).reply({fitted: true}); await tick();
  h.pending.find(p => p.url.includes('gene=B')).reply({covered: false}); await recent;
  const oldCoverage = h.pending.find(p => p.url.includes('gene=A'));
  if (oldCoverage) oldCoverage.reply({covered: true, ctrl_expr: 2});
  await old;
  assert.equal(h.get('cipherKoBtn').disabled, true);
  assert.match(h.get('cipherHint').innerHTML, /B/);
});

test('old causal coverage and errors cannot change current gene state', async () => {
  const h = harness();
  const old = h.run('checkCausal("A")');
  h.get('geneSel').value = 'B'; const recent = h.run('checkCausal("B")');
  h.pending[1].reply({covered: false}); await recent;
  const hint = h.get('causalHint').innerHTML;
  h.pending[0].reply({covered: true}); await old;
  assert.equal(h.get('causalBtn').disabled, true);
  assert.equal(h.get('causalHint').innerHTML, hint);
});

test('old expression response cannot reset the current gene slider', async () => {
  const h = harness(); h.run('selectedCell = 0');
  const old = h.run('onGeneChange()');
  h.get('geneSel').value = 'B'; const recent = h.run('onGeneChange()');
  h.pending.find(p => p.url.includes('cell_gene_expr') && p.url.includes('gene=B')).reply({value: 22});
  await recent;
  h.pending.find(p => p.url.includes('cell_gene_expr') && p.url.includes('gene=A')).reply({value: 11});
  await old;
  assert.equal(Number(h.get('exprSlider').value), 22);
});

test('expression request from an old dataset is ignored instead of mixing coordinates', async () => {
  const h = harness(); const old = h.run('doExprMap()');
  const change = h.run('loadDataset("ms")');
  h.pending[1].reply(data('ms', 2)); await change;
  h.pending[0].reply({gene: 'A', dataset: 'pbmc', values: [1, 2, 3], max: 3,
    pct_expressing: 100, by_group: [{group: 'T', n_cells: 3, mean: 2, pct_expressing: 100}]});
  await old;
  assert.equal(h.plots.length, 0);
});

test('completed expression figure keeps its own snapshot and source label', async () => {
  const h = harness(); const work = h.run('doExprMap()');
  h.pending[0].reply({gene: 'A', dataset: 'pbmc', values: [1, 2, 3], max: 3,
    pct_expressing: 100, by_group: [{group: 'T', n_cells: 3, mean: 2, pct_expressing: 100}]});
  await work;
  assert.match(h.tabs[0].title, /pbmc/i);
  h.ctx.fixture = data('ms', 2); h.run('DS = "ms"; DATA = fixture');
  h.tabs[0].render('again');
  assert.equal(h.plots.at(-1).traces[0].x.length, 3);
  assert.equal(h.plots.at(-1).traces[0].marker.color.length, 3);
});

test('API displays FastAPI validation detail rather than only an HTTP number', async () => {
  const h = harness(); const work = h.run('api("attribute", {cell: -1})');
  h.pending[0].reply({detail: [{loc: ['query', 'cell'], msg: 'must be nonnegative'}]}, 422);
  await assert.rejects(work, /cell.*must be nonnegative/);
});

test('obsolete search result cannot replace a restored list', async () => {
  const h = harness(); const init = h.run('init()');
  h.pending[0].reply(data('pbmc')); await init;
  h.get('geneFilter').value = 'AB';
  h.get('geneFilter').oninput({target: h.get('geneFilter')}); await h.flushTimers();
  const search = h.pending.find(p => p.url.includes('search_genes'));
  h.get('geneFilter').value = '';
  h.get('geneFilter').oninput({target: h.get('geneFilter')});
  search.reply({matches: ['OLD']}); await tick();
  assert.equal(h.get('geneSel').options[0].value, 'A');
});

test('in-flight search cannot replace a manually selected gene', async () => {
  const h = harness(); const init = h.run('init()');
  h.pending[0].reply(data('pbmc')); await init;
  h.get('geneFilter').value = 'AB';
  h.get('geneFilter').oninput({target: h.get('geneFilter')}); await h.flushTimers();
  const search = h.pending.find(p => p.url.includes('search_genes'));
  h.get('geneSel').value = 'B'; await h.get('geneSel').onchange();
  search.reply({matches: ['OLD']}); await tick();
  assert.equal(h.get('geneSel').value, 'B');
});

for (const action of ['doCipher("ko")', 'doBaseline()', 'doAttrFigure()',
                      'doPerturb()', 'doMultiPerturb(2)', 'doCausal()']) {
  test(`stale success and failure cannot update a new dataset: ${action}`, async () => {
    for (const fails of [false, true]) {
      const h = harness();
      h.run('selectedCell = 0; currentTop = ["A", "B"]; $("exprSlider").value = 1');
      const old = h.run(action);
      assert.equal(h.pending.length, 1);
      const change = h.run('loadDataset("ms")');
      h.pending[1].reply(data('ms')); await change;
      const status = h.get('status').innerHTML;
      if (fails) h.pending[0].reject(new Error('obsolete failure'));
      else h.pending[0].reply({}); // Ignored before any result fields are read.
      await old;
      assert.equal(h.get('status').innerHTML, status);
      assert.equal(h.plots.length, 0);
      assert.equal(h.get('causalBtn').disabled, true);
      assert.equal(h.get('cipherKoBtn').disabled, true);
    }
  });
}

for (const action of ['doPerturb()', 'doMultiPerturb(2)']) {
  test(`newest movement wins and its plot keeps source coordinates: ${action}`, async () => {
    const h = harness();
    h.run('selectedCell = 0; currentTop = ["A", "B"]; $("exprSlider").value = 1');
    const old = h.run(action), recent = h.run(action);
    h.pending[1].reply({old_xy: [0, 0], new_xy: [1, 1], umap_moved: true,
      zoom: {x: [-2, 2], y: [-2, 2], span: 2}, old_value: 2, new_value: 1,
      displacement: .2, umap_shift: 1, elapsed: 0,
      old_dist: {T: 2}, new_dist: {T: 1}, changed: false, new_nearest: 'T'});
    await recent;
    assert.match(h.get('status').innerHTML, /✅/);
    assert.match(h.get('status').innerHTML, /不代表真实细胞响应/);
    if (action.includes('Multi')) {
      assert.match(h.get('status').innerHTML, /最近参考质心/);
      assert.match(h.plots[1].layout.title.text, /实验性编码空间质心距离/);
    }
    const count = h.plots.length;
    h.pending[0].reply({}); await old;
    assert.equal(h.plots.length, count);
    assert.match(h.tabs[0].title, /pbmc/);
    h.ctx.fixture = data('ms', 2); h.run('DS = "ms"; DATA = fixture');
    h.tabs[0].render('again');
    assert.equal(h.plots.at(-1).traces[0].x.length, 3);
  });
}

test('expression failure leaves the slider disabled and explains why', async () => {
  const h = harness(); h.run('selectedCell = 0');
  const work = h.run('onGeneChange()');
  h.pending[0].reject(new Error('expression unavailable')); await work;
  assert.equal(h.get('exprSlider').disabled, true);
  assert.equal(h.get('goBtn').disabled, true);
  assert.match(h.get('status').innerHTML, /expression unavailable/);
});
