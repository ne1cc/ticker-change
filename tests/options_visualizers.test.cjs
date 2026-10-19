const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const template = fs.readFileSync('templates/options.html', 'utf8');

function harness() {
  const nodes = new Map();
  function node() {
    return {value: '', textContent: '', innerText: '', innerHTML: '', className: '', children: [], title: 'Selected metric explanation',
      style: {}, attributes: {}, append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      setAttribute(key, value) { this.attributes[key] = value; },
      classList: {toggle() {}, contains() {return false;}}};
  }
  const document = {documentElement: {classList: {contains() {return false;}}}, getElementById(id) { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id); }, createElement: node};
  const context = vm.createContext({window: {}, console});
  vm.runInContext(fs.readFileSync('static/charts.js', 'utf8'), context);
  context.TerminalCharts = {...context.window.TerminalCharts, mount(id, option) { context.last = {id, option}; }};
  context.document = document;
  context.currentGreeksChart = 'delta';
  context.comparisonRange = 'near';
  context.optionsLoadStatus = 'ready';
  context.terminalData = {convention:'naive'};
  context.location = {search:''};
  context.optionsData = {spot_price: 705, days_to_expiration: 30, time_to_expiration_years: 30/365,
    selected_expiration:'2026-11-20', source:'Fixture', source_as_of:'2026-10-19 12:00:00', options: [680, 705, 730].map(strike => ({
    strike, call_delta: .5, put_delta: -.5, call_gamma: .02, put_gamma: .02,
    call_iv: 25, put_iv: 26, call_oi: 100, put_oi: 80, call_volume: 30, put_volume: 20,
    call_pct_change: 3, put_pct_change: -2, call_change: .3, put_change: -.2, call_last: 2, put_last: 3,
    call_oi_available:true, put_oi_available:true, call_change_available:true, put_change_available:true
  }))};
  const functions = ['bsD1D2', 'bsVanna', 'bsCharm', 'bsVomma', 'renderChartReferences',
    'switchGreeksChart', 'prepareComparisonView', 'setComparisonRange', 'comparisonConvention',
    'comparisonColors', 'renderComparisonSummary', 'renderGreeksChart', 'renderGexPlot', 'drawPayoffDiagram'];
  for (const name of functions) {
    const start = template.indexOf(`  function ${name}(`);
    if (start < 0) continue;
    const end = template.indexOf('\n  }', start) + 4;
    vm.runInContext(template.slice(start, end), context);
  }
  for (const [id, value] of Object.entries({'rf-rate-input': '.045', 'payoff-strike': '705',
    'payoff-premium': '10', 'payoff-qty': '1', 'payoff-type': 'call', 'payoff-position': 'long'})) {
    document.getElementById(id).value = value;
  }
  return context;
}
function text(node) { return [node.textContent, node.innerText, ...node.children.map(text)].join(' '); }
function pnlAt(option, price) { return option.series[0].data.find(p => p[0] === price)?.[1]; }

test('Greeks fills the selected strike range, including a single-strike chain', () => {
  const h = harness(); h.renderGreeksChart();
  let axis = h.last.option.xAxis;
  assert.ok(axis.min > 0 && axis.min <= 680 && axis.max >= 730);
  assert.ok(axis.max - axis.min <= 60);
  assert.equal(axis.scale, true);
  h.optionsData.options = [h.optionsData.options[1]];
  h.renderGreeksChart(); axis = h.last.option.xAxis;
  assert.ok(Number.isFinite(axis.min) && axis.min < 705 && axis.max > 705);
});

test('all nine Greek modes render and expose their active state and explanation', () => {
  const h = harness();
  for (const mode of ['delta', 'gamma', 'iv', 'gex', 'vanna', 'charm', 'vomma', 'voloi', 'pctchg']) {
    h.document.getElementById(`chart-tab-${mode}`).title = `${mode} explanation`;
    h.switchGreeksChart(mode);
    assert.ok(h.last.option.series.length);
    assert.equal(h.document.getElementById(`chart-tab-${mode}`).attributes['aria-pressed'], 'true');
    assert.equal(h.document.getElementById('greeks-description').textContent, mode === 'pctchg'
      ? 'Option price change, not GEX. Small starting prices can create large percentage moves.' : mode === 'gex'
      ? 'Estimated dealer gamma exposure by strike. Calls and puts are OI-based estimates; net is their signed sum.' : `${mode} explanation`);
    assert.equal(h.document.getElementById(`chart-tab-${mode === 'delta' ? 'gamma' : 'delta'}`).attributes['aria-pressed'], 'false');
    assert.ok(h.last.option.series[0].markLine.data.every(ref => ref.label.show === false));
    if (mode === 'voloi') assert.equal(h.last.option.yAxis.length, 2);
    if (mode === 'pctchg') assert.equal(Array.isArray(h.last.option.yAxis), false);
  }
});

test('GEX compares calls, puts, and net on one axis with explicit convention signs', () => {
  const h = harness(); h.optionsData.spot_price = 100;
  const make = (strike, oi=100) => ({strike,call_gamma:.01,put_gamma:.01,call_oi:oi,put_oi:100,
    call_oi_available:true,put_oi_available:true,call_pct_change:25,put_pct_change:25,
    call_change_available:true,put_change_available:true});
  h.optionsData.options = [make(100),make(110,200)];
  const naive = h.prepareComparisonView(h.optionsData,'gex','loaded','naive');
  assert.equal(naive.all[0].call, .01); assert.equal(naive.all[0].put, -.01); assert.equal(naive.all[0].net, 0);
  assert.equal(naive.all[1].call, .02); assert.equal(naive.all[1].put, -.01); assert.equal(naive.all[1].net, .01);
  const shortWings = h.prepareComparisonView(h.optionsData,'gex','loaded','short_wings');
  assert.equal(shortWings.all[1].call, -.02); assert.equal(shortWings.all[1].put, .01); assert.equal(shortWings.all[1].net, -.01);
  h.switchGreeksChart('gex');
  assert.equal(JSON.stringify(h.last.option.series.map(series=>series.name)), JSON.stringify(['Calls','Puts','Net GEX']));
  assert.equal(Array.isArray(h.last.option.yAxis), false);
  assert.equal(h.last.option.series[0].yAxisIndex || 0, 0);
  assert.equal(h.last.option.series[1].yAxisIndex || 0, 0);
  assert.equal(h.last.option.series[2].type, 'line');
  assert.ok(h.last.option.series.every(series=>!series.stack));
  assert.equal(h.last.option.series[0].itemStyle.color, '#2563eb');
  assert.equal(h.last.option.series[1].itemStyle.color, '#c2410c');
});

test('Daily change uses one scale and preserves fixed call/put colors across gain and loss', () => {
  const h = harness(); h.optionsData.options[0].call_pct_change = 25; h.optionsData.options[0].put_pct_change = 25;
  h.switchGreeksChart('pctchg');
  const [calls,puts] = h.last.option.series;
  assert.equal(h.last.option.yAxis.name, 'Option price change (%)');
  assert.equal(calls.data[0][1], puts.data[0][1]);
  assert.equal(calls.itemStyle.color, '#2563eb'); assert.equal(puts.itemStyle.color, '#c2410c');
  assert.match(h.document.getElementById('greeks-description').textContent,/not GEX/);
});

test('near-spot range selects closest 20, retains full data and discloses outside extreme', () => {
  const h = harness(); h.optionsData.spot_price = 100;
  h.optionsData.options = Array.from({length:30},(_,i)=>({strike:80+i,call_gamma:.01,put_gamma:.01,call_oi:100,put_oi:100,
    call_oi_available:true,put_oi_available:true,call_change_available:true,put_change_available:true,
    call_pct_change:i===0?1166.7:2,put_pct_change:0})).reverse();
  const near = h.prepareComparisonView(h.optionsData,'pctchg','near','naive');
  assert.equal(near.visible.length,20); assert.equal(near.excludedCount,10);
  assert.deepEqual(near.visible.map(point=>point.strike),near.visible.map(point=>point.strike).slice().sort((a,b)=>a-b));
  assert.equal(near.outsideMaxAbsChange,1166.7);
  const all = h.prepareComparisonView(h.optionsData,'pctchg','loaded','naive');
  assert.equal(all.visible.length,30); assert.ok(all.visible.some(point=>point.source.call_pct_change===1166.7));
});

test('daily tooltip retains the original extreme percent and dollar values', () => {
  const h = harness(); h.optionsData.options[0].call_pct_change = 1166.7; h.optionsData.options[0].call_change = .35;
  h.switchGreeksChart('pctchg');
  const tooltip = h.last.option.tooltip.formatter([{axisValue:h.optionsData.options[0].strike}]);
  assert.match(tooltip,/1166\.7%/); assert.match(tooltip,/\$0\.35/); assert.match(tooltip,/last \$2\.00/);
});

test('missing observations stay null while known zero OI and change remain valid', () => {
  const h = harness();
  const row = h.optionsData.options[0]; row.call_oi = 0; row.call_pct_change = 0; row.call_oi_available = true; row.call_change_available = true;
  row.put_oi_available = false; row.put_change_available = false;
  const gex = h.prepareComparisonView({spot_price:100,options:[{...row,strike:100,call_gamma:.01,put_gamma:.01}]},'gex','loaded','naive');
  assert.equal(gex.all[0].call,0); assert.equal(gex.all[0].put,null); assert.equal(gex.all[0].net,null); assert.equal(gex.totals.incompleteCount,1);
  const change = h.prepareComparisonView({spot_price:100,options:[{...row,strike:100}]},'pctchg','loaded','naive');
  assert.equal(change.all[0].call,0); assert.equal(change.all[0].put,null);
});

test('missing Greek observations remain gaps', () => {
  const h = harness(); h.optionsData.options[0].call_delta = 'N/A';
  h.optionsData.options[1].put_delta = null; h.renderGreeksChart();
  assert.equal(h.last.option.series[0].data[0][1], null);
  assert.equal(h.last.option.series[1].data[1][1], null);
});

test('reference key preserves distinct names at an identical price', () => {
  const h = harness();
  assert.equal(typeof h.renderChartReferences, 'function');
  h.renderChartReferences('greeks-references', [
    h.TerminalCharts.ref('x', 705, 'Current price'), h.TerminalCharts.ref('x', 705, 'Call wall')]);
  const key = text(h.document.getElementById('greeks-references'));
  assert.match(key, /Current price/); assert.match(key, /Call wall/); assert.match(key, /705\.00/);
});

for (const type of ['call', 'put']) for (const position of ['long', 'short']) {
  test(`${position} ${type} retains expiration P&L and risk statistics`, () => {
    const h = harness();
    h.document.getElementById('payoff-type').value = type;
    h.document.getElementById('payoff-position').value = position;
    h.drawPayoffDiagram();
    const direction = position === 'long' ? 1 : -1;
    const breakeven = type === 'call' ? 715 : 695;
    assert.equal(pnlAt(h.last.option, 705), -1000 * direction);
    assert.equal(pnlAt(h.last.option, breakeven), 0);
    assert.equal(h.document.getElementById('payoff-stat-breakeven').innerText, `$${breakeven.toFixed(2)}`);
    const profit = h.document.getElementById('payoff-stat-profit').innerText;
    const loss = h.document.getElementById('payoff-stat-loss').innerText;
    assert.equal(profit, position === 'short' ? '$1000' : type === 'call' ? 'Unlimited' : '$69500');
    assert.equal(loss, position === 'long' ? '-$1000' : type === 'call' ? 'Unlimited' : '-$69500');
    const first = pnlAt(h.last.option, 705);
    h.document.getElementById('payoff-qty').value = '2'; h.drawPayoffDiagram();
    assert.equal(pnlAt(h.last.option, 705), first * 2);
    assert.equal(h.document.getElementById('payoff-stat-breakeven').innerText, `$${breakeven.toFixed(2)}`);
    assert.equal(h.last.option.xAxis.scale, true);
    assert.ok(h.last.option.xAxis.min > 0);
  });
}

test('coincident spot and breakeven keep readable keys and hide rotated labels', () => {
  const h = harness(); h.document.getElementById('payoff-premium').value = '0'; h.drawPayoffDiagram();
  const refs = h.last.option.series[0].markLine.data;
  assert.ok(refs.filter(ref => ref.xAxis != null).every(ref => ref.label.show === false));
  assert.ok(refs.some(ref => ref.name === '$0 P&L'));
  const key = text(h.document.getElementById('payoff-references'));
  assert.match(key, /Current price/); assert.match(key, /Breakeven price/);
});

test('positive breakeven outside the old range is included; negative prices are excluded', () => {
  const h = harness(); h.document.getElementById('payoff-premium').value = '400'; h.drawPayoffDiagram();
  assert.ok(h.last.option.xAxis.max > 1105);
  assert.equal(pnlAt(h.last.option, 1105), 0);
  h.document.getElementById('payoff-type').value = 'put';
  h.document.getElementById('payoff-premium').value = '800'; h.drawPayoffDiagram();
  assert.ok(h.last.option.xAxis.min >= 0);
  assert.match(text(h.document.getElementById('payoff-references')), /outside/);
});

test('chain loading ignores a stale JSON response and supports failure, empty data, and retry', async () => {
  const h = harness();
  const start = template.indexOf('  async function loadOptionsGreeks(');
  const end = template.indexOf('\n  }', start) + 4;
  vm.runInContext(template.slice(start, end), h);
  Object.assign(h, {activeTab:'chain', ticker:'META', optionsRequestGeneration:0,
    requestedExpiration:null, greeksPollInterval:1, strikeCountParam:()=>'',
    renderOptionChainTable(){}, renderGreeksChart(){}, initPayoffSimulator(){}});
  const select = h.document.getElementById('expiration-select');
  select.value = '2026-11-20';
  let finishOld;
  h.fetch = async () => ({ok:true,json:()=>new Promise(resolve=>{finishOld=resolve;})});
  const old = h.loadOptionsGreeks();
  await new Promise(resolve=>setImmediate(resolve));
  select.value = '2026-12-18';
  const latest = {selected_expiration:select.value,options:[{strike:705}]};
  h.fetch = async () => ({ok:true,json:async()=>latest});
  await h.loadOptionsGreeks();
  finishOld({selected_expiration:'2026-11-20',options:[{strike:680}]});
  await old;
  assert.equal(h.optionsData,latest);
  h.fetch = async () => {throw new Error('offline');};
  await h.loadOptionsGreeks();
  assert.equal(h.optionsData,latest);
  assert.equal(h.optionsLoadStatus,'error');
  assert.match(h.document.getElementById('option-chain-body').innerHTML,/Retry/);
  h.fetch = async () => ({ok:true,json:async()=>({options:[]})});
  await h.loadOptionsGreeks();
  assert.equal(h.optionsLoadStatus,'empty');
  h.fetch = async () => ({ok:true,json:async()=>latest});
  await h.loadOptionsGreeks();
  assert.equal(h.optionsLoadStatus,'ready');
  assert.equal(h.optionsData,latest);
});

test('polling refreshes snapshot metadata without rebuilding an unchanged chain table', async () => {
  const h = harness();
  const start = template.indexOf('  async function pollOptionsGreeks(');
  const end = template.indexOf('\n  }', start) + 4;
  vm.runInContext(template.slice(start,end),h);
  let tableRenders=0, chartRenders=0;
  Object.assign(h,{activeTab:'chain',ticker:'META',optionsRequestGeneration:2,
    strikeCountParam:()=>'',renderOptionChainTable(){tableRenders++;},renderGreeksChart(){chartRenders++;}});
  h.document.getElementById('expiration-select').value=h.optionsData.selected_expiration;
  const fresh={...h.optionsData,source_as_of:'2026-10-19 12:01:00'};
  h.fetch=async()=>({ok:true,json:async()=>fresh});
  await h.pollOptionsGreeks();
  assert.equal(h.optionsData.source_as_of,fresh.source_as_of);
  assert.equal(tableRenders,0);
  assert.equal(chartRenders,1);
});

test('dealer GEX retains the taller chart and vertical navigation after merging main', () => {
  const h = harness();
  h.terminalData = {spot_price:100,gex:{strikes:[90,100,110],call_gex_series:[1,2,3],
    put_gex_series:[-3,-2,-1],net_gex_series:[-2,0,2],call_wall:110,put_wall:90,gamma_flip:100}};
  h.renderGexPlot();
  assert.equal(h.last.id,'gex-plot-container');
  assert.match(template, /id="gex-plot-container"[^>]*style="height: 720px;"/);
  assert.equal(h.last.option.xAxis.min,-h.last.option.xAxis.max);
  assert.equal(h.last.option.dataZoom[0].type,'inside');
  assert.equal(h.last.option.dataZoom[0].yAxisIndex,0);
  assert.equal(h.last.option.dataZoom[1].orient,'vertical');
  assert.equal(h.last.option.dataZoom[1].yAxisIndex,0);
  assert.equal(h.last.option.dataZoom[1].filterMode,'none');
});
