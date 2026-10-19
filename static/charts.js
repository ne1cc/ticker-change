/* Native financial chart lifecycle. Data stays inert until its panel is visible. */
(function () {
  'use strict';
  const libraries = new Map();
  const states = new Map();
  function points(x, y) {
    return x.map((v, i) => [v, typeof y[i] === 'number' && Number.isFinite(y[i]) ? y[i] : null]);
  }
  function surfaceData(grid) {
    return grid.dtes.flatMap((dte, row) => grid.moneyness.flatMap((money, col) => {
      const value = grid.iv_matrix[row]?.[col];
      return typeof value === 'number' && Number.isFinite(value) ? [[money, dte, value]] : [];
    }));
  }
  function line(name, x, y, color = '#f59e0b', extra = {}) {
    return { name, type: 'line', data: points(x, y), showSymbol: false, connectNulls: false,
      lineStyle: { color, width: 1.8 }, itemStyle: { color }, ...extra };
  }
  function bar(name, x, y, color = '#6366f1', extra = {}) {
    const data = points(x, y).map((value, i) => Array.isArray(color) ? { value, itemStyle: { color: color[i] } } : value);
    return { name, type: 'bar', data, barMaxWidth: 22, itemStyle: { color: Array.isArray(color) ? undefined : color }, ...extra };
  }
  function ref(axis, value, name, color = '#f59e0b') {
    return { [axis + 'Axis']: value, name, label: { formatter: name, position: 'insideEnd' }, lineStyle: { color, type: 'dashed' } };
  }
  function cartesian(series, xname = '', yname = '', xtype = 'value') {
    return { series, xAxis: { type: xtype, name: xname }, yAxis: { type: 'value', name: yname, scale: true } };
  }
  function load(url) {
    if (!libraries.has(url)) libraries.set(url, new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = url; script.onload = resolve; script.onerror = () => reject(new Error('Chart library could not be loaded.'));
      document.head.append(script);
    }));
    return libraries.get(url);
  }
  function palette() {
    const dark = document.documentElement.classList.contains('dark') && !document.documentElement.classList.contains('excel');
    return { text: dark ? '#d4d4d8' : '#3f3f46', grid: dark ? '#27272a' : '#e4e4e7', bg: dark ? '#18181b' : '#ffffff' };
  }
  function themed(option, state) {
    const p = palette();
    const axis = a => ({ ...a, nameLocation: 'middle', nameGap: a.nameGap ?? (a.position === 'top' ? 28 : 35),
      nameTextStyle: { color: p.text, fontSize: 10 }, axisLabel: { color: p.text, fontSize: 10, hideOverlap: true, ...a.axisLabel },
      axisLine: { lineStyle: { color: p.grid } }, splitLine: { lineStyle: { color: p.grid }, ...a.splitLine } });
    const result = { animation: false, backgroundColor: 'transparent',
      textStyle: { fontFamily: 'JetBrains Mono, ui-monospace, monospace', color: p.text },
      grid: { left: 18, right: 22, top: 58, bottom: 62, containLabel: true },
      tooltip: { trigger: 'axis', renderMode: 'richText', confine: true, backgroundColor: p.bg, textStyle: { color: p.text }, axisPointer: { type: 'cross' } },
      aria: { enabled: true }, dataZoom: [{ type: 'inside', filterMode: 'none' }, { type: 'slider', height: 15, bottom: 10, borderColor: p.grid }], ...option };
    result.legend = {type:'scroll',top:5,...option.legend,textStyle:{color:p.text,fontSize:10}};
    for (const key of ['xAxis', 'yAxis', 'xAxis3D', 'yAxis3D', 'zAxis3D']) {
      if (option[key]) result[key] = Array.isArray(option[key]) ? option[key].map(axis) : axis(option[key]);
    }
    if (option.title) result.title = { ...option.title, top: 28, textStyle: { color: p.text, fontSize: 11, fontWeight: 400 } };
    if (option.series?.some(s => s.type === 'treemap' || s.type === 'surface')) {
      delete result.dataZoom; delete result.xAxis; delete result.yAxis;
      result.tooltip.trigger = 'item';
    }
    if (option.terminalHorizontalBars) {
      result.series = option.series.map(series => {
        if (series.type !== 'bar') return series;
        const ys = series.data.map(p => (p.value || p)[1]).sort((a,b) => a-b);
        const gaps = ys.slice(1).map((v,i) => v-ys[i]).filter(v => v > 0);
        const step = gaps.length ? Math.min(...gaps) : 1;
        return { ...series, type: 'custom', encode: { x: 0, y: 1 }, renderItem(params, api) {
          const end = api.coord([api.value(0), api.value(1)]), start = api.coord([0, api.value(1)]);
          const height = Math.max(3, Math.min(24, Math.abs(api.size([0, step])[1]) * .75));
          const rawWidth = Math.abs(end[0] - start[0]);
          const width = rawWidth > 0 ? Math.max(rawWidth, 1.5) : 0;
          const shape = echarts.graphic.clipRectByRect({ x: Math.min(start[0],end[0]), y: end[1]-height/2, width, height }, params.coordSys);
          return shape && { type: 'rect', shape, style: api.style() };
        } };
      });
    }
    if (option.series?.some(s => s.type === 'treemap')) {
      result.series = option.series.map(series => ({...series,
        // Treemaps animate updates for 900ms by default, smearing old labels over a re-zoomed or re-coloured map.
        // Keep it non-zero: at 0 the re-root path stops drawing tiles.
        animationDurationUpdate:150,
        color:['#7a0c22','#d9534f',p.grid,'#3f9d63','#0b6e4f'],
        // White text once a tile is past ~30% of the scale (scale width varies by period).
        data:series.data.map(sector => ({...sector,children:sector.children.map(leaf => ({...leaf,label:{color:Math.abs(leaf.value[1])>=.3*series.visualMax?'#ffffff':p.text}}))})),
        label:{show:true,color:p.text,formatter:params => {
          if (!params.data?.leaf) return params.name;
          // Labels follow the laid-out tile size: blank when too small to read, name only when short or narrow.
          const box = state?.chart?.getModel().getSeriesByIndex(0).getData().getItemLayout(params.dataIndex);
          if (box?.width < 30 || box?.height < 16) return '';
          return box?.height < 30 || box?.width < 46 ? params.name : params.name + '\n' + params.data.change;
        }},
        upperLabel:{show:true,height:24,color:p.text,formatter:params => params.name + (params.data?.change ? '  ' + params.data.change : '')},
        breadcrumb:{...series.breadcrumb,itemStyle:{...series.breadcrumb?.itemStyle,color:p.grid,borderColor:p.bg,textStyle:{color:p.text}},emphasis:{itemStyle:{color:p.bg}}},
        itemStyle:{borderColor:p.bg},
        levels:series.levels.map(level => ({...level,itemStyle:{...level.itemStyle,borderColor:p.bg}}))
      }));
      delete result.legend;
      const dollars = v => v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : `$${(v / 1e6).toFixed(0)}M`;
      result.tooltip.formatter = params => params.data?.leaf
        ? `${params.data.company} (${params.data.symbol})\n${params.treePathInfo[1]?.name || ''}\n$${params.data.price.toFixed(2)} · ${params.data.change}\n20d avg $ volume ${dollars(params.data.value[0])}\n${state?.rootId ? 'Click to open price table' : 'Click to zoom into ' + (params.treePathInfo[1]?.name || 'sector')}`
        : params.name + '\n' + (params.data?.change || '') + (params.data?.children ? '\nClick to zoom in' : '');
    }
    if (option.series?.some(s => s.type === 'surface')) {
      result.grid3D = { ...option.grid3D, axisPointer: { lineStyle: { color: p.text } } };
    }
    return result;
  }
  function apply(state, reset = false) {
    const { element, option } = state;
    if (!element.clientWidth || !element.clientHeight) return;
    if (!state.chart) {
      state.chart = echarts.init(element);
      state.chart.on('click', event => {
        // Full map: a click anywhere in a sector zooms into it. Zoomed in: a ticker opens its price table.
        const sector = event.data?.id?.startsWith('sector::') ? event.data.id : (event.data?.leaf && !state.rootId ? 'sector::' + event.treePathInfo?.[1]?.name : null);
        if (sector) state.chart.dispatchAction({ type: 'treemapRootToNode', targetNodeId: sector });
        else if (event.data?.symbol && event.data?.leaf) location.href = '/stock?ticker=' + encodeURIComponent(event.data.symbol);
      });
      state.chart.on('treemapRootToNode', event => state.onTreemapNavigate?.(event));
    }
    if (reset) state.rootId = '';
    const existing = !reset ? state.chart.getOption() : null;
    const styled = themed(option, state);
    if (existing?.dataZoom) styled.dataZoom = existing.dataZoom;
    if (existing?.grid3D?.[0]?.viewControl && styled.grid3D) styled.grid3D.viewControl = existing.grid3D[0].viewControl;
    if (existing?.legend?.[0]?.selected) styled.legend = { ...styled.legend, selected: existing.legend[0].selected };
    state.chart.setOption(styled, { notMerge: true });
    state.chart.resize();
    // notMerge drops the treemap zoom; restore it so theme/period changes keep the sector in view.
    if (state.rootId) state.chart.dispatchAction({ type: 'treemapRootToNode', targetNodeId: state.rootId });
    state.syncControls?.();
  }
  async function initialize(state) {
    if (state.loading || !state.element.clientWidth || !state.element.clientHeight) return;
    state.loading = true;
    try {
      await load('https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.min.js');
      if (state.option.series?.some(s => s.type === 'surface')) {
        const canvas = document.createElement('canvas');
        if (!canvas.getContext('webgl2') && !canvas.getContext('webgl')) throw new Error('3D chart requires WebGL. Other charts remain available.');
        await load('https://cdn.jsdelivr.net/npm/echarts-gl@2.0.9/dist/echarts-gl.min.js');
      }
      apply(state);
    } catch (error) { state.element.textContent = error.message; }
    finally { state.loading = false; }
  }
  let visibleObserver;
  function mount(element, option) {
    if (typeof element === 'string') element = document.getElementById(element);
    if (!element) return;
    let state = states.get(element);
    if (state) {
      state.option = option;
      if (state.chart) apply(state); else if (state.visible) initialize(state);
      return;
    }
    state = { element, option, chart: null, loading: false }; states.set(element, state);
    element.classList.add('terminal-chart');
    if (!element.style.height) element.style.height = element.clientHeight ? (element.clientHeight + 'px') : '380px';
    element.setAttribute('role','img'); element.setAttribute('aria-label',option.title?.text || 'Interactive financial chart');
    const controls = document.createElement('div'); controls.className = 'chart-controls';
    if (option.terminalViews) {
      const select = document.createElement('select'); select.setAttribute('aria-label', 'Gamma exposure view');
      option.terminalViews.labels.forEach((label,i) => select.add(new Option(label,i)));
      select.onchange = () => { state.option = option.terminalViews.options[Number(select.value)]; if(state.chart) apply(state,true); };
      controls.append(select);
    }
    const treemap = option.series?.find(s => s.type === 'treemap');
    if (treemap) {
      // Keyboard-reachable twin of clicking a sector header; kept in sync with clicks and the breadcrumb.
      const select = document.createElement('select'); select.setAttribute('aria-label', 'Zoom to sector');
      select.add(new Option('All sectors', ''));
      treemap.data.forEach(sector => select.add(new Option(sector.name, sector.id)));
      select.onchange = () => {
        if (!state.chart) return;
        if (select.value) state.chart.dispatchAction({ type: 'treemapRootToNode', targetNodeId: select.value });
        else apply(state, true);
      };
      state.onTreemapNavigate = event => {
        const id = event.targetNode?.getId?.() ?? event.targetNodeId;
        state.rootId = [...select.options].some(o => o.value === id) ? id : '';
        select.value = state.rootId;
      };
      state.syncControls = () => { select.value = state.rootId || ''; };
      controls.append(select);
    }
    if (option.terminalPeriods) {
      const group = document.createElement('div'); group.setAttribute('role', 'group'); group.setAttribute('aria-label', 'Colour by period');
      group.style.display = 'contents';
      const buttons = option.terminalPeriods.map((period, i) => {
        const button = document.createElement('button'); button.type = 'button'; button.textContent = period.key;
        button.title = 'Colour by ' + period.label + ' change (scale ±' + period.span + '%)';
        button.setAttribute('aria-pressed', String(i === 0));
        button.onclick = () => {
          buttons.forEach(b => b.setAttribute('aria-pressed', String(b === button)));
          const base = option.series[0];
          state.option = { ...option, series: [{ ...base, data: period.data, visualMin: -period.span, visualMax: period.span }] };
          if (state.chart) apply(state);
          element.dispatchEvent(new CustomEvent('terminal-period', { detail: { key: period.key, span: period.span } }));
        };
        group.append(button);
        return button;
      });
      controls.append(group);
    }
    if (option.terminalRanges) {
      option.terminalRanges.forEach(range => {
        const button = document.createElement('button'); button.type = 'button'; button.textContent = range;
        button.onclick = () => {
          if (!state.chart) return;
          if (range === 'All') { state.chart.dispatchAction({type:'dataZoom',start:0,end:100}); return; }
          const dates = option.series.flatMap(s => s.data.map(p => new Date((p.value || p)[0]).getTime())).filter(Number.isFinite);
          const end = Math.max(...dates), start = new Date(end), day = start.getUTCDate();
          start.setUTCDate(1); start.setUTCMonth(start.getUTCMonth() - (range.endsWith('Y') ? parseInt(range,10)*12 : parseInt(range,10)));
          const last = new Date(Date.UTC(start.getUTCFullYear(),start.getUTCMonth()+1,0)).getUTCDate();
          start.setUTCDate(Math.min(day,last));
          state.chart.dispatchAction({type:'dataZoom',startValue:start.getTime(),endValue:end});
        };
        controls.append(button);
      });
    }
    for (const [label, action] of [['Reset', () => { if (state.chart) apply(state,true); }], ['Export', () => {
      if (!state.chart) return;
      const a = document.createElement('a'); a.download = 'chart.png'; a.href = state.chart.getDataURL({ pixelRatio: 2, backgroundColor: palette().bg }); a.click();
    }]]) {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = label; button.onclick = action; controls.append(button);
    }
    element.before(controls);
    const resize = new ResizeObserver(() => {
      if (state.chart) state.chart.resize();
      else if (state.visible && element.clientWidth && element.clientHeight) initialize(state);
    });
    resize.observe(element);
    visibleObserver.observe(element);
  }
  window.TerminalCharts = { mount, points, surfaceData, line, bar, ref, cartesian,
    resize(id) { states.get(document.getElementById(id))?.chart?.resize(); } };
  if (typeof document === 'undefined') return;
  visibleObserver = new IntersectionObserver(entries => entries.forEach(entry => {
    const state = states.get(entry.target); state.visible = entry.isIntersecting;
    if (entry.isIntersecting) initialize(state);
  }), { rootMargin: '100px' });
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('script[data-chart]').forEach(script => mount(script.dataset.chart, JSON.parse(script.textContent)));
    new MutationObserver(() => states.forEach(state => { if (state.chart) apply(state); })).observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  });
})();
