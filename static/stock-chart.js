/* Single candlestick/volume chart; daily bars stay daily at every zoom level. */
(function () {
  'use strict';
  function rangeStart(range, latest) {
    const d = new Date(latest + 'T00:00:00Z');
    if (range === 'All') return null;
    if (range === 'YTD') return d.getUTCFullYear() + '-01-01';
    const count = Number.parseInt(range, 10);
    if (range.endsWith('D')) d.setUTCDate(d.getUTCDate() - count);
    else {
      const day = d.getUTCDate(); d.setUTCDate(1);
      d.setUTCMonth(d.getUTCMonth() - (range.endsWith('Y') ? count * 12 : count));
      const lastDay = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 0)).getUTCDate();
      d.setUTCDate(Math.min(day, lastDay));
    }
    return d.toISOString().slice(0, 10);
  }
  window.StockChart = { rangeStart };
  if (typeof document === 'undefined') return;
  document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('stock-chart');
    const source = document.getElementById('stock-chart-data');
    if (!container || !source || !window.LightweightCharts) return;
    let records = JSON.parse(source.textContent), loading = false, exhausted = false, ready = false;
    if (!records.length) return;
    const ticker = source.dataset.ticker;
    const chart = LightweightCharts.createChart(container, { width: container.clientWidth, height: container.clientHeight,
      layout: { fontFamily: 'JetBrains Mono, ui-monospace, monospace', fontSize: 11 },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      rightPriceScale: { scaleMargins: { top: .08, bottom: .25 } }, timeScale: { timeVisible: false, rightOffset: 3 } });
    const candles = chart.addCandlestickSeries({ upColor: '#16a34a', downColor: '#dc2626', borderVisible: false, wickUpColor: '#16a34a', wickDownColor: '#dc2626' });
    const volumes = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: '' });
    volumes.priceScale().applyOptions({ scaleMargins: { top: .82, bottom: 0 } });
    function setData() {
      records = [...new Map(records.map(r => [r.time || r.date, { ...r, time: r.time || r.date }])).values()]
        .filter(r => ['open','high','low','close'].every(k => typeof r[k] === 'number' && Number.isFinite(r[k])))
        .sort((a,b) => a.time.localeCompare(b.time));
      candles.setData(records.map(({time,open,high,low,close}) => ({time,open,high,low,close})));
      volumes.setData(records.map(r => ({ time:r.time, value:r.volume || 0, color:r.close >= r.open ? 'rgba(22,163,74,.3)' : 'rgba(220,38,38,.3)' })));
    }
    function theme() {
      const dark = document.documentElement.classList.contains('dark') && !document.documentElement.classList.contains('excel');
      const grid = dark ? '#27272a' : '#e4e4e7';
      chart.applyOptions({ layout: { background: { type:'solid', color:dark ? '#18181b' : '#ffffff' }, textColor:dark ? '#d4d4d8' : '#3f3f46' },
        grid: { vertLines:{color:grid}, horzLines:{color:grid} }, rightPriceScale:{borderColor:grid}, timeScale:{borderColor:grid} });
    }
    async function history(start) {
      if (loading || exhausted) return;
      loading = true;
      const end = new Date(records[0].time + 'T00:00:00Z'); end.setUTCDate(end.getUTCDate() - 1);
      const params = new URLSearchParams({ start_date:start, end_date:end.toISOString().slice(0,10) });
      try {
        const response = await fetch('/api/chart-data/' + encodeURIComponent(ticker) + '?' + params);
        if (!response.ok) { if (response.status === 404) exhausted = true; return; }
        const result = await response.json();
        if (!result.data?.length) { exhausted=true; return; }
        const visible = chart.timeScale().getVisibleRange();
        const oldFirst = records[0].time;
        records = [...result.data, ...records]; setData();
        if (records[0].time === oldFirst) exhausted=true;
        if (visible) chart.timeScale().setVisibleRange(visible);
      } catch (_) {
        const message = document.getElementById('stock-chart-readout');
        if (message) message.textContent = 'Earlier prices could not be loaded. Pan again to retry.';
      } finally { loading=false; }
    }
    const buttons = document.querySelectorAll('[data-chart-range]');
    buttons.forEach(button => button.addEventListener('click', async () => {
      const range = button.dataset.chartRange;
      const start = rangeStart(range, records.at(-1).time);
      if (start && start < records[0].time) await history(start);
      if (start) chart.timeScale().setVisibleRange({ from:start, to:records.at(-1).time });
      else chart.timeScale().fitContent();
      buttons.forEach(b => b.setAttribute('aria-pressed',String(b === button)));
      const link = document.getElementById('tradingview-link');
      if (link) link.href = 'https://www.tradingview.com/chart/?' + new URLSearchParams({ symbol:ticker, interval:'D' });
    }));
    document.getElementById('stock-chart-reset')?.addEventListener('click', () => {
      chart.timeScale().fitContent(); buttons.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.chartRange === 'All')));
    });
    chart.subscribeCrosshairMove(event => {
      const r = event.seriesData.get(candles), v = event.seriesData.get(volumes);
      const readout = document.getElementById('stock-chart-readout');
      if (readout) readout.textContent = r ? `${event.time}  O ${r.open.toFixed(2)}  H ${r.high.toFixed(2)}  L ${r.low.toFixed(2)}  C ${r.close.toFixed(2)}  Vol ${(v?.value || 0).toLocaleString()}` : 'Daily prices · hover for OHLC and volume';
    });
    chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (!ready || !range || range.from >= -2 || loading || exhausted) return;
      const d = new Date(records[0].time + 'T00:00:00Z'); d.setUTCFullYear(d.getUTCFullYear()-1);
      history(d.toISOString().slice(0,10));
    });
    new ResizeObserver(() => chart.applyOptions({ width:container.clientWidth, height:container.clientHeight })).observe(container);
    new MutationObserver(theme).observe(document.documentElement,{attributes:true,attributeFilter:['class']});
    setData(); theme(); chart.timeScale().fitContent();
    requestAnimationFrame(() => { ready = true; });
  });
})();
