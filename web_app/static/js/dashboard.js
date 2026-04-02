/* dashboard.js - light minimalist dashboard wiring */

const UIW = {
  state: {
    models: [],
    primaryModelId: 'champion',
    h: 7,
    selectedDate: null,
    showBacktest: false,
    seriesVisible: {
      actual: true,
      champion: true,
      runner: true,
    },
    payload: null,
    visitorChart: null,
    weatherChart: null,
  }
};

function $(id) { return document.getElementById(id); }

function safeText(el, text) {
  if (!el) return;
  el.textContent = (text === null || text === undefined) ? '' : String(text);
}

function fmtInt(x) {
  if (x === null || x === undefined) return '--';
  const n = Number(x);
  if (!Number.isFinite(n)) return '--';
  return Math.round(n).toLocaleString('en-US');
}

function fmt1(x, unit = '') {
  if (x === null || x === undefined) return '--';
  const n = Number(x);
  if (!Number.isFinite(n)) return '--';
  return `${n.toFixed(1)}${unit}`;
}

function setWarning(msg) {
  const el = $('uiwWarning');
  if (!el) return;
  if (!msg) {
    el.style.display = 'none';
    el.textContent = '';
    return;
  }
  el.style.display = 'block';
  el.textContent = msg;
}

function setSpinner(on) {
  const el = $('uiwSpinner');
  if (!el) return;
  el.style.display = on ? 'flex' : 'none';
  el.setAttribute('aria-hidden', on ? 'false' : 'true');
}

async function apiGetJson(url) {
  // Must-have: spinner during any /api/* calls
  const useSpinner = url.startsWith('/api/');
  if (useSpinner) setSpinner(true);
  try {
    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
    return data;
  } finally {
    if (useSpinner) setSpinner(false);
  }
}

function setQuickViewActive(h) {
  for (const btnId of ['btnView1', 'btnView3', 'btnView7']) {
    const b = $(btnId);
    if (!b) continue;
    b.classList.toggle('uiw-chip--active', Number(b.dataset.h) === Number(h));
  }
}

function applyLightTheme() {
  // base.html sets data-bs-theme="dark" globally; switch to light only for this page.
  document.documentElement.setAttribute('data-bs-theme', 'light');
}

async function loadModels() {
  const data = await apiGetJson('/api/models');
  UIW.state.models = data.models || [];

  const hasRunner = UIW.state.models.some(m => m.model_id === 'runner_up');
  $('btnRunner').disabled = !hasRunner;
  return data;
}

function setPrimaryModel(modelId) {
  UIW.state.primaryModelId = modelId;
  $('btnChampion').classList.toggle('uiw-chip--active', modelId === 'champion');
  $('btnRunner').classList.toggle('uiw-chip--active', modelId === 'runner_up');
}

function initCharts() {
  const vEl = $('visitorChart');
  const wEl = $('weatherChart');
  if (!vEl || !wEl) return;

  if (!UIW.state.visitorChart) UIW.state.visitorChart = echarts.init(vEl);
  if (!UIW.state.weatherChart) UIW.state.weatherChart = echarts.init(wEl);

  window.addEventListener('resize', () => {
    UIW.state.visitorChart?.resize();
    UIW.state.weatherChart?.resize();
  });

  // Click point -> sync Weather Card + Thermometer
  UIW.state.visitorChart.off('click');
  UIW.state.visitorChart.on('click', (params) => {
    const date = params?.name;
    if (!date) return;
    UIW.state.selectedDate = date;
    syncSideCards();
  });
}

function buildHolidayMarkAreas(payload) {
  const x = payload.time_axis || [];
  const ranges = payload.holidays || [];
  if (!x.length || !ranges.length) return [];

  const start = x[0];
  const end = x[x.length - 1];

  const colorFor = (h) => {
    const name = (h.name || '').toLowerCase();
    const type = (h.type || 'festival').toLowerCase();
    if (name.includes('春节') || name.includes('spring')) return 'rgba(255,59,48,.12)';
    if (name.includes('国庆') || name.includes('national')) return 'rgba(255,149,0,.12)';
    if (type === 'vacation' && (name.includes('暑') || name.includes('summer'))) return 'rgba(10,132,255,.10)';
    if (type === 'vacation' && (name.includes('寒') || name.includes('winter'))) return 'rgba(88,86,214,.10)';
    return type === 'vacation' ? 'rgba(10,132,255,.08)' : 'rgba(255,149,0,.10)';
  };

  const within = (a, b, s, e) => !(b < s || a > e);

  return ranges
    .filter(h => within(h.start, h.end, start, end))
    .map(h => ([
      {
        xAxis: h.start,
        itemStyle: { color: colorFor(h) },
        label: {
          show: true,
          color: 'rgba(17,24,39,.65)',
          fontSize: 10,
          formatter: h.name || 'Holiday'
        }
      },
      { xAxis: h.end }
    ]));
}

function makeForecastMask(len, startIdx, values) {
  // return array length len: null before startIdx, then values aligned by index
  const out = new Array(len).fill(null);
  for (let i = 0; i < len; i++) {
    if (i >= startIdx) out[i] = values?.[i] ?? null;
  }
  return out;
}

function renderVisitorChart(payload) {
  const x = payload.time_axis || [];
  const thrCrowd = payload.thresholds?.crowd ?? 18500;

  const actual = payload.series?.actual || [];
  const champPred = payload.series?.champion_pred || [];
  const runPred = payload.series?.runner_pred || [];

  const forecastStart = payload.forecast?.start_index ?? Math.max(0, x.length - (payload.forecast?.h || UIW.state.h));
  const showBacktest = UIW.state.showBacktest;

  const champData = showBacktest ? champPred : makeForecastMask(x.length, forecastStart, champPred);
  const runData = showBacktest ? runPred : makeForecastMask(x.length, forecastStart, runPred);

  const holidayAreas = buildHolidayMarkAreas(payload);

  UIW.state.visitorChart.setOption({
    animationDuration: 450,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      valueFormatter: (v) => (v === null || v === undefined ? '--' : fmtInt(v))
    },
    legend: {
      data: ['Actual', 'Champion', 'Runner-up', 'Threshold'],
      top: 6,
      textStyle: { color: 'rgba(17,24,39,.72)' },
      selected: {
        'Actual': UIW.state.seriesVisible.actual,
        'Champion': UIW.state.seriesVisible.champion,
        'Runner-up': UIW.state.seriesVisible.runner,
        'Threshold': true,
      }
    },
    grid: { left: 52, right: 18, top: 46, bottom: 58 },
    xAxis: {
      type: 'category',
      data: x,
      boundaryGap: false,
      axisLabel: { color: 'rgba(17,24,39,.55)' },
      axisLine: { lineStyle: { color: 'rgba(17,24,39,.18)' } },
      axisPointer: { label: { backgroundColor: 'rgba(17,24,39,.70)' } }
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: 'rgba(17,24,39,.55)' },
      splitLine: { lineStyle: { color: 'rgba(17,24,39,.08)' } }
    },
    dataZoom: [
      {
        type: 'inside',
        xAxisIndex: 0,
        filterMode: 'none'
      },
      {
        type: 'slider',
        xAxisIndex: 0,
        height: 24,
        bottom: 18,
        borderColor: 'rgba(17,24,39,.12)',
        fillerColor: 'rgba(10,132,255,.10)',
        handleStyle: { color: 'rgba(10,132,255,.65)' },
        textStyle: { color: 'rgba(17,24,39,.45)' }
      }
    ],
    series: [
      {
        name: 'Actual',
        type: 'line',
        data: actual,
        smooth: true,
        symbolSize: 6,
        showSymbol: false,
        lineStyle: { width: 2, color: 'rgba(17,24,39,.65)' },
        emphasis: { focus: 'series' },
        markArea: holidayAreas.length ? { silent: true, data: holidayAreas } : undefined,
      },
      {
        name: 'Champion',
        type: 'line',
        data: champData,
        smooth: true,
        symbolSize: 6,
        showSymbol: false,
        lineStyle: { width: 3, color: '#0a84ff' },
        emphasis: { focus: 'series' },
        connectNulls: false,
      },
      {
        name: 'Runner-up',
        type: 'line',
        data: runData,
        smooth: true,
        symbolSize: 6,
        showSymbol: false,
        lineStyle: { width: 2, color: 'rgba(88,86,214,.95)' },
        emphasis: { focus: 'series' },
        connectNulls: false,
      },
      {
        name: 'Threshold',
        type: 'line',
        data: x.map(() => thrCrowd),
        symbol: 'none',
        lineStyle: { type: 'dotted', width: 2, color: 'rgba(255,59,48,.75)' },
        tooltip: { show: false }
      }
    ]
  }, { notMerge: true });
}

function renderWeatherChart(payload) {
  const x = payload.time_axis || [];
  const w = payload.weather || {};
  const precip = w.precip_mm || [];
  const tHi = w.temp_high_c || [];
  const tLo = w.temp_low_c || [];

  UIW.state.weatherChart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: {
      data: ['Precip (mm)', 'Temp High (°C)', 'Temp Low (°C)'],
      top: 6,
      textStyle: { color: 'rgba(17,24,39,.72)' }
    },
    grid: { left: 52, right: 18, top: 46, bottom: 38 },
    xAxis: {
      type: 'category',
      data: x,
      axisLabel: { color: 'rgba(17,24,39,.55)' },
      axisLine: { lineStyle: { color: 'rgba(17,24,39,.18)' } },
      boundaryGap: true
    },
    yAxis: [
      {
        type: 'value',
        name: 'mm',
        axisLabel: { color: 'rgba(17,24,39,.55)' },
        splitLine: { lineStyle: { color: 'rgba(17,24,39,.08)' } }
      },
      {
        type: 'value',
        name: '°C',
        axisLabel: { color: 'rgba(17,24,39,.55)' },
        splitLine: { show: false }
      }
    ],
    series: [
      { name: 'Precip (mm)', type: 'bar', data: precip, yAxisIndex: 0, itemStyle: { color: 'rgba(10,132,255,.22)' } },
      { name: 'Temp High (°C)', type: 'line', data: tHi, yAxisIndex: 1, smooth: true, lineStyle: { width: 2, color: 'rgba(255,149,0,.9)' } },
      { name: 'Temp Low (°C)', type: 'line', data: tLo, yAxisIndex: 1, smooth: true, lineStyle: { width: 2, color: 'rgba(52,199,89,.85)' } },
    ]
  }, { notMerge: true });
}

function getPrimaryModelMeta(payload) {
  const p = UIW.state.primaryModelId;
  return payload?.meta?.[p] || payload?.meta?.primary || {};
}

function getPrimarySeries(payload) {
  const p = UIW.state.primaryModelId;
  return {
    pred: p === 'runner_up' ? (payload?.series?.runner_pred || []) : (payload?.series?.champion_pred || []),
    risk_level: payload?.risk?.[p]?.risk_level || payload?.risk?.primary?.risk_level || [],
    drivers: payload?.risk?.[p]?.drivers || payload?.risk?.primary?.drivers || [],
    p_warn: payload?.risk?.[p]?.p_warn || payload?.risk?.primary?.p_warn || [],
  };
}

function renderKpis(payload) {
  const x = payload.time_axis || [];
  const { pred, risk_level, drivers } = getPrimarySeries(payload);

  // KPI is computed on latest view segment (h)
  const h = UIW.state.h;
  const startIdx = Math.max(0, x.length - h);
  let maxV = -Infinity, maxIdx = -1;
  for (let i = startIdx; i < pred.length; i++) {
    const v = Number(pred[i]);
    if (Number.isFinite(v) && v > maxV) { maxV = v; maxIdx = i; }
  }

  const viewRisk = risk_level.slice(startIdx);
  const warnDays = viewRisk.filter(v => (v || 0) > 0).length;
  const maxRisk = Math.max(...viewRisk.map(v => v || 0), 0);
  const maxRiskIdxView = viewRisk.findIndex(v => (v || 0) === maxRisk);
  const maxRiskIdx = maxRiskIdxView >= 0 ? (startIdx + maxRiskIdxView) : -1;

  safeText($('kpiMaxVisitor'), maxIdx >= 0 ? fmtInt(maxV) : '--');
  safeText($('kpiMaxVisitorDate'), maxIdx >= 0 ? `Date: ${x[maxIdx]}` : '--');
  safeText($('kpiWarnDays'), warnDays.toString());
  safeText($('kpiMaxRisk'), String(maxRisk));

  const d = (maxRiskIdx >= 0 ? (drivers[maxRiskIdx] || []) : []);
  safeText($('kpiMaxRiskDrivers'), d.length ? `Drivers: ${d.join(' · ')}` : 'Drivers: --');

  const meta = getPrimaryModelMeta(payload);
  safeText($('kpiModel'), meta?.model_name || UIW.state.primaryModelId);
  safeText($('kpiRunDir'), meta?.run_dir || '--');

  const metaLine = `Generated: ${payload.meta?.generated_at || '--'} · View: latest ${UIW.state.h}D · Primary: ${meta?.model_name || UIW.state.primaryModelId}`;
  safeText($('uiwMetaLine'), metaLine);
}

function renderTimeline(payload) {
  const el = $('riskTimeline');
  const x = payload.time_axis || [];
  const { risk_level, drivers } = getPrimarySeries(payload);

  el.innerHTML = '';
  for (let i = Math.max(0, x.length - UIW.state.h); i < x.length; i++) {
    const lv = risk_level[i] || 0;
    const day = document.createElement('div');
    day.className = 'uiw-day';

    const badgeClass = lv === 0 ? 'uiw-badge-ok' : (lv >= 3 ? 'uiw-badge-risk' : 'uiw-badge-warn');
    const badgeText = lv === 0 ? 'OK' : (lv >= 3 ? 'High' : 'Warn');

    day.innerHTML = `
      <div class="uiw-day-title">${x[i]}</div>
      <div class="uiw-badge ${badgeClass}">${badgeText} (L${lv})</div>
      <div class="uiw-muted" style="margin-top:6px; font-size:.86rem;">${(drivers[i]||[]).slice(0,3).join(' / ') || '—'}</div>
    `;

    day.addEventListener('click', () => {
      UIW.state.selectedDate = x[i];
      syncSideCards();
    });

    el.appendChild(day);
  }
}

function pickBestWindow(payload) {
  const x = payload.time_axis || [];
  const w = payload.weather || {};
  const precip = w.precip_mm || [];
  const tHi = w.temp_high_c || [];
  const tLo = w.temp_low_c || [];

  const { pred, risk_level, p_warn } = getPrimarySeries(payload);

  const h = UIW.state.h;
  const startIdx = Math.max(0, x.length - h);
  const endIdx = x.length - 1;

  // Simple heuristic scoring (lower is better)
  // - risk_level dominates
  // - precip and extreme temp are secondary
  let bestIdx = startIdx;
  let bestScore = Infinity;
  for (let i = startIdx; i <= endIdx; i++) {
    const lv = Number(risk_level[i] || 0);
    const pr = Number(precip[i] ?? 0);
    const hi = Number(tHi[i] ?? 0);
    const lo = Number(tLo[i] ?? 0);
    const tempPenalty = (hi >= 28 ? 1 : 0) + (lo <= -10 ? 1 : 0);
    const crowdPenalty = pred[i] ? Math.min(1.0, Number(pred[i]) / (payload.thresholds?.crowd || 18500)) : 0.5;
    const conf = Number(p_warn[i] ?? 0.2);

    const score = (lv * 3.0) + (pr * 0.15) + (tempPenalty * 1.0) + (crowdPenalty * 1.2) + (conf * 0.6);
    if (score < bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }

  // Optionally pick a 2-day window if we have >=3 days
  const windowLen = h >= 3 ? 2 : 1;
  let bestStart = bestIdx;
  if (windowLen === 2) {
    let bestWinScore = Infinity;
    for (let s = startIdx; s <= endIdx - 1; s++) {
      const sc = (Number(risk_level[s] || 0) + Number(risk_level[s + 1] || 0)) * 2.0
        + (Number(precip[s] ?? 0) + Number(precip[s + 1] ?? 0)) * 0.12;
      if (sc < bestWinScore) {
        bestWinScore = sc;
        bestStart = s;
      }
    }
  }

  const days = windowLen === 2 ? [x[bestStart], x[bestStart + 1]] : [x[bestIdx]];
  const reasons = [];
  const avgRisk = windowLen === 2
    ? (Number(risk_level[bestStart] || 0) + Number(risk_level[bestStart + 1] || 0)) / 2
    : Number(risk_level[bestIdx] || 0);

  reasons.push(`Risk level is low (avg L${avgRisk.toFixed(1)})`);

  const pr0 = Number(precip[bestStart] ?? 0);
  if (pr0 <= 2) reasons.push('Low precipitation');
  else reasons.push('Watch for precipitation');

  const thr = payload.thresholds?.crowd || 18500;
  const crowd0 = windowLen === 2
    ? Math.max(Number(pred[bestStart] || 0), Number(pred[bestStart + 1] || 0))
    : Number(pred[bestIdx] || 0);
  if (crowd0 < thr) reasons.push('Crowd below threshold');
  else reasons.push('Crowd may be high');

  return { days, reasons };
}

function updateBestWindow(payload) {
  const el = $('bestWindow');
  if (!el) return;
  const { days, reasons } = pickBestWindow(payload);
  const title = days.length === 2 ? `${days[0]} → ${days[1]}` : days[0];
  el.innerHTML = `
    <div style="font-weight:750; margin-bottom:6px;">${title}</div>
    <ul style="margin:0; padding-left:18px; color:rgba(17,24,39,.78);">
      ${reasons.map(r => `<li>${r}</li>`).join('')}
    </ul>
  `;
}

function syncWeatherCardForDate(payload, date) {
  const x = payload.time_axis || [];
  const idx = x.indexOf(date);

  safeText($('weatherDate'), date ? date : 'No date selected');

  if (idx < 0) {
    safeText($('weatherTemp'), '--°C');
    safeText($('weatherMeta'), 'Click a point on the chart to view daily weather.');
    safeText($('wPrecip'), '--');
    safeText($('wTempHL'), '--');
    safeText($('wWind'), '--');
    safeText($('wAqi'), '--');
    return;
  }

  const w = payload.weather || {};
  const tHi = w.temp_high_c?.[idx];
  const tLo = w.temp_low_c?.[idx];
  const pr = w.precip_mm?.[idx];
  const wc = w.weather_code_en?.[idx];
  const wind = w.wind_level?.[idx];
  const windMax = w.wind_max?.[idx];
  const aqi = w.aqi_value?.[idx];
  const aqiLv = w.aqi_level_en?.[idx];

  const midTemp = (Number.isFinite(Number(tHi)) && Number.isFinite(Number(tLo)))
    ? (Number(tHi) + Number(tLo)) / 2
    : (Number.isFinite(Number(tHi)) ? Number(tHi) : (Number.isFinite(Number(tLo)) ? Number(tLo) : null));

  safeText($('weatherTemp'), midTemp === null ? '--°C' : `${Math.round(midTemp)}°C`);
  safeText($('weatherMeta'), wc ? `Condition: ${wc}` : '');

  safeText($('wPrecip'), pr === null || pr === undefined ? '--' : `${fmt1(pr, ' mm')}`);
  safeText($('wTempHL'), `${fmt1(tHi, '°C')} / ${fmt1(tLo, '°C')}`);

  const windText = [
    wind ? `L${wind}` : null,
    Number.isFinite(Number(windMax)) ? `${fmt1(windMax, ' m/s')}` : null,
  ].filter(Boolean).join(' · ');
  safeText($('wWind'), windText || '--');

  const aqiText = [
    Number.isFinite(Number(aqi)) ? `${Math.round(Number(aqi))}` : null,
    aqiLv ? `${aqiLv}` : null,
  ].filter(Boolean).join(' · ');
  safeText($('wAqi'), aqiText || '--');
}

function syncThermo(payload, date) {
  const thrCrowd = payload.thresholds?.crowd ?? 18500;
  safeText($('thrCrowd'), fmtInt(thrCrowd));

  const q = payload.thresholds?.weather_quantiles || {};
  const thrW = payload.thresholds?.weather || {};
  const qText = `P>${q.precip_high ?? '--'} / TH>${q.temp_high ?? '--'} / TL<${q.temp_low ?? '--'}`;
  safeText($('thrWeather'), qText);

  const elFill = $('thermoFill');

  if (!date) {
    safeText($('thermoTitle'), 'No date selected');
    safeText($('thermoSubtitle'), 'Showing default thresholds');
    safeText($('thermoScore'), `Crowd ≥ ${fmtInt(thrCrowd)}`);
    safeText($('thermoLevel'), `Precip ≥ ${fmt1(thrW.precip_high, 'mm')} · TH ≥ ${fmt1(thrW.temp_high, '°C')} · TL ≤ ${fmt1(thrW.temp_low, '°C')}`);
    if (elFill) elFill.style.height = '12%';
    return;
  }

  const x = payload.time_axis || [];
  const idx = x.indexOf(date);
  if (idx < 0) return;

  const { risk_level, p_warn } = getPrimarySeries(payload);
  const lv = Number(risk_level[idx] || 0);
  const score = Number(p_warn[idx] ?? 0.15);

  // Map to 0..100 for thermometer
  const v = Math.max(0, Math.min(1, (lv / 3) * 0.65 + score * 0.35));
  const pct = Math.round(v * 100);

  safeText($('thermoTitle'), date);
  safeText($('thermoSubtitle'), `Primary model: ${UIW.state.primaryModelId}`);
  safeText($('thermoScore'), `${pct}/100`);
  safeText($('thermoLevel'), lv >= 3 ? 'High risk' : (lv > 0 ? 'Warning' : 'OK'));
  if (elFill) elFill.style.height = `${Math.max(6, Math.min(100, pct))}%`;
}

function syncSideCards() {
  const payload = UIW.state.payload;
  if (!payload) return;
  const date = UIW.state.selectedDate;
  syncWeatherCardForDate(payload, date);
  syncThermo(payload, date);
}

function applyCurveVisibility() {
  const chart = UIW.state.visitorChart;
  if (!chart) return;
  chart.setOption({
    legend: {
      selected: {
        'Actual': UIW.state.seriesVisible.actual,
        'Champion': UIW.state.seriesVisible.champion,
        'Runner-up': UIW.state.seriesVisible.runner,
        'Threshold': true,
      }
    }
  });
}

async function refreshForecast() {
  setWarning(null);
  initCharts();

  const payload = await apiGetJson(`/api/forecast?h=${encodeURIComponent(UIW.state.h)}&include_all=1`);
  UIW.state.payload = payload;

  renderKpis(payload);
  renderVisitorChart(payload);
  renderWeatherChart(payload);
  renderTimeline(payload);
  updateBestWindow(payload);

  // When payload refreshes, keep selectedDate if still in axis
  if (UIW.state.selectedDate && !(payload.time_axis || []).includes(UIW.state.selectedDate)) {
    UIW.state.selectedDate = null;
  }
  syncSideCards();

  if (payload.warning) setWarning(payload.warning);
}

function resetControls() {
  UIW.state.primaryModelId = 'champion';
  UIW.state.h = 7;
  UIW.state.selectedDate = null;
  UIW.state.showBacktest = false;
  UIW.state.seriesVisible = { actual: true, champion: true, runner: true };

  setPrimaryModel('champion');
  setQuickViewActive(7);

  $('tglActual').checked = true;
  $('tglChampion').checked = true;
  $('tglRunner').checked = true;
  $('tglBacktest').checked = false;

  applyCurveVisibility();
}

async function boot() {
  try {
    applyLightTheme();
    initCharts();

    await loadModels();

    // Defaults
    setPrimaryModel('champion');
    setQuickViewActive(UIW.state.h);

    // Quick view buttons
    for (const btnId of ['btnView1', 'btnView3', 'btnView7']) {
      const b = $(btnId);
      b.addEventListener('click', async () => {
        UIW.state.h = Number(b.dataset.h);
        setQuickViewActive(UIW.state.h);
        await refreshForecast();
      });
    }

    // Primary model
    $('btnChampion').addEventListener('click', () => {
      setPrimaryModel('champion');
      renderKpis(UIW.state.payload || {});
      renderTimeline(UIW.state.payload || {});
      syncSideCards();
      updateBestWindow(UIW.state.payload || {});
    });

    $('btnRunner').addEventListener('click', () => {
      setPrimaryModel('runner_up');
      renderKpis(UIW.state.payload || {});
      renderTimeline(UIW.state.payload || {});
      syncSideCards();
      updateBestWindow(UIW.state.payload || {});
    });

    // Curve toggles
    $('tglActual').addEventListener('change', (e) => {
      UIW.state.seriesVisible.actual = !!e.target.checked;
      applyCurveVisibility();
    });
    $('tglChampion').addEventListener('change', (e) => {
      UIW.state.seriesVisible.champion = !!e.target.checked;
      applyCurveVisibility();
    });
    $('tglRunner').addEventListener('change', (e) => {
      UIW.state.seriesVisible.runner = !!e.target.checked;
      applyCurveVisibility();
    });
    $('tglBacktest').addEventListener('change', async (e) => {
      UIW.state.showBacktest = !!e.target.checked;
      // Only re-render visitor chart
      if (UIW.state.payload) renderVisitorChart(UIW.state.payload);
    });

    $('btnRefresh').addEventListener('click', refreshForecast);
    $('btnReset').addEventListener('click', async () => {
      resetControls();
      await refreshForecast();
    });

    resetControls();
    await refreshForecast();

  } catch (e) {
    setWarning(`Failed to load: ${e.message}`);
  }
}

boot();
