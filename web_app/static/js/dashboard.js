/* dashboard.js - offline artifact dashboard wiring */

const UIW = {
  state: {
    models: [],
    activeModelId: 'champion',
    h: 7,
    visitorChart: null,
    weatherChart: null,
  }
};

function $(id) { return document.getElementById(id); }

function setWarning(msg) {
  const el = $('uiwWarning');
  if (!msg) {
    el.style.display = 'none';
    el.innerText = '';
    return;
  }
  el.style.display = 'block';
  el.innerText = msg;
}

function fmtInt(x) {
  if (x === null || x === undefined) return '--';
  const n = Number(x);
  if (!Number.isFinite(n)) return '--';
  return Math.round(n).toLocaleString('en-US');
}

function safeText(el, text) {
  if (!el) return;
  el.textContent = text;
}

async function apiGetJson(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data?.error || `HTTP ${res.status}`);
  }
  return data;
}

async function loadModels() {
  const data = await apiGetJson('/api/models');
  UIW.state.models = data.models || [];

  // Default to champion if present
  if (!UIW.state.models.find(m => m.model_id === UIW.state.activeModelId)) {
    UIW.state.activeModelId = UIW.state.models[0]?.model_id || 'champion';
  }

  // Disable runner button if missing
  const hasRunner = UIW.state.models.some(m => m.model_id === 'runner_up');
  $('btnRunner').disabled = !hasRunner;

  return data;
}

function setActiveModel(modelId) {
  UIW.state.activeModelId = modelId;
  $('btnChampion').classList.toggle('active', modelId === 'champion');
  $('btnRunner').classList.toggle('active', modelId === 'runner_up');
}

function initCharts() {
  const vEl = $('visitorChart');
  const wEl = $('weatherChart');

  if (!UIW.state.visitorChart) UIW.state.visitorChart = echarts.init(vEl);
  if (!UIW.state.weatherChart) UIW.state.weatherChart = echarts.init(wEl);

  window.addEventListener('resize', () => {
    UIW.state.visitorChart?.resize();
    UIW.state.weatherChart?.resize();
  });
}

function renderVisitorChart(payload) {
  const x = payload.time_axis || [];
  const pred = payload.visitor_pred || [];
  const actual = payload.visitor_actual || [];
  const thr = payload.thresholds?.crowd ?? 18500;
  const risk = payload.risk_level || [];

  // markArea based on risk level
  const markAreas = [];
  for (let i = 0; i < x.length; i++) {
    if ((risk[i] || 0) > 0) {
      markAreas.push([
        { xAxis: x[i] },
        { xAxis: x[i] }
      ]);
    }
  }

  UIW.state.visitorChart.setOption({
    tooltip: { trigger: 'axis' },
    legend: {
      data: ['Prediction', 'Actual', 'Threshold'],
      textStyle: { color: '#cbd5e1' }
    },
    grid: { left: 50, right: 20, top: 45, bottom: 35 },
    xAxis: {
      type: 'category',
      data: x,
      axisLabel: { color: '#94a3b8' },
      axisLine: { lineStyle: { color: 'rgba(148,163,184,0.25)' } }
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: 'rgba(148,163,184,0.15)' } }
    },
    series: [
      {
        name: 'Prediction',
        type: 'line',
        data: pred,
        smooth: true,
        lineStyle: { width: 3 },
        emphasis: { focus: 'series' },
        markArea: {
          itemStyle: { color: 'rgba(255, 157, 0, 0.10)' },
          data: markAreas
        }
      },
      {
        name: 'Actual',
        type: 'line',
        data: actual,
        smooth: true,
        lineStyle: { width: 2, type: 'dashed' },
        emphasis: { focus: 'series' }
      },
      {
        name: 'Threshold',
        type: 'line',
        data: x.map(() => thr),
        symbol: 'none',
        lineStyle: { type: 'dotted', width: 2, color: '#ef4444' },
      }
    ]
  });
}

function renderWeatherChart(payload) {
  const x = payload.time_axis || [];
  const precip = payload.weather?.precip_mm || [];
  const tHi = payload.weather?.temp_high_c || [];
  const tLo = payload.weather?.temp_low_c || [];

  UIW.state.weatherChart.setOption({
    tooltip: { trigger: 'axis' },
    legend: {
      data: ['Precip (mm)', 'Temp High (°C)', 'Temp Low (°C)'],
      textStyle: { color: '#cbd5e1' }
    },
    grid: { left: 50, right: 20, top: 45, bottom: 35 },
    xAxis: {
      type: 'category',
      data: x,
      axisLabel: { color: '#94a3b8' },
      axisLine: { lineStyle: { color: 'rgba(148,163,184,0.25)' } }
    },
    yAxis: [
      {
        type: 'value',
        name: 'mm',
        axisLabel: { color: '#94a3b8' },
        splitLine: { lineStyle: { color: 'rgba(148,163,184,0.15)' } }
      },
      {
        type: 'value',
        name: '°C',
        axisLabel: { color: '#94a3b8' },
        splitLine: { show: false }
      }
    ],
    series: [
      { name: 'Precip (mm)', type: 'bar', data: precip, yAxisIndex: 0, itemStyle: { color: 'rgba(0,242,255,0.35)' } },
      { name: 'Temp High (°C)', type: 'line', data: tHi, yAxisIndex: 1, smooth: true },
      { name: 'Temp Low (°C)', type: 'line', data: tLo, yAxisIndex: 1, smooth: true, lineStyle: { type: 'dashed' } },
    ]
  });
}

function renderKpis(payload) {
  const x = payload.time_axis || [];
  const pred = payload.visitor_pred || [];
  const risk = payload.risk_level || [];
  const drivers = payload.drivers || [];

  let maxV = -Infinity, maxIdx = -1;
  for (let i = 0; i < pred.length; i++) {
    const v = Number(pred[i]);
    if (Number.isFinite(v) && v > maxV) { maxV = v; maxIdx = i; }
  }

  const warnDays = risk.filter(v => (v || 0) > 0).length;
  const maxRisk = Math.max(...risk.map(v => v || 0), 0);
  const maxRiskIdx = risk.findIndex(v => (v || 0) === maxRisk);

  safeText($('kpiMaxVisitor'), maxIdx >= 0 ? fmtInt(maxV) : '--');
  safeText($('kpiMaxVisitorDate'), maxIdx >= 0 ? `Date: ${x[maxIdx]}` : '--');
  safeText($('kpiWarnDays'), warnDays.toString());
  safeText($('kpiMaxRisk'), String(maxRisk));

  const d = (maxRiskIdx >= 0 ? (drivers[maxRiskIdx] || []) : []);
  safeText($('kpiMaxRiskDrivers'), d.length ? `Drivers: ${d.join(' | ')}` : 'Drivers: --');

  safeText($('kpiModel'), payload.meta?.model_name || '--');
  safeText($('kpiRunDir'), payload.meta?.run_dir || '--');

  const metaLine = `生成时间：${payload.meta?.generated_at || '--'}  |  Model: ${payload.meta?.model_name || '--'}`;
  safeText($('uiwMetaLine'), metaLine);
}

function renderTimeline(payload) {
  const el = $('riskTimeline');
  const x = payload.time_axis || [];
  const risk = payload.risk_level || [];
  const ca = payload.crowd_alert || [];
  const wh = payload.weather_hazard || [];
  const drv = payload.drivers || [];

  el.innerHTML = '';
  for (let i = 0; i < x.length; i++) {
    const lv = risk[i] || 0;
    const day = document.createElement('div');
    day.className = 'uiw-day';

    const badgeClass = lv === 0 ? 'uiw-badge-ok' : (lv >= 3 ? 'uiw-badge-risk' : 'uiw-badge-warn');
    const badgeText = lv === 0 ? '正常' : (lv >= 3 ? '高风险' : '预警');

    const flags = [];
    if (ca[i]) flags.push('Crowd');
    if (wh[i]) flags.push('Weather');

    day.innerHTML = `
      <div class="uiw-day-title">${x[i]}</div>
      <div class="uiw-badge ${badgeClass}">${badgeText} (L${lv})</div>
      <div class="text-white-50" style="margin-top:6px; font-size:0.85rem;">${flags.length ? flags.join(' + ') : '—'}</div>
      <div class="text-white-50" style="margin-top:6px; font-size:0.8rem;">${(drv[i]||[]).join(' / ')}</div>
    `;
    el.appendChild(day);
  }
}

async function refreshForecast() {
  setWarning(null);
  initCharts();

  const modelId = UIW.state.activeModelId;
  const payload = await apiGetJson(`/api/forecast?model_id=${encodeURIComponent(modelId)}&h=${UIW.state.h}`);

  renderKpis(payload);
  renderVisitorChart(payload);
  renderWeatherChart(payload);
  renderTimeline(payload);

  if (payload.warning) setWarning(payload.warning);
}

async function boot() {
  try {
    initCharts();
    await loadModels();

    setActiveModel('champion');

    $('btnChampion').addEventListener('click', async () => {
      setActiveModel('champion');
      await refreshForecast();
    });

    $('btnRunner').addEventListener('click', async () => {
      setActiveModel('runner_up');
      await refreshForecast();
    });

    $('btnRefresh').addEventListener('click', refreshForecast);

    await refreshForecast();
  } catch (e) {
    setWarning(`加载失败：${e.message}`);
  }
}

boot();
