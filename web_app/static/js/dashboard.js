/* dashboard.js - light minimalist dashboard wiring */

const UIW = {
  state: {
    models: [],
    primaryModelId: 'champion',
    h: 7,
    selectedDate: null,
    showBacktest: false,
    onlineForecast: false,
    lang: 'zh',
    theme: 'light',
    seriesVisible: {
      actual: true,
      champion: true,
      runner: true,
    },
    payload: null,
    visitorChart: null,
    weatherChart: null,
    activeMainPanel: 'visitors',
  }
};

const I18N = {
  zh: {
    page_title: '九寨沟客流预测看板',
    meta_default: '离线产物模式',
    btn_latest_1: '最近1天',
    btn_latest_3: '最近3天',
    btn_latest_7: '最近7天',
    btn_champion: '冠军模型',
    btn_runner: '亚军模型',
    tgl_actual: '真实',
    tgl_champion: '冠军预测',
    tgl_runner: '亚军预测',
    tgl_backtest: '回测段',
    tgl_theme: '日/夜',
    tgl_lang: '中/EN',
    tgl_online: '在线预测',
    btn_refresh: '刷新',
    btn_reset: '一键重置',
    kpi_max_forecast: '预测峰值（视窗）',
    kpi_warn_days: '预警天数（风险>0）',
    kpi_warn_foot: '客流 + 天气',
    kpi_max_risk: '最高风险等级',
    kpi_primary_model: '主模型',
    place_name: '九寨沟',
    kpi_artifact_note: '离线产物（回测最新窗口）',
    panel_visitors: '客流预测',
    hint_click: '提示：鼠标悬停查看十字线；点击日期同步“天气/风险”。',
    panel_weather: '天气',
    weather_no_date: '未选择日期',
    weather_click_tip: '点击图表上的点查看当日天气与风险标记。',
    weather_precip: '降水',
    weather_temp_hl: '温度（高/低）',
    weather_wind: '风',
    weather_aqi: '空气质量',
    panel_thresholds: '阈值 & 风险',
    thermo_no_date: '未选择日期',
    thermo_default: '展示默认阈值',
    thermo_crowd_thr: '客流阈值',
    thermo_weather_q: '天气分位阈值',
    panel_best_window: '最佳出行窗口（视窗）',
    best_wait: '等待预测结果…',
    panel_weather_series: '天气时间序列（参考）',
    panel_risk_timeline: '风险时间线',
    loading: '加载中…',

    legend_actual: '真实',
    legend_champion: '冠军预测',
    legend_runner: '亚军预测',
    legend_threshold: '阈值',
    tooltip_threshold: '客流阈值',

    best_title: '推荐窗口',
    best_reason_low_risk: (lv) => `风险较低（平均 L${lv}）`,
    best_reason_precip_low: '降水较少',
    best_reason_precip_watch: '注意降水',
    best_reason_crowd_below: '客流低于阈值',
    best_reason_crowd_high: '客流可能偏高',
    best_reason_holiday_avoid: (name) => `尽量避开节假日：${name}`,
    best_reason_holiday_ok: '无明显节假日拥挤影响',

    thermo_level_ok: '正常',
    thermo_level_warn: '预警',
    thermo_level_high: '高风险',
    meta_line: (genAt, h, modelName) => `生成时间：${genAt || '--'} · 视窗：最近${h}天 · 主模型：${modelName}`,
    kpi_date_prefix: '日期：',
    kpi_drivers_prefix: '原因：',

    drv_crowd_over_threshold: '客流预测超过阈值',
    drv_precip_high: '降水偏强',
    drv_temp_high: '高温',
    drv_temp_low: '低温',
  },
  en: {
    page_title: 'Jiuzhaigou Dashboard',
    meta_default: 'Offline artifact mode',
    btn_latest_1: 'Latest 1D',
    btn_latest_3: 'Latest 3D',
    btn_latest_7: 'Latest 7D',
    btn_champion: 'Champion',
    btn_runner: 'Runner-up',
    tgl_actual: 'Actual',
    tgl_champion: 'Champion',
    tgl_runner: 'Runner-up',
    tgl_backtest: 'Backtest',
    tgl_theme: 'Day/Night',
    tgl_lang: 'CN/EN',
    tgl_online: 'Online forecast',
    btn_refresh: 'Refresh',
    btn_reset: 'Reset',
    kpi_max_forecast: 'Max forecast (view)',
    kpi_warn_days: 'Warn days (risk>0)',
    kpi_warn_foot: 'Crowd + Weather',
    kpi_max_risk: 'Max risk level',
    kpi_primary_model: 'Primary model',
    place_name: 'Jiuzhaigou',
    kpi_artifact_note: 'Offline artifacts (latest window backtest)',
    panel_visitors: 'Visitor forecast',
    hint_click: 'Tip: hover for crosshair; click a date to sync Weather & Risk.',
    panel_weather: 'Weather',
    weather_no_date: 'No date selected',
    weather_click_tip: 'Click a point on the chart to view daily weather & risk flags.',
    weather_precip: 'Precip',
    weather_temp_hl: 'Temp (H/L)',
    weather_wind: 'Wind',
    weather_aqi: 'AQI',
    panel_thresholds: 'Thresholds & risk',
    thermo_no_date: 'No date selected',
    thermo_default: 'Showing default thresholds',
    thermo_crowd_thr: 'Crowd threshold',
    thermo_weather_q: 'Weather quantiles',
    panel_best_window: 'Best travel window (view)',
    best_wait: 'Waiting for forecast…',
    panel_weather_series: 'Weather series (reference)',
    panel_risk_timeline: 'Risk timeline',
    loading: 'Loading…',

    legend_actual: 'Actual',
    legend_champion: 'Champion',
    legend_runner: 'Runner-up',
    legend_threshold: 'Threshold',
    tooltip_threshold: 'Threshold',

    best_title: 'Recommended window',
    best_reason_low_risk: (lv) => `Low risk (avg L${lv})`,
    best_reason_precip_low: 'Low precipitation',
    best_reason_precip_watch: 'Watch for precipitation',
    best_reason_crowd_below: 'Crowd below threshold',
    best_reason_crowd_high: 'Crowd may be high',
    best_reason_holiday_avoid: (name) => `Avoid holiday peak: ${name}`,
    best_reason_holiday_ok: 'No obvious holiday crowd impact',

    thermo_level_ok: 'OK',
    thermo_level_warn: 'Warning',
    thermo_level_high: 'High risk',
    meta_line: (genAt, h, modelName) => `Generated: ${genAt || '--'} · View: latest ${h}D · Primary: ${modelName}`,
    kpi_date_prefix: 'Date: ',
    kpi_drivers_prefix: 'Drivers: ',

    drv_crowd_over_threshold: 'Crowd above threshold',
    drv_precip_high: 'High precipitation',
    drv_temp_high: 'High temperature',
    drv_temp_low: 'Low temperature',
  }
};

function t(key, ...args) {
  const lang = UIW.state.lang;
  const v = I18N?.[lang]?.[key] ?? I18N?.en?.[key] ?? '';
  return (typeof v === 'function') ? v(...args) : v;
}

function $(id) { return document.getElementById(id); }

function safeText(el, text) {
  if (!el) return;
  el.textContent = (text === null || text === undefined) ? '' : String(text);
}

function fmtInt(x) {
  if (x === null || x === undefined) return '--';
  const n = Number(x);
  if (!Number.isFinite(n)) return '--';
  return Math.round(n).toLocaleString(UIW.state.lang === 'zh' ? 'zh-CN' : 'en-US');
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
  // base.html sets data-bs-theme="dark" globally; dashboard defaults to light.
  document.documentElement.setAttribute('data-bs-theme', 'light');
  document.documentElement.setAttribute('data-uiw-theme', 'light');
}

function applyTheme(theme) {
  UIW.state.theme = theme;
  const isDark = theme === 'dark';
  document.documentElement.setAttribute('data-bs-theme', isDark ? 'dark' : 'light');
  document.documentElement.setAttribute('data-uiw-theme', isDark ? 'dark' : 'light');
}

function applyLang(lang) {
  UIW.state.lang = lang;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (!key) return;
    const val = t(key);
    if (val) el.textContent = val;
  });

  // Re-render charts/UI text that is not in DOM attributes
  if (UIW.state.payload) {
    renderKpis(UIW.state.payload);
    renderVisitorChart(UIW.state.payload);
    renderWeatherChart(UIW.state.payload);
    renderTimeline(UIW.state.payload);
    updateBestWindow(UIW.state.payload);
    syncSideCards();
  }
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

  // Fallback: click anywhere on the plot area to select nearest x-axis date.
  // (Line series may not emit a point click when symbols are hidden.)
  try {
    const zr = UIW.state.visitorChart.getZr();
    zr.off('click');
    zr.on('click', (ev) => {
      const chart = UIW.state.visitorChart;
      if (!chart) return;

      // Only react to clicks inside the main plotting grid (avoid dataZoom slider clicks).
      try {
        if (!chart.containPixel('grid', [ev.offsetX, ev.offsetY])) return;
      } catch (_) {}

      const payload = UIW.state.payload;
      const x = payload?.time_axis || chart.getOption?.()?.xAxis?.[0]?.data || [];
      if (!x || !x.length) return;

      // Convert pixel -> xAxis value. For category axis this is usually an index.
      const v = chart.convertFromPixel({ xAxisIndex: 0 }, [ev.offsetX, ev.offsetY]);
      const xi = Array.isArray(v) ? v[0] : v;
      let idx = -1;
      if (typeof xi === 'number' && Number.isFinite(xi)) idx = Math.round(xi);
      else if (typeof xi === 'string') idx = x.indexOf(xi);
      if (idx < 0 || idx >= x.length) return;

      UIW.state.selectedDate = x[idx];
      syncSideCards();
    });
  } catch (_) {}
}

function setMainPanel(panel) {
  UIW.state.activeMainPanel = panel === 'weather' ? 'weather' : 'visitors';
  const isV = UIW.state.activeMainPanel === 'visitors';
  // Tabs reuse chip styles (STYLE_MAP)
  $('tabVisitors')?.classList.toggle('uiw-chip--active', isV);
  $('tabWeather')?.classList.toggle('uiw-chip--active', !isV);
  $('tabVisitors')?.setAttribute('aria-selected', isV ? 'true' : 'false');
  $('tabWeather')?.setAttribute('aria-selected', !isV ? 'true' : 'false');
  $('visitorChartWrap')?.classList.toggle('uiw-panel-body--active', isV);
  $('weatherChartWrap')?.classList.toggle('uiw-panel-body--active', !isV);
  // Resize chart after DOM visibility change
  setTimeout(() => {
    UIW.state.visitorChart?.resize();
    UIW.state.weatherChart?.resize();
  }, 30);
}

function buildHolidayMarkAreas(payload) {
  const x = payload.time_axis || [];
  const ranges = payload.holidays || [];
  if (!x.length || !ranges.length) return [];

  const start = x[0];
  const end = x[x.length - 1];

  const colorFor = (h) => {
    const name = ((h.name_zh || h.name_en || h.name || '') + '').toLowerCase();
    const type = (h.type || 'festival').toLowerCase();
    if (name.includes('春节') || name.includes('spring')) return 'rgba(255,59,48,.12)';
    if (name.includes('国庆') || name.includes('national')) return 'rgba(255,149,0,.12)';
    if (type === 'vacation' && (name.includes('暑') || name.includes('summer'))) return 'rgba(10,132,255,.10)';
    if (type === 'vacation' && (name.includes('寒') || name.includes('winter'))) return 'rgba(88,86,214,.10)';
    return type === 'vacation' ? 'rgba(10,132,255,.08)' : 'rgba(255,149,0,.10)';
  };

  const within = (a, b, s, e) => !(b < s || a > e);

  const pickName = (h) => {
    if (UIW.state.lang === 'zh') return h.name_zh || h.name || 'Holiday';
    return h.name_en || h.name || 'Holiday';
  };

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
          formatter: pickName(h)
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

  const LEG_ACT = t('legend_actual');
  const LEG_C = t('legend_champion');
  const LEG_R = t('legend_runner');
  const LEG_T = t('legend_threshold');

  UIW.state.visitorChart.setOption({
    animationDuration: 450,
    tooltip: {
      // Restore axis tooltip for hover readout (actual passenger value per day).
      // Disable axisPointer visuals to avoid the annoying crosshair effect,
      // especially when hovering the dataZoom slider.
      trigger: 'axis',
      axisPointer: { type: 'none' },
      confine: true,
      formatter: (items) => {
        const arr = Array.isArray(items) ? items : (items ? [items] : []);
        if (!arr.length) return '';
        const date = arr[0]?.axisValueLabel || arr[0]?.name || '';
        const rows = arr
          .filter(it => it && it.seriesName)
          .map(it => {
            const v = Array.isArray(it.value) ? it.value?.[1] : it.value;
            const val = (v === null || v === undefined || v === '-') ? '--' : fmtInt(v);
            return `${it.marker}${it.seriesName}: ${val}`;
          });
        return `${date}<br/>${rows.join('<br/>')}`;
      }
    },
    // Extra guard: ensure no global axisPointer is rendered.
    axisPointer: { show: false },
    legend: {
      data: [LEG_ACT, LEG_C, LEG_R, LEG_T],
      top: 6,
      textStyle: { color: 'rgba(17,24,39,.72)' },
      selected: {
        [LEG_ACT]: UIW.state.seriesVisible.actual,
        [LEG_C]: UIW.state.seriesVisible.champion,
        [LEG_R]: UIW.state.seriesVisible.runner,
        [LEG_T]: true,
      }
    },
    grid: { left: 52, right: 18, top: 46, bottom: 58 },
    xAxis: {
      type: 'category',
      data: x,
      boundaryGap: false,
      axisLabel: { color: 'rgba(17,24,39,.55)' },
      axisLine: { lineStyle: { color: 'rgba(17,24,39,.18)' } },
      axisPointer: { show: false }
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
        name: LEG_ACT,
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
        name: LEG_C,
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
        name: LEG_R,
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
        name: LEG_T,
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

  const L1 = UIW.state.lang === 'zh' ? '降水 (mm)' : 'Precip (mm)';
  const L2 = UIW.state.lang === 'zh' ? '最高温 (°C)' : 'Temp High (°C)';
  const L3 = UIW.state.lang === 'zh' ? '最低温 (°C)' : 'Temp Low (°C)';

  UIW.state.weatherChart.setOption({
    tooltip: { trigger: 'item' },
    legend: {
      data: [L1, L2, L3],
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
      { name: L1, type: 'bar', data: precip, yAxisIndex: 0, itemStyle: { color: 'rgba(10,132,255,.22)' } },
      { name: L2, type: 'line', data: tHi, yAxisIndex: 1, smooth: true, lineStyle: { width: 2, color: 'rgba(255,149,0,.9)' } },
      { name: L3, type: 'line', data: tLo, yAxisIndex: 1, smooth: true, lineStyle: { width: 2, color: 'rgba(52,199,89,.85)' } },
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
    crowd_alert: payload?.risk?.[p]?.crowd_alert || [],
    weather_hazard: payload?.risk?.[p]?.weather_hazard || [],
    risk_score: payload?.risk?.[p]?.risk_score || [],
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
  safeText($('kpiMaxVisitorDate'), maxIdx >= 0 ? `${t('kpi_date_prefix')}${x[maxIdx]}` : '--');
  safeText($('kpiWarnDays'), warnDays.toString());
  safeText($('kpiMaxRisk'), String(maxRisk));

  const d = (maxRiskIdx >= 0 ? (drivers[maxRiskIdx] || []) : []);
  const dText = (d || [])
    .map(code => {
      if (!code) return null;
      if (code === 'crowd_over_threshold') return t('drv_crowd_over_threshold');
      if (code === 'precip_high') return t('drv_precip_high');
      if (code === 'temp_high') return t('drv_temp_high');
      if (code === 'temp_low') return t('drv_temp_low');
      return String(code);
    })
    .filter(Boolean);
  safeText($('kpiMaxRiskDrivers'), dText.length ? `${t('kpi_drivers_prefix')}${dText.join(' · ')}` : `${t('kpi_drivers_prefix')}--`);

  const meta = getPrimaryModelMeta(payload);
  safeText($('kpiModel'), meta?.model_name || UIW.state.primaryModelId);
  // Do not display run_dir paths in UI; show a stable offline mode note.
  safeText($('kpiRunDir'), t('kpi_artifact_note'));

  safeText($('uiwMetaLine'), t('meta_line', payload.meta?.generated_at || '--', UIW.state.h, meta?.model_name || UIW.state.primaryModelId));
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
    const badgeText = UIW.state.lang === 'zh'
      ? (lv === 0 ? '正常' : (lv >= 3 ? '高风险' : '预警'))
      : (lv === 0 ? 'OK' : (lv >= 3 ? 'High' : 'Warn'));

    day.innerHTML = `
      <div class="uiw-day-title">${x[i]}</div>
      <div class="uiw-badge ${badgeClass}">${badgeText} (L${lv})</div>
      <div class="uiw-muted" style="margin-top:6px; font-size:.86rem;">${((drivers[i]||[]).slice(0,3).map(code => {
        if (code === 'crowd_over_threshold') return t('drv_crowd_over_threshold');
        if (code === 'precip_high') return t('drv_precip_high');
        if (code === 'temp_high') return t('drv_temp_high');
        if (code === 'temp_low') return t('drv_temp_low');
        return String(code || '');
      }).filter(Boolean).join(' / ')) || '—'}</div>
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

  const holidayOf = (dateStr) => {
    const ranges = payload.holidays || [];
    for (const h of ranges) {
      if (!h?.start || !h?.end) continue;
      if (dateStr >= h.start && dateStr <= h.end) return h;
    }
    return null;
  };

  // Simple heuristic scoring (lower is better)
  // - risk_level dominates
  // - weather is secondary
  // - holiday adds penalty
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

     const hh = holidayOf(x[i]);
     const holidayPenalty = hh ? (String(hh.type || '').toLowerCase() === 'festival' ? 2.0 : 1.0) : 0.0;

    const score = (lv * 3.0) + (pr * 0.15) + (tempPenalty * 1.0) + (crowdPenalty * 1.2) + (conf * 0.6) + (holidayPenalty * 1.2);
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

  reasons.push(t('best_reason_low_risk', avgRisk.toFixed(1)));

  const pr0 = Number(precip[bestStart] ?? 0);
  if (pr0 <= 2) reasons.push(t('best_reason_precip_low'));
  else reasons.push(t('best_reason_precip_watch'));

  const thr = payload.thresholds?.crowd || 18500;
  const crowd0 = windowLen === 2
    ? Math.max(Number(pred[bestStart] || 0), Number(pred[bestStart + 1] || 0))
    : Number(pred[bestIdx] || 0);
  if (crowd0 < thr) reasons.push(t('best_reason_crowd_below'));
  else reasons.push(t('best_reason_crowd_high'));

  // Holiday reason
  const h0 = holidayOf(days[0]);
  const h1 = (days.length === 2) ? holidayOf(days[1]) : null;
  const hh = h0 || h1;
  if (hh) {
    const nm = (UIW.state.lang === 'zh')
      ? (hh.name_zh || hh.name || hh.type || '节假日')
      : (hh.name_en || hh.name || hh.type || 'Holiday');
    reasons.push(t('best_reason_holiday_avoid', nm));
  }
  else reasons.push(t('best_reason_holiday_ok'));

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

  safeText($('weatherDate'), date ? date : t('weather_no_date'));

  const setRow = (rowId, show) => {
    const row = $(rowId);
    if (!row) return;
    row.style.display = show ? '' : 'none';
  };

  const setFlags = (flags) => {
    const box = $('weatherFlags');
    if (!box) return;
    if (!flags || !flags.length) {
      box.style.display = 'none';
      box.innerHTML = '';
      return;
    }
    box.style.display = 'flex';
    box.innerHTML = flags.map(f => `<span class="uiw-flag uiw-flag--${f.kind}">${f.text}</span>`).join('');
  };

  if (idx < 0) {
    safeText($('weatherTemp'), '--°C');
    safeText($('weatherMeta'), t('weather_click_tip'));
    setFlags([]);
    setRow('rowPrecip', false);
    setRow('rowTempHL', false);
    setRow('rowWind', false);
    setRow('rowAqi', false);
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
  safeText($('weatherMeta'), wc ? (UIW.state.lang === 'zh' ? `天气：${wc}` : `Condition: ${wc}`) : '');

  // Hide missing rows (no placeholders)
  const hasPrecip = Number.isFinite(Number(pr));
  const hasTemp = Number.isFinite(Number(tHi)) || Number.isFinite(Number(tLo));
  const hasWind = (wind !== null && wind !== undefined && wind !== 0) || Number.isFinite(Number(windMax));
  const hasAqi = Number.isFinite(Number(aqi)) || !!aqiLv;

  setRow('rowPrecip', hasPrecip);
  setRow('rowTempHL', hasTemp);
  setRow('rowWind', hasWind);
  setRow('rowAqi', hasAqi);

  safeText($('wPrecip'), hasPrecip ? `${fmt1(pr, ' mm')}` : '');
  safeText($('wTempHL'), hasTemp ? `${fmt1(tHi, '°C')} / ${fmt1(tLo, '°C')}` : '');

  const windText = [
    wind ? `L${wind}` : null,
    Number.isFinite(Number(windMax)) ? `${fmt1(windMax, ' m/s')}` : null,
  ].filter(Boolean).join(' · ');
  safeText($('wWind'), windText || '');

  const aqiText = [
    Number.isFinite(Number(aqi)) ? `${Math.round(Number(aqi))}` : null,
    aqiLv ? `${aqiLv}` : null,
  ].filter(Boolean).join(' · ');
  safeText($('wAqi'), aqiText || '');

  // Hazard flags (crowd / weather / drivers)
  const { crowd_alert, weather_hazard, drivers } = getPrimarySeries(payload);
  const flags = [];
  const ca = !!crowd_alert?.[idx];
  const whz = !!weather_hazard?.[idx];
  const dd = (drivers?.[idx] || []);

  if (!ca && !whz && dd.length === 0) {
    flags.push({ kind: 'ok', text: UIW.state.lang === 'zh' ? '无明显风险' : 'No obvious risk' });
  } else {
    if (ca) flags.push({ kind: 'risk', text: UIW.state.lang === 'zh' ? '客流偏高' : 'Crowd high' });
    if (whz) flags.push({ kind: 'warn', text: UIW.state.lang === 'zh' ? '天气风险' : 'Weather hazard' });
    for (const d of dd.slice(0, 3)) {
      const txt = (d === 'crowd_over_threshold') ? t('drv_crowd_over_threshold')
        : (d === 'precip_high') ? t('drv_precip_high')
        : (d === 'temp_high') ? t('drv_temp_high')
        : (d === 'temp_low') ? t('drv_temp_low')
        : String(d || '');
      const kind = (d === 'crowd_over_threshold') ? 'risk' : 'warn';
      if (txt) flags.push({ kind, text: txt });
    }
  }
  setFlags(flags);
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
    safeText($('thermoTitle'), t('thermo_no_date'));
    safeText($('thermoSubtitle'), t('thermo_default'));
    safeText($('thermoScore'), UIW.state.lang === 'zh' ? `客流 ≥ ${fmtInt(thrCrowd)}` : `Crowd ≥ ${fmtInt(thrCrowd)}`);
    safeText($('thermoLevel'), UIW.state.lang === 'zh'
      ? `降水 ≥ ${fmt1(thrW.precip_high, 'mm')} · 高温 ≥ ${fmt1(thrW.temp_high, '°C')} · 低温 ≤ ${fmt1(thrW.temp_low, '°C')}`
      : `Precip ≥ ${fmt1(thrW.precip_high, 'mm')} · TH ≥ ${fmt1(thrW.temp_high, '°C')} · TL ≤ ${fmt1(thrW.temp_low, '°C')}`);
    if (elFill) elFill.style.height = '12%';
    return;
  }

  const x = payload.time_axis || [];
  const idx = x.indexOf(date);
  if (idx < 0) return;

  const { risk_level, p_warn, risk_score } = getPrimarySeries(payload);
  const lv = Number(risk_level[idx] || 0);
  const score = Number(p_warn[idx] ?? 0.15);
  const pct = Number.isFinite(Number(risk_score?.[idx]))
    ? Math.round(Number(risk_score[idx]))
    : Math.round(Math.max(0, Math.min(1, (lv / 3) * 0.65 + score * 0.35)) * 100);

  safeText($('thermoTitle'), date);
  const meta = getPrimaryModelMeta(payload);
  const modelLabel = meta?.model_name || UIW.state.primaryModelId;
  safeText($('thermoSubtitle'), UIW.state.lang === 'zh'
    ? `主模型：${modelLabel}`
    : `Primary model: ${modelLabel}`);
  safeText($('thermoScore'), `${pct}/100`);
  safeText($('thermoLevel'), lv >= 3 ? t('thermo_level_high') : (lv > 0 ? t('thermo_level_warn') : t('thermo_level_ok')));
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
  const LEG_ACT = t('legend_actual');
  const LEG_C = t('legend_champion');
  const LEG_R = t('legend_runner');
  const LEG_T = t('legend_threshold');
  chart.setOption({
    legend: {
      selected: {
        [LEG_ACT]: UIW.state.seriesVisible.actual,
        [LEG_C]: UIW.state.seriesVisible.champion,
        [LEG_R]: UIW.state.seriesVisible.runner,
        [LEG_T]: true,
      }
    }
  });
}

async function refreshForecast() {
  setWarning(null);
  initCharts();

  const mode = UIW.state.onlineForecast ? 'online' : 'offline';
  const payload = await apiGetJson(`/api/forecast?h=${encodeURIComponent(UIW.state.h)}&include_all=1&mode=${encodeURIComponent(mode)}`);
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
  UIW.state.onlineForecast = false;
  UIW.state.seriesVisible = { actual: true, champion: true, runner: true };

  setPrimaryModel('champion');
  setQuickViewActive(7);

  $('tglActual').checked = true;
  $('tglChampion').checked = true;
  $('tglRunner').checked = true;
  $('tglBacktest').checked = false;
  $('tglOnline').checked = false;

  applyCurveVisibility();

  // Reset chart zoom to span all available dates
  try {
    UIW.state.visitorChart?.dispatchAction({ type: 'dataZoom', start: 0, end: 100 });
  } catch (_) {}
}

async function boot() {
  try {
    applyLightTheme();
    initCharts();

    // Defaults
    applyTheme('light');
    applyLang('zh');

    await loadModels();

    // Main panel tabs (Visitor / Weather)
    $('tabVisitors')?.addEventListener('click', () => setMainPanel('visitors'));
    $('tabWeather')?.addEventListener('click', () => setMainPanel('weather'));
    setMainPanel('visitors');

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

    // Online forecast toggle (default OFF)
    $('tglOnline')?.addEventListener('change', async (e) => {
      UIW.state.onlineForecast = !!e.target.checked;
      // Online forecast is a different timeline (appends future h days)
      UIW.state.selectedDate = null;
      await refreshForecast();
    });

    $('btnRefresh').addEventListener('click', refreshForecast);

    // Theme toggle (Day/Night)
    $('tglTheme')?.addEventListener('change', (e) => {
      applyTheme(e.target.checked ? 'dark' : 'light');
      // Re-render charts to match tooltip/axis colors if needed
      if (UIW.state.payload) {
        renderVisitorChart(UIW.state.payload);
        renderWeatherChart(UIW.state.payload);
      }
    });

    // Language toggle (CN/EN)
    $('tglLang')?.addEventListener('change', (e) => {
      applyLang(e.target.checked ? 'en' : 'zh');
    });
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
