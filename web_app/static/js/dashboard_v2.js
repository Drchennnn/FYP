/* dashboard_v2.js

Manual test notes (P0):
1) Start Flask:
   - From repo root:  cd FYP/web_app && python app.py
   - Open: http://127.0.0.1:5000/dashboard/v2
2) Offline artifact mode (default):
   - Ensure FYP/output/backups/backup_STAR/ exists and contains run_STAR with STAR_test_predictions.csv
   - Page should load chart with Actual + Champion + Runner (if available)
3) Online forecast mode:
   - Toggle "在线预测" ON
   - If LSTM model is loaded successfully, last h days become true future forecast.
   - If not available, UI should show a warning and gracefully fall back.
4) Interactions:
   - Mouse wheel / drag inside chart: dataZoom should be smooth; crosshair should not block zooming.
   - Hover: tooltip shows passenger numbers (Actual/Champion/Runner).
   - Click on a point: Weather card + Risk/Thermo update for that day.
   - Holidays: visible as shaded markArea ranges.
5) Toggles:
   - 日/夜 toggles Bootstrap theme (data-bs-theme)
   - 中/EN toggles UI strings + holiday names where available

No external dependencies beyond ECharts (already in base.html).
*/

(() => {
  'use strict';
  console.log('dashboard_v2.js loaded');

  /** @typedef {{
   *  timeAxis: string[],
   *  forecast: {h:number, startIndex:number, endIndex:number},
   *  meta: {generatedAt?:string, forecastMode?:string, championName?:string, runnerName?:string|null},
   *  series: {actual:(number|null)[], champion:(number|null)[], runner:(number|null)[]},
   *  thresholds: {crowd?:number|null, weather?:{precipHigh?:number|null,tempHigh?:number|null,tempLow?:number|null}, weatherQuantiles?:any},
   *  weather: {precipMm:(number|null)[], tempHighC:(number|null)[], tempLowC:(number|null)[], weatherCodeEn:(string|null)[], windLevel:(number|null)[], windDirEn:(string|null)[], windMax:(number|null)[], aqiValue:(number|null)[], aqiLevelEn:(string|null)[]},
   *  holidays: {start:string,end:string,nameZh?:string,nameEn?:string,type?:string}[],
   *  risk: {champion?:any, runner?:any},
   *  warning?: string|null
   * }} NormalizedPayload
   */

  const I18N = {
    zh: {
      tgl_theme: '日/夜',
      tgl_lang: '中/EN',
      tgl_online: '在线预测',
      btn_refresh: '刷新',
      btn_reset: '一键重置',
      panel_visitors: '客流预测',
      panel_weather: '天气',
      panel_thresholds: '阈值 & 风险',
      panel_best_window: '最佳出行窗口（视窗）',
      hint_click: '点击日期同步“天气/风险”，滚轮缩放，拖拽平移。',
      weather_no_date: '未选择日期',
      weather_click_tip: '点击图表上的点查看当日天气与风险标记。',
      weather_precip: '降水',
      weather_temp_hl: '温度（高/低）',
      weather_wind: '风',
      weather_aqi: '空气质量',
      thermo_no_date: '未选择日期',
      thermo_default: '展示默认阈值',
      thermo_crowd_thr: '客流阈值',
      thermo_weather_q: '天气分位阈值',
      best_wait: '等待预测结果…',
      risk_lv_0: '正常',
      risk_lv_1: '关注',
      risk_lv_2: '预警',
      risk_lv_3: '高风险',
      offline_mode: 'Offline artifact mode',
      online_mode: 'Online forecast',
      warn_fallback: '在线预测失败，已自动回退离线产物。'
    },
    en: {
      tgl_theme: 'Light/Dark',
      tgl_lang: 'ZH/EN',
      tgl_online: 'Online',
      btn_refresh: 'Refresh',
      btn_reset: 'Reset',
      panel_visitors: 'Visitor Forecast',
      panel_weather: 'Weather',
      panel_thresholds: 'Thresholds & Risk',
      panel_best_window: 'Best Window (in view)',
      hint_click: 'Click a day to sync Weather/Risk. Wheel zoom, drag pan.',
      weather_no_date: 'No date selected',
      weather_click_tip: 'Click a point on chart to view daily weather & risk flags.',
      weather_precip: 'Precip',
      weather_temp_hl: 'Temp (Hi/Lo)',
      weather_wind: 'Wind',
      weather_aqi: 'Air Quality',
      thermo_no_date: 'No date selected',
      thermo_default: 'Showing default thresholds',
      thermo_crowd_thr: 'Crowd threshold',
      thermo_weather_q: 'Weather quantile threshold',
      best_wait: 'Waiting for forecast…',
      risk_lv_0: 'Normal',
      risk_lv_1: 'Watch',
      risk_lv_2: 'Warning',
      risk_lv_3: 'High Risk',
      offline_mode: 'Offline artifact mode',
      online_mode: 'Online forecast',
      warn_fallback: 'Online forecast failed; falling back to offline artifacts.'
    }
  };

  const $ = (id) => document.getElementById(id);

  const dom = {
    spinner: $('uiwSpinner'),
    alert: $('uiwAlert'),
    title: $('dv2Title'),
    meta: $('dv2Meta'),
    chartTitle: $('dv2ChartTitle'),
    chartHint: $('dv2ChartHint'),

    hBtns: [$('dv2H1'), $('dv2H3'), $('dv2H7')],
    modelBtns: [$('dv2ModelBoth'), $('dv2ModelChampion'), $('dv2ModelRunner'), $('dv2ModelThird')],

    tTheme: $('dv2Theme'),
    tLang: $('dv2Lang'),
    tOnline: $('dv2Online'),
    btnRefresh: $('dv2Refresh'),
    btnReset: $('dv2Reset'),

    // weather
    wDate: $('dv2WeatherDate'),
    wTemp: $('dv2WeatherTemp'),
    wMeta: $('dv2WeatherMeta'),
    wFlags: $('dv2WeatherFlags'),
    wPrecip: $('dv2WPrecip'),
    wTempHL: $('dv2WTempHL'),
    wWind: $('dv2WWind'),
    wAqi: $('dv2WAqi'),

    // thermo
    thTitle: $('dv2ThermoTitle'),
    thSubtitle: $('dv2ThermoSubtitle'),
    thFill: $('dv2ThermoFill'),
    thScore: $('dv2ThermoScore'),
    thLevel: $('dv2ThermoLevel'),
    thrCrowd: $('dv2ThrCrowd'),
    thrWeather: $('dv2ThrWeather'),

    reco: $('dv2Reco'),
    chartBox: $('dv2VisitorChart')
  };

  const state = {
    h: 7,
    includeAll: true,
    mode: 'offline',
    lang: 'zh',
    theme: 'dark',
    modelView: 'both', // both|champion|runner|third
    selectedIndex: null,
    payload: /** @type {NormalizedPayload|null} */ (null),
    chart: null,
    isZooming: false
  };

  function safeNum(x) {
    if (x === null || x === undefined) return null;
    if (typeof x === 'number' && Number.isFinite(x)) return x;
    const v = Number(x);
    return Number.isFinite(v) ? v : null;
  }

  function showSpinner(on) {
    if (!dom.spinner) return;
    dom.spinner.style.display = on ? 'grid' : 'none';
    dom.spinner.setAttribute('aria-hidden', on ? 'false' : 'true');
  }

  let alertTimer = null;
  function showAlert(msg, kind) {
    if (!dom.alert) return;
    if (!msg) {
      dom.alert.style.display = 'none';
      dom.alert.textContent = '';
      dom.alert.className = 'uiw-alert';
      return;
    }
    dom.alert.textContent = String(msg);
    dom.alert.style.display = 'block';
    dom.alert.className = 'uiw-alert' + (kind ? ` uiw-alert--${kind}` : '');
    if (alertTimer) window.clearTimeout(alertTimer);
    alertTimer = window.setTimeout(() => showAlert('', null), 4500);
  }

  function t(key) {
    const pack = I18N[state.lang] || I18N.zh;
    return pack[key] || key;
  }

  function applyI18n() {
    const nodes = document.querySelectorAll('[data-i18n]');
    nodes.forEach((el) => {
      const k = el.getAttribute('data-i18n');
      if (!k) return;
      el.textContent = t(k);
    });

    // static title/meta (not using data-i18n to keep v2 explicit)
    if (dom.title) dom.title.textContent = state.lang === 'zh' ? '九寨沟客流预测看板 v2' : 'Jiuzhaigou Forecast Dashboard v2';
  }

  function setTheme(theme) {
    state.theme = theme === 'light' ? 'light' : 'dark';
    const html = document.documentElement;
    html.setAttribute('data-bs-theme', state.theme);
    if (dom.tTheme) dom.tTheme.checked = (state.theme === 'light');
    // Re-render chart to match label colors
    if (state.payload && state.chart) renderChart(state.payload);
  }

  function setLang(lang) {
    state.lang = (lang === 'en') ? 'en' : 'zh';
    if (dom.tLang) dom.tLang.checked = (state.lang === 'en');
    applyI18n();
    if (state.payload && state.chart) renderChart(state.payload);
    // refresh selection display (holiday names, risk levels)
    if (state.payload && state.selectedIndex !== null) updateSelection(state.selectedIndex);
  }

  function setH(h) {
    const hv = Number(h);
    state.h = [1, 3, 7, 14].includes(hv) ? hv : 7;
    dom.hBtns.forEach((b) => {
      if (!b) return;
      const bh = Number(b.getAttribute('data-h'));
      b.classList.toggle('uiw-chip--active', bh === state.h);
    });
  }

  function setModelView(view) {
    state.modelView = ['both', 'champion', 'runner', 'third'].includes(view) ? view : 'both';
    dom.modelBtns.forEach((b) => {
      if (!b) return;
      const m = b.getAttribute('data-model');
      b.classList.toggle('uiw-chip--active', m === state.modelView);
    });
    if (state.payload && state.chart) renderChart(state.payload);
    if (state.payload && state.selectedIndex !== null) updateSelection(state.selectedIndex);
  }

  async function apiFetchJson(url) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
    }
    return await res.json();
  }

  /**
   * normalizeForecastPayload(raw)
   * Schema fallbacks supported:
   *  - New schema from /api/forecast: {time_axis, series{actual,champion_pred,runner_pred}, weather{...}, thresholds{...}, holidays[], risk{...}, forecast{...}, meta{...}}
   *  - Legacy schema from /api/data: {dates, true_vals, pred_vals, holiday_ranges}
   */
  function normalizeForecastPayload(raw) {
    const out = /** @type {NormalizedPayload} */ ({
      timeAxis: [],
      forecast: { h: state.h, startIndex: 0, endIndex: 0 },
      meta: {},
      series: { actual: [], champion: [], runner: [], third: [] },
      thresholds: { crowd: null, weather: {}, weatherQuantiles: null },
      weather: {
        precipMm: [], tempHighC: [], tempLowC: [], weatherCodeEn: [],
        windLevel: [], windDirEn: [], windMax: [], aqiValue: [], aqiLevelEn: []
      },
      holidays: [],
      risk: { champion: null, runner: null, third: null },
      warning: null
    });

    if (!raw || typeof raw !== 'object') return out;

    // --- New schema ---
    if (Array.isArray(raw.time_axis) || Array.isArray(raw.timeAxis)) {
      const axis = raw.time_axis || raw.timeAxis || [];
      out.timeAxis = axis.map((d) => String(d));

      const meta = raw.meta || {};
      out.meta.generatedAt = meta.generated_at || meta.generatedAt;
      out.meta.forecastMode = meta.forecast_mode || meta.forecastMode;
      out.meta.championName = (meta.champion && meta.champion.model_name) || meta.championName;
      out.meta.runnerName = (meta.runner_up && meta.runner_up.model_name) || meta.runnerName || null;
      out.meta.thirdName = (meta.third && meta.third.model_name) || meta.thirdName || null;

      const fc = raw.forecast || {};
      out.forecast.h = safeNum(fc.h) || state.h;
      out.forecast.startIndex = Math.max(0, safeNum(fc.start_index) ?? safeNum(fc.startIndex) ?? 0);
      out.forecast.endIndex = Math.max(0, safeNum(fc.end_index) ?? safeNum(fc.endIndex) ?? (out.timeAxis.length - 1));

      const s = raw.series || {};
      out.series.actual = (s.actual || s.true_vals || []).map(safeNum);
      out.series.champion = (s.champion_pred || s.pred_vals || s.pred || []).map(safeNum);
      out.series.runner = (s.runner_pred || []).map(safeNum);
      out.series.third = (s.third_pred || []).map(safeNum);

      const thr = raw.thresholds || {};
      out.thresholds.crowd = safeNum(thr.crowd);
      const wthr = thr.weather || {};
      out.thresholds.weather = {
        precipHigh: safeNum(wthr.precip_high),
        tempHigh: safeNum(wthr.temp_high),
        tempLow: safeNum(wthr.temp_low)
      };
      out.thresholds.weatherQuantiles = thr.weather_quantiles || null;

      const w = raw.weather || {};
      out.weather.precipMm = (w.precip_mm || w.precip || []).map(safeNum);
      out.weather.tempHighC = (w.temp_high_c || w.temp_high || []).map(safeNum);
      out.weather.tempLowC = (w.temp_low_c || w.temp_low || []).map(safeNum);
      out.weather.weatherCodeEn = (w.weather_code_en || []).map((x) => (x === null || x === undefined) ? null : String(x));
      out.weather.windLevel = (w.wind_level || []).map(safeNum);
      out.weather.windDirEn = (w.wind_dir_en || []).map((x) => (x === null || x === undefined) ? null : String(x));
      out.weather.windMax = (w.wind_max || []).map(safeNum);
      out.weather.aqiValue = (w.aqi_value || []).map(safeNum);
      out.weather.aqiLevelEn = (w.aqi_level_en || []).map((x) => (x === null || x === undefined) ? null : String(x));

      const hs = raw.holidays || raw.holiday_ranges || [];
      out.holidays = (Array.isArray(hs) ? hs : []).map((h) => ({
        start: String(h.start || h[0] || ''),
        end: String(h.end || h[1] || ''),
        nameZh: h.name_zh || h.nameZh || h.name || null,
        nameEn: h.name_en || h.nameEn || null,
        type: h.type || null
      })).filter((h) => h.start && h.end);

      const r = raw.risk || {};
      out.risk.champion = r.champion || null;
      out.risk.runner = r.runner_up || r.runner || null;
      out.risk.third = r.third || null;

      out.warning = raw.warning || null;

      alignAllArrays(out);
      out.forecast = clampForecast(out.forecast, out.timeAxis.length);
      return out;
    }

    // --- Legacy schema (/api/data) ---
    if (Array.isArray(raw.dates)) {
      out.timeAxis = raw.dates.map((d) => String(d));
      out.series.actual = (raw.true_vals || []).map(safeNum);
      out.series.champion = (raw.pred_vals || []).map(safeNum);
      out.series.runner = new Array(out.timeAxis.length).fill(null);
      const hs = raw.holiday_ranges || [];
      out.holidays = (Array.isArray(hs) ? hs : []).map((h) => ({
        start: String(h.start || ''),
        end: String(h.end || ''),
        nameZh: h.name || null,
        nameEn: null,
        type: h.type || null
      })).filter((h) => h.start && h.end);

      // derive forecast window from last prediction point
      const lastPred = lastNonNullIndex(out.series.champion);
      const endIdx = (lastPred >= 0) ? lastPred : (out.timeAxis.length - 1);
      out.forecast = clampForecast({ h: state.h, startIndex: Math.max(0, endIdx - state.h + 1), endIndex: endIdx }, out.timeAxis.length);

      // no weather/risk/thresholds in legacy schema
      alignAllArrays(out);
      return out;
    }

    return out;
  }

  function lastNonNullIndex(arr) {
    if (!Array.isArray(arr)) return -1;
    for (let i = arr.length - 1; i >= 0; i--) {
      if (arr[i] !== null && arr[i] !== undefined && Number.isFinite(arr[i])) return i;
    }
    return -1;
  }

  function clampForecast(forecast, n) {
    const h = Math.max(1, Math.min(14, Number(forecast.h || state.h) || state.h));
    let s = Number(forecast.startIndex || 0) || 0;
    let e = Number(forecast.endIndex || (n - 1)) || (n - 1);
    s = Math.max(0, Math.min(n - 1, s));
    e = Math.max(0, Math.min(n - 1, e));
    if (s > e) [s, e] = [e, s];
    // keep length roughly h when possible
    if (e - s + 1 > h) s = Math.max(0, e - h + 1);
    return { h, startIndex: s, endIndex: e };
  }

  function alignAllArrays(p) {
    const n = p.timeAxis.length;
    function pad(a) {
      const out = Array.isArray(a) ? a.slice(0, n) : [];
      while (out.length < n) out.push(null);
      return out;
    }
    p.series.actual = pad(p.series.actual);
    p.series.champion = pad(p.series.champion);
    p.series.runner = pad(p.series.runner);
    p.series.third = pad(p.series.third);

    p.weather.precipMm = pad(p.weather.precipMm);
    p.weather.tempHighC = pad(p.weather.tempHighC);
    p.weather.tempLowC = pad(p.weather.tempLowC);
    p.weather.weatherCodeEn = pad(p.weather.weatherCodeEn);
    p.weather.windLevel = pad(p.weather.windLevel);
    p.weather.windDirEn = pad(p.weather.windDirEn);
    p.weather.windMax = pad(p.weather.windMax);
    p.weather.aqiValue = pad(p.weather.aqiValue);
    p.weather.aqiLevelEn = pad(p.weather.aqiLevelEn);
  }

  function buildHolidayMarkAreas(payload) {
    const items = [];
    (payload.holidays || []).forEach((h) => {
      const name = (state.lang === 'zh') ? (h.nameZh || h.nameEn || 'Holiday') : (h.nameEn || h.nameZh || 'Holiday');
      items.push([
        { name, xAxis: h.start },
        { xAxis: h.end }
      ]);
    });
    return items;
  }

  function fmtInt(x) {
    const v = safeNum(x);
    if (v === null) return '--';
    try {
      return Math.round(v).toLocaleString();
    } catch {
      return String(Math.round(v));
    }
  }

  function fmt1(x) {
    const v = safeNum(x);
    if (v === null) return '--';
    return (Math.round(v * 10) / 10).toFixed(1);
  }

  function riskLevelText(lv) {
    const v = Number(lv);
    if (!Number.isFinite(v)) return '--';
    if (v <= 0) return t('risk_lv_0');
    if (v === 1) return t('risk_lv_1');
    if (v === 2) return t('risk_lv_2');
    return t('risk_lv_3');
  }

  function pickActiveRisk(payload) {
    if (!payload || !payload.risk) return null;
    if (state.modelView === 'runner' && payload.risk.runner) return payload.risk.runner;
    if (state.modelView === 'third' && payload.risk.third) return payload.risk.third;
    return payload.risk.champion || payload.risk.runner || payload.risk.third || null;
  }

  function updateMetaLine(payload) {
    if (!dom.meta) return;
    const modeLabel = (state.mode === 'online') ? t('online_mode') : t('offline_mode');
    const gen = payload.meta.generatedAt ? ` • ${payload.meta.generatedAt}` : '';
    const fm = payload.meta.forecastMode ? ` • ${payload.meta.forecastMode}` : '';
    dom.meta.textContent = `${modeLabel}${gen}${fm}`;
  }

  function renderChart(payload) {
    console.log('renderChart() called');
    if (!payload || !dom.chartBox || !window.echarts) return;

    const textColor = getComputedStyle(document.documentElement).getPropertyValue('--uiw-text').trim() || 'rgba(255,255,255,0.92)';
    const mutedColor = getComputedStyle(document.documentElement).getPropertyValue('--uiw-muted').trim() || 'rgba(255,255,255,0.65)';

    if (!state.chart) {
      state.chart = echarts.init(dom.chartBox, null, { renderer: 'canvas' });
      window.addEventListener('resize', () => {
        try { state.chart && state.chart.resize(); } catch { /* ignore */ }
      });

      // smoother zooming: hide tooltip when dataZoom is active
      state.chart.on('dataZoom', () => {
        state.isZooming = true;
        try { state.chart.dispatchAction({ type: 'hideTip' }); } catch { /* ignore */ }
        window.setTimeout(() => { state.isZooming = false; }, 140);
      });

      state.chart.on('click', (params) => {
        if (!params) return;
        const idx = params.dataIndex;
        if (typeof idx !== 'number') return;
        updateSelection(idx);
      });
    }

    const markAreaData = buildHolidayMarkAreas(payload);

    const showActual = true;
    const showChampion = (state.modelView === 'both' || state.modelView === 'champion');
    const showRunner = (state.modelView === 'both' || state.modelView === 'runner');
    const showThird = (state.modelView === 'both' || state.modelView === 'third');

    const crowdThr = safeNum(payload.thresholds.crowd);

    const option = {
      animation: false,
      grid: { left: 46, right: 18, top: 26, bottom: 52 },
      xAxis: {
        type: 'category',
        data: payload.timeAxis,
        boundaryGap: false,
        axisLabel: { color: mutedColor, formatter: (v) => String(v).slice(5) },
        axisLine: { lineStyle: { color: 'rgba(255,255,255,0.16)' } },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: mutedColor },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.10)' } }
      },
      dataZoom: [
        {
          type: 'inside',
          throttle: 60,
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
          moveOnMouseWheel: true,
          preventDefaultMouseMove: false
        },
        {
          type: 'slider',
          height: 26,
          bottom: 10,
          borderColor: 'rgba(255,255,255,0.12)',
          backgroundColor: 'rgba(255,255,255,0.04)',
          fillerColor: 'rgba(124,92,255,0.18)',
          textStyle: { color: mutedColor }
        }
      ],
      tooltip: {
        trigger: 'axis',
        triggerOn: 'mousemove|click',
        confine: true,
        axisPointer: {
          type: 'line',
          snap: true,
          animation: false,
          lineStyle: { color: 'rgba(124,92,255,0.65)', width: 1 }
        },
        formatter: (items) => {
          if (state.isZooming) return '';
          const it = Array.isArray(items) ? items : [items];
          const first = it[0];
          const idx = first ? first.dataIndex : null;
          const date = (idx !== null && payload.timeAxis[idx]) ? payload.timeAxis[idx] : '';
          const lines = [`<div style="font-weight:700;margin-bottom:6px;">${date}</div>`];
          it.forEach((x) => {
            const name = x.seriesName;
            const v = (x.data === null || x.data === undefined) ? null : Number(x.data);
            const val = (v === null || !Number.isFinite(v)) ? '--' : fmtInt(v);
            lines.push(`<div style="display:flex;gap:10px;align-items:center;justify-content:space-between;">
              <span>${x.marker}${name}</span>
              <span style="font-family:var(--uiw-font-mono);">${val}</span>
            </div>`);
          });
          lines.push(`<div style="margin-top:6px;color:rgba(255,255,255,0.70);font-size:12px;">Passengers</div>`);
          return lines.join('');
        }
      },
      series: [
        {
          name: state.lang === 'zh' ? '真实' : 'Actual',
          type: 'line',
          data: showActual ? payload.series.actual : payload.series.actual.map(() => null),
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 2, color: 'rgba(255,255,255,0.75)' },
          itemStyle: { color: 'rgba(255,255,255,0.75)' }
        },
        {
          name: state.lang === 'zh' ? '冠军预测' : 'Champion',
          type: 'line',
          data: showChampion ? payload.series.champion : payload.series.champion.map(() => null),
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 2, color: 'rgba(124,92,255,0.92)' },
          itemStyle: { color: 'rgba(124,92,255,0.92)' },
          markArea: markAreaData.length ? {
            silent: true,
            itemStyle: { color: 'rgba(255, 215, 64, 0.10)' },
            data: markAreaData
          } : undefined,
          markLine: (crowdThr !== null) ? {
            silent: true,
            lineStyle: { color: 'rgba(255,77,79,0.55)', type: 'dashed' },
            label: { color: mutedColor, formatter: state.lang === 'zh' ? '阈值' : 'Threshold' },
            data: [{ yAxis: crowdThr }]
          } : undefined
        },
        {
          name: state.lang === 'zh' ? '亚军预测' : 'Runner-up',
          type: 'line',
          data: showRunner ? payload.series.runner : payload.series.runner.map(() => null),
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 2, color: 'rgba(34,197,94,0.88)' },
          itemStyle: { color: 'rgba(34,197,94,0.88)' }
        },
        {
          name: state.lang === 'zh' ? '第三预测' : 'Third',
          type: 'line',
          data: showThird ? payload.series.third : payload.series.third.map(() => null),
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 2, color: 'rgba(251,146,60,0.88)' },
          itemStyle: { color: 'rgba(251,146,60,0.88)' }
        }
      ]
    };

    try {
      state.chart.setOption(option, true);
    } catch (e) {
      console.error(e);
      showAlert('Chart render failed', 'error');
    }

    // default zoom to forecast window
    const n = payload.timeAxis.length;
    const s = payload.forecast.startIndex;
    const e = payload.forecast.endIndex;
    if (n > 0 && s >= 0 && e >= s) {
      const startPct = (s / Math.max(1, n - 1)) * 100;
      const endPct = (e / Math.max(1, n - 1)) * 100;
      try {
        state.chart.dispatchAction({ type: 'dataZoom', start: Math.max(0, startPct - 3), end: Math.min(100, endPct + 3) });
      } catch {
        // ignore
      }
    }
  }

  function updateSelection(idx) {
    const payload = state.payload;
    if (!payload) return;
    const n = payload.timeAxis.length;
    if (idx === null || idx === undefined) return;
    const i = Math.max(0, Math.min(n - 1, Number(idx)));
    state.selectedIndex = i;

    const date = payload.timeAxis[i] || t('weather_no_date');

    // Weather
    if (dom.wDate) dom.wDate.textContent = date;

    const ph = safeNum(payload.weather.tempHighC[i]);
    const pl = safeNum(payload.weather.tempLowC[i]);
    const avg = (ph !== null && pl !== null) ? (ph + pl) / 2 : (ph !== null ? ph : (pl !== null ? pl : null));
    if (dom.wTemp) dom.wTemp.textContent = (avg === null) ? '--°C' : `${fmt1(avg)}°C`;

    const precip = safeNum(payload.weather.precipMm[i]);
    if (dom.wPrecip) dom.wPrecip.textContent = (precip === null) ? '--' : `${fmt1(precip)} mm`;

    if (dom.wTempHL) {
      const hi = (ph === null) ? '--' : `${fmt1(ph)}°C`;
      const lo = (pl === null) ? '--' : `${fmt1(pl)}°C`;
      dom.wTempHL.textContent = `${hi} / ${lo}`;
    }

    const wl = safeNum(payload.weather.windLevel[i]);
    const wd = payload.weather.windDirEn[i];
    const wm = safeNum(payload.weather.windMax[i]);
    if (dom.wWind) {
      const parts = [];
      if (wl !== null) parts.push(state.lang === 'zh' ? `等级 ${fmt1(wl)}` : `Lv ${fmt1(wl)}`);
      if (wd) parts.push(String(wd));
      if (wm !== null) parts.push(`${fmt1(wm)} m/s`);
      dom.wWind.textContent = parts.length ? parts.join(' · ') : '--';
    }

    const aqi = safeNum(payload.weather.aqiValue[i]);
    const aqiLv = payload.weather.aqiLevelEn[i];
    if (dom.wAqi) {
      const parts = [];
      if (aqi !== null) parts.push(String(Math.round(aqi)));
      if (aqiLv) parts.push(String(aqiLv));
      dom.wAqi.textContent = parts.length ? parts.join(' · ') : '--';
    }

    // flags
    const flags = [];
    const holidayName = holidayHit(payload, date);
    if (holidayName) flags.push({ kind: 'holiday', text: holidayName });

    const activeRisk = pickActiveRisk(payload);
    const drivers = (activeRisk && Array.isArray(activeRisk.drivers) && activeRisk.drivers[i]) ? activeRisk.drivers[i] : [];
    if (Array.isArray(drivers) && drivers.length) {
      drivers.forEach((d) => flags.push({ kind: 'driver', text: translateDriver(d) }));
    }

    if (dom.wFlags) {
      dom.wFlags.innerHTML = '';
      if (flags.length) {
        dom.wFlags.style.display = 'flex';
        flags.forEach((f) => {
          const el = document.createElement('span');
          el.className = 'uiw-weather-flag';
          el.textContent = f.text;
          dom.wFlags.appendChild(el);
        });
      } else {
        dom.wFlags.style.display = 'none';
      }
    }

    if (dom.wMeta) {
      dom.wMeta.textContent = flags.length ? (state.lang === 'zh' ? '已同步天气与风险标记。' : 'Synced weather and risk flags.') : t('weather_click_tip');
    }

    // Thermo / Risk
    if (dom.thTitle) dom.thTitle.textContent = date;

    const crowdThr = safeNum(payload.thresholds.crowd);
    if (dom.thrCrowd) dom.thrCrowd.textContent = crowdThr === null ? '--' : fmtInt(crowdThr);

    const wthr = payload.thresholds.weather || {};
    const q = payload.thresholds.weatherQuantiles || {};
    // show precip quantile if available; otherwise show raw thresholds
    const qLine = [];
    if (q && typeof q === 'object') {
      if (q.precip_high !== undefined && q.precip_high !== null) qLine.push(`P≥Q${q.precip_high}`);
      if (q.temp_high !== undefined && q.temp_high !== null) qLine.push(`Th≥Q${q.temp_high}`);
      if (q.temp_low !== undefined && q.temp_low !== null) qLine.push(`Tl≤Q${q.temp_low}`);
    }
    if (!qLine.length) {
      if (wthr.precipHigh !== null && wthr.precipHigh !== undefined) qLine.push(`P≥${fmt1(wthr.precipHigh)}mm`);
      if (wthr.tempHigh !== null && wthr.tempHigh !== undefined) qLine.push(`Th≥${fmt1(wthr.tempHigh)}°C`);
      if (wthr.tempLow !== null && wthr.tempLow !== undefined) qLine.push(`Tl≤${fmt1(wthr.tempLow)}°C`);
    }
    if (dom.thrWeather) dom.thrWeather.textContent = qLine.length ? qLine.join(' · ') : '--';

    const risk = activeRisk;
    const score = risk && Array.isArray(risk.risk_score) ? safeNum(risk.risk_score[i]) : null;
    const lv = risk && Array.isArray(risk.risk_level) ? Number(risk.risk_level[i]) : null;

    if (dom.thScore) dom.thScore.textContent = (score === null) ? '--' : `${fmt1(score)}`;
    if (dom.thLevel) dom.thLevel.textContent = riskLevelText(lv);

    if (dom.thFill) {
      const pct = (score === null) ? 12 : Math.max(0, Math.min(100, score));
      dom.thFill.style.height = `${pct}%`;
    }

    if (dom.thSubtitle) {
      dom.thSubtitle.textContent = (score === null && lv === null) ? t('thermo_default') : (state.lang === 'zh' ? '已同步到所选日期。' : 'Synced to selected day.');
    }

    // Reco (simple best-window suggestion within forecast segment)
    updateReco(payload);
  }

  function translateDriver(code) {
    const mapZh = {
      crowd_over_threshold: '客流超阈值',
      precip_high: '降水偏高',
      temp_high: '高温',
      temp_low: '低温'
    };
    const mapEn = {
      crowd_over_threshold: 'Crowd over threshold',
      precip_high: 'High precipitation',
      temp_high: 'High temperature',
      temp_low: 'Low temperature'
    };
    const m = (state.lang === 'zh') ? mapZh : mapEn;
    return m[code] || String(code);
  }

  function holidayHit(payload, dateStr) {
    if (!payload || !payload.holidays || !dateStr) return null;
    const d = String(dateStr);
    for (const h of payload.holidays) {
      if (h.start <= d && d <= h.end) {
        return (state.lang === 'zh') ? (h.nameZh || h.nameEn || '节假日') : (h.nameEn || h.nameZh || 'Holiday');
      }
    }
    return null;
  }

  function updateReco(payload) {
    if (!dom.reco) return;
    if (!payload || !payload.timeAxis.length) {
      dom.reco.textContent = t('best_wait');
      return;
    }

    const fc = payload.forecast;
    const s = fc.startIndex;
    const e = fc.endIndex;

    const risk = pickActiveRisk(payload);
    const lvArr = risk && Array.isArray(risk.risk_level) ? risk.risk_level : null;

    const rows = [];
    for (let i = s; i <= e; i++) {
      const pred = payload.series.champion[i];
      const lv = lvArr ? Number(lvArr[i]) : null;
      rows.push({ i, date: payload.timeAxis[i], pred: safeNum(pred), lv: Number.isFinite(lv) ? lv : null });
    }

    // prioritize low risk then low forecast
    rows.sort((a, b) => {
      const alv = a.lv === null ? 9 : a.lv;
      const blv = b.lv === null ? 9 : b.lv;
      if (alv !== blv) return alv - blv;
      const ap = a.pred === null ? Number.POSITIVE_INFINITY : a.pred;
      const bp = b.pred === null ? Number.POSITIVE_INFINITY : b.pred;
      return ap - bp;
    });

    const best = rows.slice(0, 3).filter((r) => r.date);
    if (!best.length) {
      dom.reco.textContent = t('best_wait');
      return;
    }

    const lines = best.map((r) => {
      const lvTxt = riskLevelText(r.lv);
      const pv = r.pred === null ? '--' : fmtInt(r.pred);
      return `${r.date}: ${pv} (${lvTxt})`;
    });

    dom.reco.textContent = (state.lang === 'zh')
      ? `推荐（视窗内）：\n${lines.join('\n')}`
      : `Recommended (in view):\n${lines.join('\n')}`;
  }

  async function loadAndRender() {
    showSpinner(true);
    showAlert('', null);

    const qs = new URLSearchParams();
    qs.set('h', String(state.h));
    qs.set('include_all', state.includeAll ? '1' : '0');
    qs.set('mode', state.mode);

    try {
      const raw = await apiFetchJson(`/api/forecast?${qs.toString()}`);
      const payload = normalizeForecastPayload(raw);
      state.payload = payload;

      updateMetaLine(payload);
      renderChart(payload);

      if (payload.warning) {
        showAlert(payload.warning, 'warn');
      } else if (state.mode === 'online' && payload.meta && payload.meta.forecastMode && String(payload.meta.forecastMode).includes('offline')) {
        showAlert(t('warn_fallback'), 'warn');
      }

      // auto select end of forecast window (most relevant)
      const autoIdx = payload.forecast.endIndex;
      updateSelection(autoIdx);

    } catch (e) {
      console.error(e);
      showAlert(String(e && e.message ? e.message : e), 'error');
      // fallback: try legacy /api/data if forecast is unavailable
      try {
        const raw2 = await apiFetchJson('/api/data');
        const payload2 = normalizeForecastPayload(raw2);
        state.payload = payload2;
        updateMetaLine(payload2);
        renderChart(payload2);
        updateSelection(payload2.forecast.endIndex);
        showAlert(state.lang === 'zh' ? '已回退到旧数据接口 /api/data。' : 'Fell back to legacy /api/data endpoint.', 'warn');
      } catch (e2) {
        console.error(e2);
      }
    } finally {
      showSpinner(false);
    }
  }

  function bindEvents() {
    dom.hBtns.forEach((b) => {
      if (!b) return;
      b.addEventListener('click', () => {
        setH(b.getAttribute('data-h'));
        loadAndRender().catch(() => {});
      });
    });

    dom.modelBtns.forEach((b) => {
      if (!b) return;
      b.addEventListener('click', () => {
        setModelView(b.getAttribute('data-model'));
      });
    });

    if (dom.tTheme) {
      dom.tTheme.addEventListener('change', () => {
        setTheme(dom.tTheme.checked ? 'light' : 'dark');
      });
    }

    if (dom.tLang) {
      dom.tLang.addEventListener('change', () => {
        setLang(dom.tLang.checked ? 'en' : 'zh');
      });
    }

    if (dom.tOnline) {
      dom.tOnline.addEventListener('change', () => {
        state.mode = dom.tOnline.checked ? 'online' : 'offline';
        loadAndRender().catch(() => {});
      });
    }

    if (dom.btnRefresh) dom.btnRefresh.addEventListener('click', () => loadAndRender().catch(() => {}));

    if (dom.btnReset) {
      dom.btnReset.addEventListener('click', () => {
        setH(7);
        setModelView('both');
        setTheme('dark');
        setLang('zh');
        state.mode = 'offline';
        if (dom.tOnline) dom.tOnline.checked = false;
        state.selectedIndex = null;
        showAlert('', null);
        loadAndRender().catch(() => {});
      });
    }
  }

  function init() {
    console.log('init() called');
    // defaults
    setH(state.h);
    setTheme(state.theme);
    setLang(state.lang);
    setModelView(state.modelView);
    applyI18n();
    bindEvents();
    loadAndRender().catch(() => {});
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
