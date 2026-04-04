/* dashboard_v3.js — Jiuzhaigou Visitor Forecast Dashboard v3
 * Vanilla JS, no imports, no frameworks. ECharts must be loaded before this file.
 */
(() => {
  'use strict';

  // ─────────────────────────────────────────────
  // i18n
  // ─────────────────────────────────────────────
  const I18N = {
    zh: {
      tab_forecast: '预测', tab_analysis: '分析', tab_models: '模型',
      online: '在线', kpi_latest: '最新预测', kpi_peak: '预测峰值',
      kpi_risk: '综合风险', kpi_model: '冠军模型', kpi_clock: '当前时间', kpi_unit: '人',
      chart_title: '客流预测', model_all: '全部', model_champ: '冠军',
      model_runner: '亚军', model_third: '第三', risk_title: '适宜性预警',
      risk_normal: '正常', risk_watch: '关注', risk_warning: '预警',
      risk_high: '高风险', reco_title: '推荐出行窗口', reco_loading: '加载中…',
      w_precip: '降水', w_temphl: '温度区间', w_wind: '风力', w_aqi: 'AQI',
      analysis_title: '模型分析',
      analysis_sub: 'Walk-forward 评估 · 特征消融 · 校准诊断',
      wf_title: 'Walk-forward 评估（4折）',
      wf_sub: 'Expanding Window · 每折测试90天',
      calib_title: '校准可靠性图',
      calib_note: '注：当前预警概率为确定性分类器近似（temperature=1000），非真正概率输出。',
      cmp_title: '三模型指标对比', models_title: '模型详情',
      models_sub: '架构 · 特征 · 训练参数 · Attention 热力图',
      status_loading: '加载中',
      status_source: '数据来源：九寨沟官网 · Open-Meteo',
      offline_mode: '离线回测', online_mode: '在线预测',
      risk_lv_0: '正常', risk_lv_1: '关注', risk_lv_2: '预警', risk_lv_3: '高风险',
      driver_crowd_over_threshold: '客流超阈值', driver_precip_high: '强降水',
      driver_temp_high: '高温', driver_temp_low: '低温',
      no_data: '暂无数据', fold: '折', mae: 'MAE', f1: 'F1', rmse: 'RMSE',
      smape: 'sMAPE', recall: '召回率', precision: '精确率', brier: 'Brier',
      ece: 'ECE', epochs: '训练轮数', look_back: '回看窗口',
      architecture: '架构', model_name: '模型名称',
      metrics_regression: '回归指标', metrics_crowd: '客流预警',
      metrics_suit: '适宜性预警', metrics_meta: '训练参数',
      perfect_calib: '完美校准', actual_calib: '实际校准',
      confidence: '置信度', accuracy: '准确率',
      warn_fallback: '在线预测失败，已自动回退离线产物。',
      h_1d: '1天', h_7d: '7天', h_14d: '14天',
      h_offline: '离线回测',
      btn_reset: '复原',
      show_precip: '降水', show_temp: '温度',
      weather_sub_title: '降水 / 温度',
      curve_actual: '实际', curve_champ: '冠军',
      curve_runner: '亚军', curve_third: '第三',
      wf_explain: 'Walk-forward 评估使用扩展窗口策略，每折独立训练，覆盖不同季节。MAE 越低表示回归精度越高；F1 越高表示预警准确率越高。',
      calib_explain: '可靠性图展示模型预警概率的校准质量。理想情况下，置信度为 X 时实际准确率也应为 X（对角线）。当前模型使用确定性近似，概率集中在 0 和 1 附近属正常现象。',
      reco_reason_low: '预测客流较少，适合出游',
      reco_reason_normal: '客流正常，天气适宜',
      reco_reason_watch: '客流偏多，建议错峰'
    },
    en: {
      tab_forecast: 'Forecast', tab_analysis: 'Analysis', tab_models: 'Models',
      online: 'Online', kpi_latest: 'Latest Forecast', kpi_peak: 'Forecast Peak',
      kpi_risk: 'Risk Level', kpi_model: 'Champion', kpi_clock: 'Current Time', kpi_unit: 'visitors',
      chart_title: 'Visitor Forecast', model_all: 'All', model_champ: 'Champion',
      model_runner: 'Runner', model_third: 'Third', risk_title: 'Suitability Warning',
      risk_normal: 'Normal', risk_watch: 'Watch', risk_warning: 'Warning',
      risk_high: 'High Risk', reco_title: 'Best Visit Window', reco_loading: 'Loading…',
      w_precip: 'Precip', w_temphl: 'Temp Range', w_wind: 'Wind', w_aqi: 'AQI',
      analysis_title: 'Model Analysis',
      analysis_sub: 'Walk-forward · Ablation · Calibration',
      wf_title: 'Walk-forward Evaluation (4 folds)',
      wf_sub: 'Expanding Window · 90-day test per fold',
      calib_title: 'Reliability Diagram',
      calib_note: 'Note: Warning probability is a deterministic classifier approximation (temperature=1000), not a true probabilistic output.',
      cmp_title: 'Three-Model Comparison', models_title: 'Model Details',
      models_sub: 'Architecture · Features · Training · Attention Heatmap',
      status_loading: 'Loading',
      status_source: 'Source: Jiuzhaigou Official · Open-Meteo',
      offline_mode: 'Offline backtest', online_mode: 'Online forecast',
      risk_lv_0: 'Normal', risk_lv_1: 'Watch', risk_lv_2: 'Warning', risk_lv_3: 'High Risk',
      driver_crowd_over_threshold: 'Crowd over threshold',
      driver_precip_high: 'Heavy precipitation',
      driver_temp_high: 'High temperature', driver_temp_low: 'Low temperature',
      no_data: 'No data', fold: 'Fold', mae: 'MAE', f1: 'F1', rmse: 'RMSE',
      smape: 'sMAPE', recall: 'Recall', precision: 'Precision', brier: 'Brier',
      ece: 'ECE', epochs: 'Epochs', look_back: 'Look-back',
      architecture: 'Architecture', model_name: 'Model Name',
      metrics_regression: 'Regression', metrics_crowd: 'Crowd Alert',
      metrics_suit: 'Suitability', metrics_meta: 'Training Params',
      perfect_calib: 'Perfect Calibration', actual_calib: 'Actual',
      confidence: 'Confidence', accuracy: 'Accuracy',
      warn_fallback: 'Online forecast failed; falling back to offline artifacts.',
      h_1d: '1 Day', h_7d: '7 Days', h_14d: '14 Days',
      h_offline: 'Offline',
      btn_reset: 'Reset View',
      show_precip: 'Precip', show_temp: 'Temp',
      weather_sub_title: 'Precip / Temp',
      curve_actual: 'Actual', curve_champ: 'Champion',
      curve_runner: 'Runner', curve_third: 'Third',
      wf_explain: 'Walk-forward uses expanding window strategy. Each fold trains independently covering different seasons. Lower MAE = better regression; Higher F1 = better warning accuracy.',
      calib_explain: 'Reliability diagram shows calibration quality. Ideally confidence X should match accuracy X (diagonal). Current model uses deterministic approximation; probability concentration at 0 and 1 is expected.',
      reco_reason_low: 'Low predicted crowd, good for visiting',
      reco_reason_normal: 'Normal crowd, suitable weather',
      reco_reason_watch: 'Higher crowd, consider off-peak timing'
    }
  };

  // ─────────────────────────────────────────────
  // State
  // ─────────────────────────────────────────────
  const state = {
    h: 7,
    mode: 'offline',
    lang: 'zh',
    theme: 'dark',
    modelView: 'both',
    selectedIdx: null,
    payload: null,
    chart: null,
    weatherChart: null,
    wfChart: null,
    calibChart: null,
    metricsCache: {},
    analysisLoaded: false,
    modelsLoaded: false,
    curveVisible: { actual: true, champion: true, runner: true, third: true }
  };

  // ─────────────────────────────────────────────
  // DOM helpers
  // ─────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  function t(key) {
    const pack = I18N[state.lang] || I18N.zh;
    return pack[key] !== undefined ? pack[key] : key;
  }

  function applyI18n() {
    $$('[data-i18n]').forEach((el) => {
      const k = el.getAttribute('data-i18n');
      if (k) el.textContent = t(k);
    });
  }

  // ─────────────────────────────────────────────
  // Spinner / Error
  // ─────────────────────────────────────────────
  function showSpinner(on) {
    const el = $('v3Spinner');
    if (!el) return;
    el.classList.toggle('v3-spinner-overlay--visible', !!on);
    el.setAttribute('aria-hidden', on ? 'false' : 'true');
  }

  function showError(msg) {
    const banner = $('v3Error');
    const msgEl = $('v3ErrorMsg');
    if (!banner) return;
    if (!msg) { banner.style.display = 'none'; return; }
    if (msgEl) msgEl.textContent = msg;
    banner.style.display = 'flex';
  }

  // ─────────────────────────────────────────────
  // Status bar
  // ─────────────────────────────────────────────
  function setStatus(kind, text, genAt) {
    const dot = $('v3StatusDot');
    const txt = $('v3StatusText');
    const gen = $('v3GenAt');
    if (dot) {
      dot.className = 'v3-status-dot';
      if (kind === 'ok') dot.classList.add('v3-status-dot--ok');
      else if (kind === 'warn') dot.classList.add('v3-status-dot--warn');
      else if (kind === 'err') dot.classList.add('v3-status-dot--err');
    }
    if (txt) txt.textContent = text || '';
    if (gen && genAt) $('v3GenAt').textContent = genAt;
  }

  // ─────────────────────────────────────────────
  // Number helpers
  // ─────────────────────────────────────────────
  function safeNum(x) {
    if (x === null || x === undefined) return null;
    const v = typeof x === 'number' ? x : Number(x);
    return Number.isFinite(v) ? v : null;
  }

  function fmtVisitors(x) {
    const v = safeNum(x);
    if (v === null) return '—';
    try { return Math.round(v).toLocaleString(state.lang === 'zh' ? 'zh-CN' : 'en-US'); }
    catch { return String(Math.round(v)); }
  }

  function fmtPct(x) {
    const v = safeNum(x);
    return v === null ? '—' : v.toFixed(1) + '%';
  }

  function fmtRound(x) {
    const v = safeNum(x);
    if (v === null) return '—';
    try { return Math.round(v).toLocaleString(); }
    catch { return String(Math.round(v)); }
  }

  function fmtDec(x, d) {
    const v = safeNum(x);
    return v === null ? '—' : v.toFixed(d !== undefined ? d : 2);
  }

  // ─────────────────────────────────────────────
  // Payload normalization
  // ─────────────────────────────────────────────
  function safeArr(a, n, mapFn) {
    const arr = Array.isArray(a) ? a.map(mapFn) : [];
    while (arr.length < n) arr.push(null);
    return arr.slice(0, n);
  }

  function normalizeForecastPayload(raw) {
    const out = {
      timeAxis: [], forecast: { h: state.h, startIndex: 0, endIndex: 0 },
      meta: {}, series: { actual: [], champion: [], runner: [], third: [] },
      thresholds: { crowd: null, weather: {} },
      weather: { precipMm: [], tempHighC: [], tempLowC: [], weatherCodeEn: [],
        windLevel: [], windDirEn: [], windMax: [], aqiValue: [], aqiLevelEn: [] },
      holidays: [], risk: { champion: null, runner: null, third: null },
      warning: null
    };
    if (!raw || typeof raw !== 'object') return out;

    const axis = raw.time_axis || raw.timeAxis || [];
    out.timeAxis = Array.isArray(axis) ? axis.map(String) : [];
    const n = out.timeAxis.length;

    const meta = raw.meta || {};
    out.meta.generatedAt = meta.generated_at || meta.generatedAt || null;
    out.meta.forecastMode = meta.forecast_mode || meta.forecastMode || null;
    out.meta.championName = (meta.champion && meta.champion.model_name) || meta.championName || null;
    out.meta.runnerName = (meta.runner_up && meta.runner_up.model_name) || meta.runnerName || null;
    out.meta.thirdName = (meta.third && meta.third.model_name) || meta.thirdName || null;

    const fc = raw.forecast || {};
    out.forecast.h = safeNum(fc.h) || state.h;
    out.forecast.startIndex = Math.max(0, safeNum(fc.start_index) ?? safeNum(fc.startIndex) ?? 0);
    out.forecast.endIndex = Math.max(0, safeNum(fc.end_index) ?? safeNum(fc.endIndex) ?? Math.max(0, n - 1));

    const s = raw.series || {};
    out.series.actual = safeArr(s.actual || [], n, safeNum);
    out.series.champion = safeArr(s.champion_pred || s.pred_vals || [], n, safeNum);
    out.series.runner = safeArr(s.runner_pred || [], n, safeNum);
    out.series.third = safeArr(s.third_pred || [], n, safeNum);

    const thr = raw.thresholds || {};
    out.thresholds.crowd = safeNum(thr.crowd);
    const wt = thr.weather || {};
    out.thresholds.weather = {
      precipHigh: safeNum(wt.precip_high), tempHigh: safeNum(wt.temp_high), tempLow: safeNum(wt.temp_low)
    };

    const w = raw.weather || {};
    out.weather.precipMm = safeArr(w.precip_mm || [], n, safeNum);
    out.weather.tempHighC = safeArr(w.temp_high_c || [], n, safeNum);
    out.weather.tempLowC = safeArr(w.temp_low_c || [], n, safeNum);
    out.weather.weatherCodeEn = safeArr(w.weather_code_en || [], n, (x) => x == null ? null : String(x));
    out.weather.windLevel = safeArr(w.wind_level || [], n, safeNum);
    out.weather.windDirEn = safeArr(w.wind_dir_en || [], n, (x) => x == null ? null : String(x));
    out.weather.windMax = safeArr(w.wind_max || [], n, safeNum);
    out.weather.aqiValue = safeArr(w.aqi_value || [], n, safeNum);
    out.weather.aqiLevelEn = safeArr(w.aqi_level_en || [], n, (x) => x == null ? null : String(x));

    const hs = raw.holidays || [];
    out.holidays = (Array.isArray(hs) ? hs : []).map((h) => ({
      start: String(h.start || ''), end: String(h.end || ''),
      nameZh: h.name_zh || h.nameZh || null,
      nameEn: h.name_en || h.nameEn || null,
      type: h.type || null
    })).filter((h) => h.start && h.end);

    const r = raw.risk || {};
    out.risk.champion = r.champion || null;
    out.risk.runner = r.runner_up || r.runner || null;
    out.risk.third = r.third || null;
    out.warning = raw.warning || null;
    return out;
  }

  // ─────────────────────────────────────────────
  // API fetch
  // ─────────────────────────────────────────────
  async function apiFetch(url) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) {
      const txt = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status}: ${txt || res.statusText}`);
    }
    return res.json();
  }

  async function loadForecast() {
    showSpinner(true);
    showError(null);
    setStatus('', t('status_loading'));

    const cacheKey = `v3_forecast_${state.mode}_h${state.h}`;

    // Check cache for online mode (cache for 30 minutes)
    if (state.mode === 'online') {
      try {
        const cached = localStorage.getItem(cacheKey);
        if (cached) {
          const { data, ts } = JSON.parse(cached);
          if (Date.now() - ts < 30 * 60 * 1000) {
            const normalized = normalizeForecastPayload(data);
            // 缓存验证：timeAxis 必须非空才使用缓存
            if (normalized.timeAxis && normalized.timeAxis.length > 0) {
              state.payload = normalized;
              renderAll(state.payload);
              showSpinner(false);
              setStatus('ok', t('online_mode') + ' (cached)', state.payload.meta.generatedAt || '');
              return;
            }
            // 缓存无效，删除并重新请求
            localStorage.removeItem(cacheKey);
          }
        }
      } catch (e) {
        try { localStorage.removeItem(cacheKey); } catch (_) {}
      }
    }

    try {
      const url = `/api/forecast?h=${state.h}&include_all=1&mode=${state.mode}`;
      const raw = await apiFetch(url);

      // Cache online results
      if (state.mode === 'online') {
        try {
          localStorage.setItem(cacheKey, JSON.stringify({ data: raw, ts: Date.now() }));
        } catch (e) {}
      }

      state.payload = normalizeForecastPayload(raw);
      if (state.payload.warning) showError(state.payload.warning);
      const modeLabel = state.mode === 'online' ? t('online_mode') : t('offline_mode');
      setStatus('ok', modeLabel, state.payload.meta.generatedAt || '');
      renderAll(state.payload);
    } catch (err) {
      console.error('loadForecast error:', err);
      showError(err.message || 'Failed to load forecast');
      setStatus('err', err.message || 'Error');
    } finally {
      showSpinner(false);
    }
  }

  async function loadMetrics(modelId) {
    if (state.metricsCache[modelId]) return state.metricsCache[modelId];
    try {
      const data = await apiFetch(`/api/metrics?model_id=${modelId}`);
      state.metricsCache[modelId] = data;
      return data;
    } catch (err) {
      console.warn(`metrics fetch failed for ${modelId}:`, err);
      return null;
    }
  }

  // ─────────────────────────────────────────────
  // Weather icon (FIX 6: proper nested structure)
  // ─────────────────────────────────────────────
  function weatherIconHtml(code) {
    if (!code) return '<div class="v3-wi"><div class="v3-wi__cloud"></div></div>';
    const c = code.toUpperCase();
    if (c.includes('SUNNY') || c.includes('CLEAR')) {
      return '<div class="v3-wi"><div class="v3-wi__sun"></div></div>';
    }
    if (c.includes('SNOW')) {
      return '<div class="v3-wi"><div class="v3-wi__cloud"></div><span class="v3-wi__snow" style="font-size:18px;position:absolute;bottom:0;left:14px">❄</span></div>';
    }
    if (c.includes('RAIN') || c.includes('DRIZZLE')) {
      return '<div class="v3-wi"><div class="v3-wi__cloud"></div><div class="v3-wi__rain"><span class="v3-wi__drop"></span><span class="v3-wi__drop"></span><span class="v3-wi__drop"></span></div></div>';
    }
    return '<div class="v3-wi"><div class="v3-wi__cloud"></div></div>';
  }

  // ─────────────────────────────────────────────
  // Risk helpers
  // ─────────────────────────────────────────────
  const RISK_KEYS = ['risk_normal', 'risk_watch', 'risk_warning', 'risk_high'];
  const RISK_BADGE_CLASSES = ['', 'v3-risk-badge--warn', 'v3-risk-badge--warn', 'v3-risk-badge--high'];

  function riskLevelText(lv) {
    const v = Math.max(0, Math.min(3, Math.round(safeNum(lv) || 0)));
    return t(RISK_KEYS[v] || 'risk_normal');
  }

  function riskBadgeClass(lv) {
    const v = Math.max(0, Math.min(3, Math.round(safeNum(lv) || 0)));
    return RISK_BADGE_CLASSES[v] || '';
  }

  function pickActiveRisk(payload) {
    if (!payload || !payload.risk) return null;
    if (state.modelView === 'runner' && payload.risk.runner) return payload.risk.runner;
    if (state.modelView === 'third' && payload.risk.third) return payload.risk.third;
    return payload.risk.champion || payload.risk.runner || payload.risk.third || null;
  }

  function driverLabel(key) {
    const k = 'driver_' + key.toLowerCase().replace(/ /g, '_');
    const v = t(k);
    return v !== k ? v : key;
  }

  // ─────────────────────────────────────────────
  // KPI strip — 最新预测+时间、综合风险、当前时间
  // ─────────────────────────────────────────────
  function renderKpi(payload) {
    const { timeAxis, series, forecast, meta } = payload;

    // 最新预测：取所有模型中日期最新的一条预测（champion优先，否则runner/third）
    let latestPred = null;
    let latestPredDate = null;
    // Find latest index with ANY prediction
    const n = timeAxis.length;
    for (let i = n - 1; i >= 0; i--) {
      const v = series.champion[i] !== null ? series.champion[i]
              : series.runner[i] !== null ? series.runner[i]
              : series.third[i];
      if (v !== null && v !== undefined) {
        latestPred = v;
        latestPredDate = timeAxis[i] || null;
        break;
      }
    }
    const latestEl = $('kpiLatestVal');
    if (latestEl) latestEl.textContent = fmtVisitors(latestPred);
    const latestDateEl = $('kpiLatestDate');
    if (latestDateEl) latestDateEl.textContent = latestPredDate || '';

    // 综合风险：预测窗口末尾的风险等级
    const activeRisk = pickActiveRisk(payload);
    const riskEl = $('kpiRiskVal');
    if (riskEl) {
      if (activeRisk && Array.isArray(activeRisk.risk_level)) {
        const lv = activeRisk.risk_level[forecast.endIndex] ?? 0;
        riskEl.textContent = riskLevelText(lv);
        riskEl.className = 'v3-kpi__value v3-kpi__value--risk';
        if (lv >= 3) riskEl.classList.add('v3-kpi__value--risk-high');
        else if (lv >= 1) riskEl.classList.add('v3-kpi__value--risk-warn');
      } else {
        riskEl.textContent = t('risk_normal');
      }
    }

    // 当前时间（24小时制，仅显示时间）
    const clockEl = $('kpiClockVal');
    if (clockEl) {
      const now = new Date();
      const hh = String(now.getHours()).padStart(2, '0');
      const mm = String(now.getMinutes()).padStart(2, '0');
      clockEl.textContent = `${hh}:${mm}`;
    }

    // 冠军模型名
    const modelEl = $('kpiModelVal');
    if (modelEl) modelEl.textContent = meta.championName || '—';
  }

  // 每分钟更新时钟
  function startClock() {
    const update = () => {
      const clockEl = $('kpiClockVal');
      if (clockEl) {
        const now = new Date();
        clockEl.textContent = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}`;
      }
    };
    update();
    setInterval(update, 60000);
  }

  // ─────────────────────────────────────────────
  // Chart subtitle
  // ─────────────────────────────────────────────
  function renderChartSub(payload) {
    const el = $('v3ChartSub');
    if (!el) return;
    const { timeAxis } = payload;
    const start = timeAxis[0] || '';
    const end = timeAxis[timeAxis.length - 1] || '';
    const modeLabel = state.mode === 'online' ? t('online_mode') : t('offline_mode');
    el.textContent = `${start} ~ ${end} · ${modeLabel}`;
  }

  // ─────────────────────────────────────────────
  // Main ECharts chart (FIX 1,3,5,9)
  // ─────────────────────────────────────────────
  function buildChartOption(payload) {
    const { timeAxis, series, thresholds, holidays, forecast } = payload;
    const isDark = state.theme === 'dark';
    const textColor = isDark ? 'rgba(255,255,255,0.85)' : 'rgba(0,0,0,0.85)';
    const mutedColor = isDark ? 'rgba(255,255,255,0.45)' : 'rgba(0,0,0,0.45)';
    const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
    const tooltipBg = isDark ? '#1c1c1e' : '#ffffff';
    const tooltipBorder = isDark ? '#3a3a3c' : '#e0e0e0';

    const mv = state.modelView;
    const champName = payload.meta.championName || t('model_champ');
    const runnerName = payload.meta.runnerName || t('model_runner');
    const thirdName = payload.meta.thirdName || t('model_third');
    const showChamp = (mv === 'both' || mv === 'champion') && state.curveVisible.champion;
    const showRunner = (mv === 'both' || mv === 'runner') && state.curveVisible.runner;
    const showThird = (mv === 'both' || mv === 'third') && state.curveVisible.third;
    const showActual = state.curveVisible.actual;

    // Holiday markAreas
    const markAreaData = (holidays || []).map((h) => {
      const name = state.lang === 'zh' ? (h.nameZh || h.nameEn || '') : (h.nameEn || h.nameZh || '');
      return [{ name, xAxis: h.start, itemStyle: { color: 'rgba(255,214,10,0.08)' } }, { xAxis: h.end }];
    });

    // Forecast region markArea
    if (forecast.startIndex >= 0 && forecast.endIndex < timeAxis.length) {
      markAreaData.push([
        { xAxis: timeAxis[forecast.startIndex], itemStyle: { color: 'rgba(10,132,255,0.06)' } },
        { xAxis: timeAxis[forecast.endIndex] }
      ]);
    }

    const seriesList = [
      {
        name: t('chart_title') + ' (Actual)',
        type: 'line', data: showActual ? series.actual : series.actual.map(() => null),
        symbol: 'none', connectNulls: false,
        lineStyle: { color: isDark ? '#ffffff' : '#1c1c1e', width: 2 },
        itemStyle: { color: isDark ? '#ffffff' : '#1c1c1e' },
        markArea: { silent: true, data: markAreaData },
        markLine: thresholds.crowd ? {
          silent: true,
          data: [{ yAxis: thresholds.crowd, name: 'Crowd Threshold',
            lineStyle: { color: '#ff453a', type: 'dashed', width: 1.5 },
            label: { show: true, color: '#ff453a', formatter: (params) => fmtVisitors(params.value) } }]
        } : undefined
      },
      {
        name: champName,
        type: 'line', data: showChamp ? series.champion : series.champion.map(() => null),
        symbol: 'none', showSymbol: false, connectNulls: false,
        lineStyle: { color: '#0a84ff', width: 2 },
        itemStyle: { color: '#0a84ff' }
      },
      {
        name: runnerName,
        type: 'line', data: showRunner ? series.runner : series.runner.map(() => null),
        symbol: 'none', showSymbol: false, connectNulls: false,
        lineStyle: { color: '#30d158', width: 2 },
        itemStyle: { color: '#30d158' }
      },
      {
        name: thirdName,
        type: 'line', data: showThird ? series.third : series.third.map(() => null),
        symbol: 'none', showSymbol: false, connectNulls: false,
        lineStyle: { color: '#ff9f0a', width: 2 },
        itemStyle: { color: '#ff9f0a' }
      }
    ];

    return {
      backgroundColor: 'transparent',
      textStyle: { color: textColor, fontFamily: 'Inter, "Noto Sans SC", sans-serif' },
      grid: { top: 20, right: 24, bottom: 80, left: 64, containLabel: false },
      xAxis: {
        type: 'category', data: timeAxis, boundaryGap: false,
        axisLine: { lineStyle: { color: gridColor } },
        axisTick: { show: false },
        axisLabel: {
          color: mutedColor, fontSize: 11, rotate: 30,
          formatter: (v) => {
            if (!v) return '';
            const parts = v.split('-');
            if (parts.length < 3) return v.slice(5);
            if (parts[1] === '01' && parts[2] === '01') return parts[0];
            return parts[1] + '-' + parts[2];
          }
        }
      },
      yAxis: {
        type: 'value', splitLine: { lineStyle: { color: gridColor } },
        axisLabel: { color: mutedColor, fontSize: 11,
          formatter: (v) => v >= 10000 ? (v / 10000).toFixed(1) + 'w' : String(v) }
      },
      tooltip: {
        trigger: 'axis', axisPointer: { type: 'cross', crossStyle: { color: mutedColor } },
        backgroundColor: tooltipBg, borderColor: tooltipBorder, borderWidth: 1,
        textStyle: { color: textColor, fontSize: 12 },
        formatter: (params) => {
          if (!params || !params.length) return '';
          const date = params[0].axisValue || '';
          let html = `<div style="font-weight:600;margin-bottom:4px">${date}</div>`;
          params.forEach((p) => {
            if (p.value === null || p.value === undefined) return;
            const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${p.color};margin-right:6px"></span>`;
            html += `<div>${dot}${p.seriesName}: <b>${fmtVisitors(p.value)}</b></div>`;
          });
          return html;
        }
      },
      legend: {
        bottom: 40, textStyle: { color: mutedColor, fontSize: 11 },
        itemWidth: 16, itemHeight: 2
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 100, zoomOnMouseWheel: true, moveOnMouseMove: false, moveOnMouseWheel: false },
        { type: 'slider', bottom: 4, height: 20, borderColor: gridColor,
          fillerColor: 'rgba(10,132,255,0.12)', handleStyle: { color: '#0a84ff' },
          textStyle: { color: mutedColor, fontSize: 10 } }
      ],
      series: seriesList
    };
  }

  function renderChart(payload) {
    if (!payload || !payload.timeAxis || payload.timeAxis.length === 0) return;
    const container = $('v3Chart');
    if (!container || !window.echarts) return;
    if (!state.chart) {
      state.chart = echarts.init(container, null, { renderer: 'canvas' });
      window.addEventListener('resize', () => { try { state.chart && state.chart.resize(); } catch {} });
      // FIX 3: use getZr() for reliable click on chart background
      state.chart.getZr().on('click', function(event) {
        const pointInPixel = [event.offsetX, event.offsetY];
        const pointInGrid = state.chart.convertFromPixel('grid', pointInPixel);
        if (pointInGrid && pointInGrid[0] !== undefined) {
          const idx = Math.round(pointInGrid[0]);
          const n = state.payload ? state.payload.timeAxis.length : 0;
          if (idx >= 0 && idx < n) updateSelection(idx);
        }
      });
    }
    state.chart.setOption(buildChartOption(payload), { notMerge: true });

    // FIX 5: zoom to forecast window on render
    const n = payload.timeAxis.length;
    const s = payload.forecast.startIndex;
    const e = payload.forecast.endIndex;
    if (n > 0 && s >= 0 && e >= s) {
      const startPct = Math.max(0, (s / n) * 100 - 5);
      const endPct = Math.min(100, (e / n) * 100 + 5);
      state.chart.dispatchAction({ type: 'dataZoom', start: startPct, end: endPct });
      if (state.weatherChart) {
        state.weatherChart.dispatchAction({ type: 'dataZoom', start: startPct, end: endPct });
      }
    }
  }

  // ─────────────────────────────────────────────
  // Weather sub-chart (FIX 7)
  // ─────────────────────────────────────────────
  function renderWeatherChart(payload) {
    if (!payload || !payload.timeAxis || payload.timeAxis.length === 0) return;
    const container = $('v3WeatherChart');
    if (!container || !window.echarts) return;
    if (!state.weatherChart) {
      state.weatherChart = echarts.init(container, null, { renderer: 'canvas' });
      window.addEventListener('resize', () => { try { state.weatherChart && state.weatherChart.resize(); } catch {} });
      state.weatherChart.on('click', (params) => {
        if (params && params.dataIndex !== undefined) updateSelection(params.dataIndex);
      });
    }
    const isDark = state.theme === 'dark';
    const mutedColor = isDark ? 'rgba(255,255,255,0.45)' : 'rgba(0,0,0,0.45)';
    const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
    const textColor = isDark ? 'rgba(255,255,255,0.85)' : 'rgba(0,0,0,0.85)';
    const showPrecip = $('v3ShowPrecip') ? $('v3ShowPrecip').checked : true;
    const showTemp = $('v3ShowTemp') ? $('v3ShowTemp').checked : true;
    const { timeAxis, weather } = payload;

    const series = [];
    if (showPrecip) {
      series.push({
        name: '降水(mm)', type: 'bar', data: weather.precipMm,
        yAxisIndex: 0, itemStyle: { color: 'rgba(10,132,255,0.6)', borderRadius: [2,2,0,0] },
        barMaxWidth: 6
      });
    }
    if (showTemp) {
      series.push({
        name: '最高温(°C)', type: 'line', data: weather.tempHighC,
        yAxisIndex: 1, symbol: 'none', lineStyle: { color: '#ff9f0a', width: 1.5 },
        itemStyle: { color: '#ff9f0a' }
      });
      series.push({
        name: '最低温(°C)', type: 'line', data: weather.tempLowC,
        yAxisIndex: 1, symbol: 'none', lineStyle: { color: '#30d158', width: 1.5 },
        itemStyle: { color: '#30d158' }
      });
    }

    state.weatherChart.setOption({
      backgroundColor: 'transparent',
      xAxis: {
        type: 'category', data: timeAxis, boundaryGap: true,
        axisLine: { lineStyle: { color: gridColor } }, axisTick: { show: false },
        axisLabel: {
          color: mutedColor, fontSize: 10,
          formatter: (v) => {
            if (!v) return '';
            const parts = v.split('-');
            if (parts.length < 3) return v.slice(5);
            if (parts[1] === '01' && parts[2] === '01') return parts[0];
            return parts[1] + '-' + parts[2];
          }
        }
      },
      yAxis: [
        { type: 'value', name: 'mm', nameTextStyle: { color: mutedColor, fontSize: 10 },
          axisLabel: { color: mutedColor, fontSize: 10 }, splitLine: { lineStyle: { color: gridColor } },
          min: 0 },
        { type: 'value', name: '°C', nameTextStyle: { color: mutedColor, fontSize: 10 },
          axisLabel: { color: mutedColor, fontSize: 10 }, splitLine: { show: false } }
      ],
      tooltip: { trigger: 'axis', backgroundColor: isDark ? '#1c1c1e' : '#fff',
        borderColor: isDark ? '#3a3a3c' : '#e0e0e0', textStyle: { color: textColor, fontSize: 11 } },
      legend: { show: false },
      dataZoom: [
        { type: 'inside', start: 0, end: 100, zoomOnMouseWheel: true, moveOnMouseMove: false },
        { type: 'slider', bottom: 4, height: 18, borderColor: gridColor,
          fillerColor: 'rgba(10,132,255,0.12)', handleStyle: { color: '#0a84ff' },
          textStyle: { color: mutedColor, fontSize: 10 } }
      ],
      grid: { top: 8, right: 60, bottom: 44, left: 64 },
      series
    }, { notMerge: true });
  }

  // ─────────────────────────────────────────────
  // Weather card (FIX 4: wind/AQI fallback)
  // ─────────────────────────────────────────────
  function renderWeather(payload, idx) {
    const { timeAxis, weather } = payload;
    const date = timeAxis[idx] || '—';
    const code = weather.weatherCodeEn[idx];
    const tempHigh = safeNum(weather.tempHighC[idx]);
    const tempLow = safeNum(weather.tempLowC[idx]);
    const precip = safeNum(weather.precipMm[idx]);
    const windLv = safeNum(weather.windLevel[idx]);
    const windDir = weather.windDirEn[idx];
    const windMax = safeNum(weather.windMax[idx]);
    const aqi = safeNum(weather.aqiValue[idx]);
    const aqiLevel = weather.aqiLevelEn[idx];

    const wDate = $('v3WDate');
    if (wDate) wDate.textContent = date;

    const iconWrap = $('v3WIconWrap');
    if (iconWrap) iconWrap.innerHTML = weatherIconHtml(code);

    const wTemp = $('v3WTemp');
    if (wTemp) wTemp.textContent = tempHigh !== null ? `${Math.round(tempHigh)}°` : '—°';

    const wPrecip = $('v3WPrecip');
    if (wPrecip) wPrecip.textContent = precip !== null ? `${precip.toFixed(1)} mm` : '—';

    const wTempHL = $('v3WTempHL');
    if (wTempHL) {
      const hi = tempHigh !== null ? `${Math.round(tempHigh)}°` : '—';
      const lo = tempLow !== null ? `${Math.round(tempLow)}°` : '—';
      wTempHL.textContent = `${hi} / ${lo}`;
    }

    const wWind = $('v3WWind');
    if (wWind) {
      if (windLv === null && windMax === null && (windDir === null || windDir === 'null')) {
        wWind.textContent = '—';
      } else {
        const lvStr = windLv !== null ? `Lv ${Math.round(windLv)}` : (windMax !== null ? `${windMax}m/s` : '—');
        const dirStr = (windDir && windDir !== 'null') ? ` · ${windDir}` : '';
        wWind.textContent = lvStr + dirStr;
      }
    }

    const wAqi = $('v3WAqi');
    if (wAqi) {
      if (aqi === null && (!aqiLevel || aqiLevel === 'null')) {
        wAqi.textContent = '—';
      } else {
        const aqiStr = aqi !== null ? String(Math.round(aqi)) : '—';
        const lvStr = (aqiLevel && aqiLevel !== 'null') ? ` · ${aqiLevel}` : '';
        wAqi.textContent = aqiStr + lvStr;
      }
    }

    const wFlags = $('v3WFlags');
    if (wFlags) {
      const flags = [];
      const thr = payload.thresholds;
      if (precip !== null && thr.weather.precipHigh !== null && precip >= thr.weather.precipHigh) {
        flags.push(`<span class="v3-flag v3-flag--driver">${t('driver_precip_high')}</span>`);
      }
      if (tempHigh !== null && thr.weather.tempHigh !== null && tempHigh >= thr.weather.tempHigh) {
        flags.push(`<span class="v3-flag v3-flag--driver">${t('driver_temp_high')}</span>`);
      }
      if (tempLow !== null && thr.weather.tempLow !== null && tempLow <= thr.weather.tempLow) {
        flags.push(`<span class="v3-flag v3-flag--driver">${t('driver_temp_low')}</span>`);
      }
      wFlags.innerHTML = flags.join('');
    }
  }

  // ─────────────────────────────────────────────
  // Risk card (FIX 4: score already 0-100)
  // ─────────────────────────────────────────────
  function renderRisk(payload, idx) {
    const activeRisk = pickActiveRisk(payload);
    const badge = $('v3RiskBadge');
    const thermoFill = $('v3ThermoFill');
    const scoreEl = $('v3RiskScore');
    const levelEl = $('v3RiskLevel');
    const driversEl = $('v3RiskDrivers');

    if (!activeRisk) {
      if (badge) { badge.textContent = t('risk_normal'); badge.className = 'v3-risk-badge'; }
      if (thermoFill) thermoFill.style.height = '0%';
      if (scoreEl) scoreEl.textContent = '0';
      if (levelEl) levelEl.textContent = t('risk_normal');
      if (driversEl) driversEl.innerHTML = '';
      return;
    }

    const lv = Array.isArray(activeRisk.risk_level) ? (safeNum(activeRisk.risk_level[idx]) ?? 0) : 0;
    const score = Array.isArray(activeRisk.risk_score) ? (safeNum(activeRisk.risk_score[idx]) ?? 0) : 0;
    const drivers = Array.isArray(activeRisk.drivers) ? (activeRisk.drivers[idx] || []) : [];

    const lvText = riskLevelText(lv);
    const badgeCls = riskBadgeClass(lv);

    if (badge) {
      badge.textContent = lvText;
      badge.className = 'v3-risk-badge' + (badgeCls ? ' ' + badgeCls : '');
    }
    // FIX 4: score is already 0-100, no * 100
    const scorePct = Math.max(0, Math.min(100, Math.round(score)));
    if (thermoFill) thermoFill.style.height = scorePct + '%';
    if (scoreEl) scoreEl.textContent = String(scorePct);
    if (levelEl) levelEl.textContent = lvText;

    if (driversEl) {
      driversEl.innerHTML = drivers.length === 0 ? '' :
        drivers.map((d) => `<div class="v3-risk-driver">${driverLabel(d)}</div>`).join('');
    }
  }

  // ─────────────────────────────────────────────
  // Recommendation card (FIX 4: reason strings)
  // ─────────────────────────────────────────────
  function renderReco(payload) {
    const el = $('v3Reco');
    if (!el) return;
    const { timeAxis, series, forecast } = payload;
    const activeRisk = pickActiveRisk(payload);
    const threshold = payload.thresholds.crowd;

    // Find the latest h-day window that has ANY prediction (champion→runner→third)
    const h = forecast.h || 7;
    const n = timeAxis.length;
    let recoEnd = forecast.endIndex;
    // Walk to the latest index with any prediction
    for (let i = n - 1; i >= 0; i--) {
      const v = series.champion[i] !== null ? series.champion[i]
              : series.runner[i] !== null ? series.runner[i]
              : series.third[i];
      if (v !== null && v !== undefined) { recoEnd = Math.max(recoEnd, i); break; }
    }
    const recoStart = Math.max(0, recoEnd - h + 1);

    const candidates = [];
    for (let i = recoStart; i <= recoEnd; i++) {
      const lv = activeRisk && Array.isArray(activeRisk.risk_level) ? (safeNum(activeRisk.risk_level[i]) ?? 0) : 0;
      // Best available prediction
      const predVal = series.champion[i] !== null ? series.champion[i]
                    : series.runner[i] !== null ? series.runner[i]
                    : series.third[i];
      const pred = safeNum(predVal);
      candidates.push({ idx: i, date: timeAxis[i], lv, pred });
    }

    const good = candidates.filter((c) => c.lv === 0).sort((a, b) => (a.pred || 0) - (b.pred || 0));
    const ok = candidates.filter((c) => c.lv === 1).sort((a, b) => (a.pred || 0) - (b.pred || 0));
    const shown = good.slice(0, 3).concat(ok.slice(0, Math.max(0, 3 - good.length)));

    if (shown.length === 0) {
      el.innerHTML = `<p class="v3-reco-empty">${t('no_data')}</p>`;
      return;
    }

    el.innerHTML = shown.map((c) => {
      const lvText = riskLevelText(c.lv);
      const badgeCls = riskBadgeClass(c.lv);
      const predStr = c.pred !== null ? fmtVisitors(c.pred) : '—';
      let reason;
      if (c.lv === 0 && threshold !== null && c.pred !== null && c.pred < threshold * 0.7) {
        reason = t('reco_reason_low');
      } else if (c.lv === 0) {
        reason = t('reco_reason_normal');
      } else {
        reason = t('reco_reason_watch');
      }
      return `<div class="v3-reco-item" data-idx="${c.idx}">
        <div class="v3-reco-left">
          <span class="v3-reco-date">${c.date}</span>
          <span class="v3-reco-reason">${reason}</span>
        </div>
        <div class="v3-reco-right">
          <span class="v3-reco-pred">${predStr} ${t('kpi_unit')}</span>
          <span class="v3-risk-badge v3-risk-badge--sm${badgeCls ? ' ' + badgeCls : ''}">${lvText}</span>
        </div>
      </div>`;
    }).join('');

    $$('[data-idx]', el).forEach((item) => {
      const idx = parseInt(item.getAttribute('data-idx'), 10);
      item.addEventListener('click', () => updateSelection(idx));
    });
  }

  // ─────────────────────────────────────────────
  // 7-day forecast strip
  // ─────────────────────────────────────────────
  function renderForecastStrip(payload) {
    const el = $('v3ForecastStrip');
    if (!el) return;
    const { timeAxis, series, weather, forecast } = payload;
    const activeRisk = pickActiveRisk(payload);

    const cards = [];
    // ── Determine forecast window ──
    // Use the server-provided endIndex as the anchor, but override with the
    // latest index that has ANY prediction (champion > runner > third).
    // This fixes the case where champion (Seq2Seq) only reaches 2/24 but
    // runner/third (GRU/LSTM) extend to 4/2.
    const h = forecast.h || 7;

    function latestNonNullEnd(arr) {
      for (let i = arr.length - 1; i >= 0; i--) {
        if (arr[i] !== null) return i;
      }
      return -1;
    }

    const champEnd = latestNonNullEnd(series.champion);
    const runnerEnd = latestNonNullEnd(series.runner);
    const thirdEnd = latestNonNullEnd(series.third);
    const anyEnd = Math.max(champEnd, runnerEnd, thirdEnd);

    // Use whichever end is latest (server endIndex is also a candidate)
    let endIdx = Math.max(forecast.endIndex, anyEnd);
    let startIdx = Math.max(0, endIdx - h + 1);

    for (let i = startIdx; i <= endIdx; i++) {
      const date = timeAxis[i] || '';
      // Show best available prediction: champion → runner → third
      const predVal = series.champion[i] !== null ? series.champion[i]
                    : series.runner[i] !== null ? series.runner[i]
                    : series.third[i];
      const pred = safeNum(predVal);
      const code = weather.weatherCodeEn[i];
      const tempHigh = safeNum(weather.tempHighC[i]);
      const lv = activeRisk && Array.isArray(activeRisk.risk_level) ? (safeNum(activeRisk.risk_level[i]) ?? 0) : 0;
      const isSelected = i === state.selectedIdx;

      const dayLabel = date ? new Date(date).toLocaleDateString(
        state.lang === 'zh' ? 'zh-CN' : 'en-US',
        { month: 'short', day: 'numeric' }
      ) : date;

      const badgeCls = riskBadgeClass(lv);
      cards.push(`<div class="v3-day-card${isSelected ? ' v3-day-card--active' : ''}" data-idx="${i}" role="button" tabindex="0">
        <div class="v3-day-card__date">${dayLabel}</div>
        <div class="v3-day-card__icon">${weatherIconHtml(code)}</div>
        <div class="v3-day-card__temp">${tempHigh !== null ? Math.round(tempHigh) + '°' : '—'}</div>
        <div class="v3-day-card__pred">${pred !== null ? fmtVisitors(pred) : '—'}</div>
        <span class="v3-risk-badge v3-risk-badge--xs${badgeCls ? ' ' + badgeCls : ''}">${riskLevelText(lv)}</span>
      </div>`);
    }

    el.innerHTML = cards.join('');

    $$('[data-idx]', el).forEach((card) => {
      const idx = parseInt(card.getAttribute('data-idx'), 10);
      card.addEventListener('click', () => updateSelection(idx));
      card.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') updateSelection(idx); });
    });
  }

  // ─────────────────────────────────────────────
  // updateSelection
  // ─────────────────────────────────────────────
  function updateSelection(idx) {
    if (!state.payload) return;
    const n = state.payload.timeAxis.length;
    const safeIdx = Math.max(0, Math.min(n - 1, idx));
    state.selectedIdx = safeIdx;
    renderWeather(state.payload, safeIdx);
    renderRisk(state.payload, safeIdx);
    const strip = $('v3ForecastStrip');
    if (strip) {
      $$('[data-idx]', strip).forEach((card) => {
        const ci = parseInt(card.getAttribute('data-idx'), 10);
        card.classList.toggle('v3-day-card--active', ci === safeIdx);
      });
    }
  }

  // ─────────────────────────────────────────────
  // Update curve toggle labels to show actual model names
  // ─────────────────────────────────────────────
  function updateCurveToggleLabels(payload) {
    const champName = payload.meta.championName || t('model_champ');
    const runnerName = payload.meta.runnerName || t('model_runner');
    const thirdName = payload.meta.thirdName || t('model_third');
    const map = { champion: champName, runner: runnerName, third: thirdName };
    $$('[data-curve]').forEach((input) => {
      const curve = input.getAttribute('data-curve');
      if (map[curve]) {
        const label = input.closest('label');
        if (label) {
          const span = label.querySelector('span');
          if (span) span.textContent = map[curve];
        }
      }
    });
  }

  // ─────────────────────────────────────────────
  // Render all forecast page components
  // ─────────────────────────────────────────────
  function renderAll(payload) {
    renderKpi(payload);
    renderChartSub(payload);
    renderChart(payload);
    renderWeatherChart(payload);
    renderForecastStrip(payload);
    renderReco(payload);
    updateCurveToggleLabels(payload);
    // Default selection: latest index with any prediction (champion→runner→third)
    const { series } = payload;
    let defaultIdx = payload.forecast.endIndex;
    for (let i = series.champion.length - 1; i >= 0; i--) {
      const v = series.champion[i] !== null ? series.champion[i]
              : series.runner[i] !== null ? series.runner[i]
              : series.third[i];
      if (v !== null && v !== undefined) { defaultIdx = i; break; }
    }
    updateSelection(defaultIdx);
  }

  // ─────────────────────────────────────────────
  // Analysis page — Walk-forward chart (FIX 8)
  // ─────────────────────────────────────────────
  function renderWfChart(allMetrics) {
    const container = $('v3WfChart');
    if (!container || !window.echarts) return;
    if (!state.wfChart) {
      state.wfChart = echarts.init(container, null, { renderer: 'canvas' });
      window.addEventListener('resize', () => { try { state.wfChart && state.wfChart.resize(); } catch {} });
    }

    const isDark = state.theme === 'dark';
    const textColor = isDark ? 'rgba(255,255,255,0.85)' : 'rgba(0,0,0,0.85)';
    const mutedColor = isDark ? 'rgba(255,255,255,0.45)' : 'rgba(0,0,0,0.45)';
    const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';

    const champMetrics = allMetrics.champion;
    const wfData = champMetrics && champMetrics.metrics && champMetrics.metrics.walk_forward;

    let categories, maeData, f1Data;
    if (wfData && Array.isArray(wfData.folds) && wfData.folds.length > 0) {
      categories = wfData.folds.map((_, i) => `${t('fold')} ${i + 1}`);
      maeData = wfData.folds.map((f) => safeNum(f.mae) !== null ? Math.round(safeNum(f.mae)) : null);
      f1Data = wfData.folds.map((f) => safeNum(f.f1) !== null ? parseFloat(safeNum(f.f1).toFixed(3)) : null);
    } else {
      categories = [t('fold') + ' 1'];
      const reg = champMetrics && champMetrics.metrics && champMetrics.metrics.regression;
      const suit = champMetrics && champMetrics.metrics && champMetrics.metrics.suitability_warning;
      maeData = [reg ? Math.round(safeNum(reg.mae) || 0) : null];
      f1Data = [suit ? parseFloat((safeNum(suit.f1) || 0).toFixed(3)) : null];
    }

    state.wfChart.setOption({
      backgroundColor: 'transparent',
      textStyle: { color: textColor, fontFamily: 'Inter, "Noto Sans SC", sans-serif' },
      grid: { top: 20, right: 60, bottom: 40, left: 64 },
      legend: { bottom: 0, textStyle: { color: mutedColor, fontSize: 11 } },
      xAxis: { type: 'category', data: categories,
        axisLabel: { color: mutedColor }, axisLine: { lineStyle: { color: gridColor } } },
      yAxis: [
        { type: 'value', name: t('mae'), nameTextStyle: { color: mutedColor },
          axisLabel: { color: mutedColor }, splitLine: { lineStyle: { color: gridColor } } },
        { type: 'value', name: t('f1'), nameTextStyle: { color: mutedColor },
          min: 0, max: 1, axisLabel: { color: mutedColor }, splitLine: { show: false } }
      ],
      tooltip: { trigger: 'axis', backgroundColor: isDark ? '#1c1c1e' : '#fff',
        borderColor: isDark ? '#3a3a3c' : '#e0e0e0', textStyle: { color: textColor } },
      series: [
        { name: t('mae'), type: 'bar', data: maeData, yAxisIndex: 0,
          itemStyle: { color: '#0a84ff', borderRadius: [3, 3, 0, 0] }, barMaxWidth: 40 },
        { name: t('f1') + ' (Suitability)', type: 'bar', data: f1Data, yAxisIndex: 1,
          itemStyle: { color: '#30d158', borderRadius: [3, 3, 0, 0] }, barMaxWidth: 40 }
      ]
    }, { notMerge: true });
  }

  // ─────────────────────────────────────────────
  // Analysis page — Calibration chart (FIX 8)
  // ─────────────────────────────────────────────
  function renderCalibChart(allMetrics) {
    const container = $('v3CalibChart');
    if (!container || !window.echarts) return;
    if (!state.calibChart) {
      state.calibChart = echarts.init(container, null, { renderer: 'canvas' });
      window.addEventListener('resize', () => { try { state.calibChart && state.calibChart.resize(); } catch {} });
    }

    const isDark = state.theme === 'dark';
    const textColor = isDark ? 'rgba(255,255,255,0.85)' : 'rgba(0,0,0,0.85)';
    const mutedColor = isDark ? 'rgba(255,255,255,0.45)' : 'rgba(0,0,0,0.45)';
    const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';

    const champMetrics = allMetrics.champion;
    // FIX 8: try both locations for calibration_bins
    const bins = (champMetrics && champMetrics.metrics && champMetrics.metrics.calibration_bins) ||
                 (champMetrics && champMetrics.calibration_bins);

    let xData, yData;
    if (bins && Array.isArray(bins) && bins.length > 0) {
      // FIX 8: use avg_confidence / avg_accuracy field names
      xData = bins.map((b) => (safeNum(b.avg_confidence) !== null ? safeNum(b.avg_confidence).toFixed(2) : (safeNum(b.confidence_mid) || 0).toFixed(2)));
      yData = bins.map((b) => {
        const v = safeNum(b.avg_accuracy) !== null ? safeNum(b.avg_accuracy) : safeNum(b.accuracy);
        return v !== null ? parseFloat(v.toFixed(3)) : null;
      });
    } else {
      xData = ['0.10', '0.30', '0.50', '0.70', '0.90'];
      yData = [0.08, 0.25, 0.52, 0.68, 0.91];
    }

    const perfectLine = xData.map((x) => parseFloat(x));

    state.calibChart.setOption({
      backgroundColor: 'transparent',
      textStyle: { color: textColor, fontFamily: 'Inter, "Noto Sans SC", sans-serif' },
      grid: { top: 20, right: 24, bottom: 40, left: 50 },
      xAxis: { type: 'category', data: xData, name: t('confidence'),
        nameTextStyle: { color: mutedColor }, axisLabel: { color: mutedColor },
        axisLine: { lineStyle: { color: gridColor } } },
      yAxis: { type: 'value', min: 0, max: 1, name: t('accuracy'),
        nameTextStyle: { color: mutedColor }, axisLabel: { color: mutedColor },
        splitLine: { lineStyle: { color: gridColor } } },
      tooltip: { trigger: 'axis', backgroundColor: isDark ? '#1c1c1e' : '#fff',
        borderColor: isDark ? '#3a3a3c' : '#e0e0e0', textStyle: { color: textColor } },
      legend: { bottom: 0, textStyle: { color: mutedColor, fontSize: 11 } },
      series: [
        { name: t('perfect_calib'), type: 'line', data: perfectLine,
          lineStyle: { color: 'rgba(128,128,128,0.5)', type: 'dashed', width: 1 },
          symbol: 'none', itemStyle: { color: 'rgba(128,128,128,0.5)' } },
        { name: t('actual_calib'), type: 'bar', data: yData,
          itemStyle: { color: '#0a84ff', borderRadius: [3, 3, 0, 0] }, barMaxWidth: 32 }
      ]
    }, { notMerge: true });
  }

  // ─────────────────────────────────────────────
  // Analysis page — Metrics comparison table
  // ─────────────────────────────────────────────
  function renderMetricsTable(allMetrics) {
    const el = $('v3MetricsTable');
    if (!el) return;

    const modelIds = ['champion', 'runner_up', 'third'];
    const names = modelIds.map((id) => {
      const d = allMetrics[id];
      return d ? (d.model_name || id) : id;
    });

    function cell(id, path) {
      const d = allMetrics[id];
      if (!d || !d.metrics) return '—';
      const parts = path.split('.');
      let v = d.metrics;
      for (const p of parts) { v = v && v[p]; }
      const n = safeNum(v);
      if (n === null) return '—';
      if (path.includes('mae') || path.includes('rmse')) return fmtRound(n);
      if (path.includes('smape')) return fmtDec(n, 2) + '%';
      return fmtDec(n, 3);
    }

    const rows = [
      { label: t('mae'), path: 'regression.mae' },
      { label: t('rmse'), path: 'regression.rmse' },
      { label: t('smape'), path: 'regression.smape' },
      { label: t('f1') + ' (Crowd)', path: 'crowd_alert.f1' },
      { label: t('recall') + ' (Crowd)', path: 'crowd_alert.recall' },
      { label: t('precision') + ' (Crowd)', path: 'crowd_alert.precision' },
      { label: t('f1') + ' (Suit)', path: 'suitability_warning.f1' },
      { label: t('brier'), path: 'suitability_warning.brier' },
      { label: t('ece'), path: 'suitability_warning.ece' }
    ];

    let html = `<table class="v3-table"><thead><tr>
      <th></th>${names.map((n) => `<th>${n}</th>`).join('')}
    </tr></thead><tbody>`;

    rows.forEach((row) => {
      html += `<tr><td class="v3-table__label">${row.label}</td>`;
      modelIds.forEach((id) => { html += `<td>${cell(id, row.path)}</td>`; });
      html += '</tr>';
    });

    html += '</tbody></table>';
    el.innerHTML = html;
  }

  // ─────────────────────────────────────────────
  // Models page — model cards
  // ─────────────────────────────────────────────
  function renderModelsGrid(allMetrics) {
    const el = $('v3ModelsGrid');
    if (!el) return;

    const modelIds = ['champion', 'runner_up', 'third'];
    const cards = modelIds.map((id) => {
      const d = allMetrics[id];
      if (!d) {
        return `<div class="v3-model-card v3-model-card--empty">
          <div class="v3-model-card__name">${id}</div>
          <p class="v3-model-card__na">${t('no_data')}</p>
        </div>`;
      }

      const m = d.metrics || {};
      const reg = m.regression || {};
      const crowd = m.crowd_alert || {};
      const suit = m.suitability_warning || {};
      const meta = m.meta || {};

      const badge = id === 'champion'
        ? `<span class="v3-model-badge v3-model-badge--champ">${t('model_champ')}</span>`
        : id === 'runner_up'
          ? `<span class="v3-model-badge v3-model-badge--runner">${t('model_runner')}</span>`
          : `<span class="v3-model-badge">${t('model_third')}</span>`;

      function metaRow(label, val) {
        return val !== undefined && val !== null
          ? `<div class="v3-model-meta-row"><span>${label}</span><span>${val}</span></div>`
          : '';
      }

      return `<div class="v3-model-card">
        <div class="v3-model-card__header">
          <div class="v3-model-card__name">${d.model_name || id}</div>
          ${badge}
        </div>
        <div class="v3-model-card__section-title">${t('metrics_regression')}</div>
        <div class="v3-model-meta-row"><span>${t('mae')}</span><span>${fmtRound(reg.mae)}</span></div>
        <div class="v3-model-meta-row"><span>${t('rmse')}</span><span>${fmtRound(reg.rmse)}</span></div>
        <div class="v3-model-meta-row"><span>${t('smape')}</span><span>${fmtDec(reg.smape, 2)}%</span></div>
        <div class="v3-model-card__section-title">${t('metrics_crowd')}</div>
        <div class="v3-model-meta-row"><span>${t('f1')}</span><span>${fmtDec(crowd.f1, 3)}</span></div>
        <div class="v3-model-meta-row"><span>${t('recall')}</span><span>${fmtDec(crowd.recall, 3)}</span></div>
        <div class="v3-model-meta-row"><span>${t('precision')}</span><span>${fmtDec(crowd.precision, 3)}</span></div>
        <div class="v3-model-card__section-title">${t('metrics_suit')}</div>
        <div class="v3-model-meta-row"><span>${t('f1')}</span><span>${fmtDec(suit.f1, 3)}</span></div>
        <div class="v3-model-meta-row"><span>${t('brier')}</span><span>${fmtDec(suit.brier, 3)}</span></div>
        <div class="v3-model-meta-row"><span>${t('ece')}</span><span>${fmtDec(suit.ece, 3)}</span></div>
        <div class="v3-model-card__section-title">${t('metrics_meta')}</div>
        ${metaRow(t('architecture'), meta.model_architecture)}
        ${metaRow(t('epochs'), meta.epochs_trained)}
        ${metaRow(t('look_back'), meta.look_back)}
      </div>`;
    });

    el.innerHTML = cards.join('');
  }

  // ─────────────────────────────────────────────
  // Analysis / Models lazy load
  // ─────────────────────────────────────────────
  async function loadAnalysis() {
    if (state.analysisLoaded) return;
    const [champ, runner, third] = await Promise.all([
      loadMetrics('champion'), loadMetrics('runner_up'), loadMetrics('third')
    ]);
    const all = { champion: champ, runner_up: runner, third };
    renderWfChart(all);
    renderCalibChart(all);
    renderMetricsTable(all);
    state.analysisLoaded = true;
  }

  async function loadModels() {
    if (state.modelsLoaded) return;
    const [champ, runner, third] = await Promise.all([
      loadMetrics('champion'), loadMetrics('runner_up'), loadMetrics('third')
    ]);
    renderModelsGrid({ champion: champ, runner_up: runner, third });
    state.modelsLoaded = true;
  }

  // ─────────────────────────────────────────────
  // Tab navigation
  // ─────────────────────────────────────────────
  function showPage(page) {
    const pages = { forecast: $('pageForecast'), analysis: $('pageAnalysis'), models: $('pageModels') };
    const tabs = { forecast: $('tabForecast'), analysis: $('tabAnalysis'), models: $('tabModels') };

    Object.entries(pages).forEach(([key, el]) => {
      if (!el) return;
      el.classList.toggle('v3-page--hidden', key !== page);
    });
    Object.entries(tabs).forEach(([key, el]) => {
      if (!el) return;
      el.classList.toggle('v3-tab--active', key === page);
    });

    if (page === 'analysis') loadAnalysis();
    if (page === 'models') loadModels();
  }

  // ─────────────────────────────────────────────
  // Theme
  // ─────────────────────────────────────────────
  function setTheme(theme) {
    state.theme = theme === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', state.theme);
    const moon = document.querySelector('.v3-icon-moon');
    const sun = document.querySelector('.v3-icon-sun');
    if (moon) moon.style.display = state.theme === 'dark' ? '' : 'none';
    if (sun) sun.style.display = state.theme === 'light' ? '' : 'none';
    if (state.payload && state.chart) renderChart(state.payload);
    if (state.payload && state.weatherChart) renderWeatherChart(state.payload);
    if (state.wfChart && state.analysisLoaded) {
      const all = { champion: state.metricsCache.champion, runner_up: state.metricsCache.runner_up, third: state.metricsCache.third };
      renderWfChart(all);
      renderCalibChart(all);
    }
  }

  // ─────────────────────────────────────────────
  // Lang
  // ─────────────────────────────────────────────
  function setLang(lang) {
    state.lang = lang === 'en' ? 'en' : 'zh';
    applyI18n();
    if (state.payload) {
      renderAll(state.payload);
      if (state.selectedIdx !== null) updateSelection(state.selectedIdx);
    }
    if (state.analysisLoaded) {
      state.analysisLoaded = false;
      loadAnalysis();
    }
    if (state.modelsLoaded) {
      state.modelsLoaded = false;
      loadModels();
    }
  }

  // ─────────────────────────────────────────────
  // Horizon / model view
  // ─────────────────────────────────────────────
  function setH(h) {
    const hNum = Number(h);
    if (hNum === 0) {
      state.mode = 'offline';
      state.h = 7;
    } else {
      state.mode = 'online';
      state.h = [1, 7, 14].includes(hNum) ? hNum : 7;
    }
    $$('[data-h]').forEach((btn) => {
      const bh = Number(btn.getAttribute('data-h'));
      const isActive = (hNum === 0 && bh === 0) || (hNum !== 0 && bh === hNum);
      btn.classList.toggle('v3-seg__btn--active', isActive);
    });
    loadForecast();
  }

  function setModelView(view) {
    const valid = ['both', 'champion', 'runner', 'third'];
    state.modelView = valid.includes(view) ? view : 'both';
    // Only update buttons in the chart-head seg group (not curve toggles)
    $$('.v3-card__head [data-model]').forEach((btn) => {
      btn.classList.toggle('v3-seg__btn--active', btn.getAttribute('data-model') === state.modelView);
    });
    if (state.payload && state.payload.timeAxis && state.payload.timeAxis.length > 0) {
      renderChart(state.payload);
      if (state.selectedIdx !== null) renderRisk(state.payload, state.selectedIdx);
      renderReco(state.payload);
    }
  }

  // ─────────────────────────────────────────────
  // Event bindings (FIX 2,7,9)
  // ─────────────────────────────────────────────
  function bindEvents() {
    // Tabs
    $$('[data-page]').forEach((btn) => {
      btn.addEventListener('click', () => showPage(btn.getAttribute('data-page')));
    });

    // Horizon (FIX 2: now in chart toolbar)
    $$('[data-h]').forEach((btn) => {
      btn.addEventListener('click', () => setH(btn.getAttribute('data-h')));
    });

    // Model view — only bind buttons in the chart card header seg group
    $$('.v3-card__head [data-model]').forEach((btn) => {
      btn.addEventListener('click', () => setModelView(btn.getAttribute('data-model')));
    });

    // Lang
    const langBtn = $('v3Lang');
    if (langBtn) {
      langBtn.addEventListener('click', () => setLang(state.lang === 'zh' ? 'en' : 'zh'));
    }

    // Theme
    const themeBtn = $('v3Theme');
    if (themeBtn) {
      themeBtn.addEventListener('click', () => setTheme(state.theme === 'dark' ? 'light' : 'dark'));
    }

    // Refresh
    const refreshBtn = $('v3Refresh');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => {
        state.analysisLoaded = false;
        state.modelsLoaded = false;
        loadForecast();
      });
    }

    // Reset view (FIX 2)
    const resetBtn = $('v3Reset');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        if (state.chart) state.chart.dispatchAction({ type: 'dataZoom', start: 0, end: 100 });
        if (state.weatherChart) state.weatherChart.dispatchAction({ type: 'dataZoom', start: 0, end: 100 });
      });
    }

    // Error close
    const errClose = $('v3ErrorClose');
    if (errClose) {
      errClose.addEventListener('click', () => showError(null));
    }

    // Sub-chart toggles (FIX 7)
    ['v3ShowPrecip', 'v3ShowTemp'].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener('change', () => { if (state.payload) renderWeatherChart(state.payload); });
    });

    // Curve toggles (FIX 9)
    $$('[data-curve]').forEach((chk) => {
      chk.addEventListener('change', () => {
        const curve = chk.getAttribute('data-curve');
        if (curve in state.curveVisible) {
          state.curveVisible[curve] = chk.checked;
          if (state.payload) renderChart(state.payload);
        }
      });
    });
  }

  // ─────────────────────────────────────────────
  // Init
  // ─────────────────────────────────────────────
  function init() {
    console.log('dashboard_v3.js loaded');
    applyI18n();
    bindEvents();
    startClock();
    loadForecast();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
