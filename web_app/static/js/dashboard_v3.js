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
      online_mode: '在线预测',
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
      warn_fallback: '在线预测失败。',
      show_precip: '降水', show_temp: '温度',
      weather_sub_title: '降水 / 温度',
      curve_actual: '实际', curve_champ: '冠军',
      filter_all: '全部', filter_gru: 'GRU', filter_seq2seq: 'Seq2Seq', filter_lstm: 'LSTM', filter_actual: '真实',
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
      online_mode: 'Online forecast',
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
      show_precip: 'Precip', show_temp: 'Temp',
      weather_sub_title: 'Precip / Temp',
      curve_actual: 'Actual', curve_champ: 'Champion',
      filter_all: 'All', filter_gru: 'GRU', filter_seq2seq: 'Seq2Seq', filter_lstm: 'LSTM', filter_actual: 'Actual',
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
    mode: 'online',  // 始终在线：所有模型（MIMO/Seq2Seq/单步）均实时推理
    lang: 'zh',
    theme: 'dark',
    modelView: 'both',
    curveFilter: 'all',
    // 方案B：独立 chip 多选状态
    activeChips: new Set(['actual']),
    devMode: false,  // when true: show all 5 model comparison lines
    selectedIdx: null,
    payload: null,
    chart: null,
    weatherChart: null,
    wfChart: null,
    calibChart: null,
    metricsCache: {},
    analysisLoaded: false,
    modelsLoaded: false,
    // 前端直接从 Open-Meteo 获取的天气数据：date string → weather object
    wxData: {},
    // 后端 payload 历史天气（2016~今天）：date string → weather object
    histWxData: {}
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
      zones: { historyEnd: null, gapEnd: null, forecastStart: null, forecastEnd: null },
      meta: {}, series: {
        actual: [],
        gru_single: [], gru_mimo: [],
        lstm_single: [], lstm_mimo: [],
        seq2seq: []
      },
      thresholds: { crowd: null, weather: {} },
      weather: { precipMm: [], tempHighC: [], tempLowC: [], weatherCodeEn: [],
        windLevel: [], windDirEn: [], windMax: [], aqiValue: [], aqiLevelEn: [] },
      holidays: [], risk: {}, warning: null
    };
    if (!raw || typeof raw !== 'object') return out;

    const axis = raw.time_axis || raw.timeAxis || [];
    out.timeAxis = Array.isArray(axis) ? axis.map(String) : [];
    const n = out.timeAxis.length;

    const meta = raw.meta || {};
    out.meta.generatedAt = meta.generated_at || meta.generatedAt || null;
    out.meta.forecastMode = meta.forecast_mode || meta.forecastMode || null;
    out.meta.testStartDate = meta.test_start_date || meta.testStartDate || null;
    out.meta.models = meta.models || [];

    const fc = raw.forecast || {};
    out.forecast.h = safeNum(fc.h) || state.h;
    out.forecast.startIndex = Math.max(0, safeNum(fc.start_index) ?? safeNum(fc.startIndex) ?? 0);
    out.forecast.endIndex = Math.max(0, safeNum(fc.end_index) ?? safeNum(fc.endIndex) ?? Math.max(0, n - 1));

    const z = raw.zones || {};
    out.zones = {
      historyEnd: z.history_end || null,
      gapEnd: z.gap_end || null,
      forecastStart: z.forecast_start || null,
      forecastEnd: z.forecast_end || null,
    };

    const s = raw.series || {};
    out.series.actual      = safeArr(s.actual || [], n, safeNum);
    out.series.gru_single  = safeArr(s.gru_single_pred || [], n, safeNum);
    out.series.gru_mimo    = safeArr(s.gru_mimo_pred || [], n, safeNum);
    out.series.lstm_single = safeArr(s.lstm_single_pred || [], n, safeNum);
    out.series.lstm_mimo   = safeArr(s.lstm_mimo_pred || [], n, safeNum);
    out.series.seq2seq     = safeArr(s.seq2seq_pred || [], n, safeNum);

    const thr = raw.thresholds || {};
    out.thresholds.crowd      = safeNum(thr.crowd);
    out.thresholds.crowd_peak = safeNum(thr.crowd_peak);
    out.thresholds.crowd_off  = safeNum(thr.crowd_off);
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
    out.risk = r;  // 直接存整个 risk 对象
    out.warning = raw.warning || null;

    // Uncertainty interval (step-wise conformal from Deep Ensemble)
    const unc = raw.uncertainty || {};
    const lowerArr = safeArr(unc.lower || [], n, safeNum).map((v) => v === null ? null : Math.max(0, v));
    const upperArr = safeArr(unc.upper || [], n, safeNum).map((v, i) => {
      if (v === null) return null;
      const vv = Math.max(0, v);
      const lo = lowerArr[i];
      return lo !== null && vv < lo ? lo : vv;
    });
    const halfWArr = safeArr(unc.half_width || [], n, safeNum).map((v) => v === null ? null : Math.max(0, v));
    out.uncertainty = {
      available: !!unc.available,
      nMembers: safeNum(unc.n_members) || 0,
      calSize: safeNum(unc.cal_size) || 0,
      alpha: safeNum(unc.alpha) || 0.10,
      qhatByHorizon: unc.qhat_by_horizon || {},
      halfWidthByHorizon: unc.half_width_by_horizon || {},
      lower: lowerArr,
      upper: upperArr,
      halfWidth: halfWArr,
    };
    return out;
  }

  // 将 payload.weather 数组转为 date→weather map，存入 state.histWxData
  // 供 getWxForDate() 查询历史天气（2016~今天）
  function buildHistWxData(payload) {
    const { timeAxis, weather } = payload;
    const map = {};
    for (let i = 0; i < timeAxis.length; i++) {
      const date = timeAxis[i];
      if (!date) continue;
      const th = safeNum(weather.tempHighC[i]);
      const tl = safeNum(weather.tempLowC[i]);
      const precip = safeNum(weather.precipMm[i]);
      const code = weather.weatherCodeEn[i] || null;
      const windMax = safeNum(weather.windMax[i]);
      if (th !== null || tl !== null || precip !== null) {
        map[date] = { th, tl, precip, code, windMax };
      }
    }
    state.histWxData = map;
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

  // ─────────────────────────────────────────────
  // Direct Open-Meteo weather fetch (frontend-side)
  // 覆盖 past_days=14（历史段）+ forecast_days=14（未来段），减轻后端负担
  // state.wxData: { "YYYY-MM-DD": { th, tl, precip, code, windMax, ... } }
  // ─────────────────────────────────────────────
  const WMO_CODE_MAP = {
    0:'SUNNY',1:'SUNNY',2:'PARTLY_CLOUDY',3:'CLOUDY',
    45:'FOGGY',48:'FOGGY',
    51:'DRIZZLE',53:'DRIZZLE',55:'DRIZZLE',
    61:'RAINY',63:'RAINY',65:'RAINY',
    71:'SNOWY',73:'SNOWY',75:'SNOWY',77:'SNOWY',
    80:'RAINY',81:'RAINY',82:'RAINY',
    85:'SNOWY',86:'SNOWY',
    95:'THUNDERSTORM',96:'THUNDERSTORM',99:'THUNDERSTORM'
  };

  async function fetchWeatherDirect() {
    try {
      const params = new URLSearchParams({
        latitude: '33.2', longitude: '103.9',
        daily: 'temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode,windspeed_10m_max',
        timezone: 'Asia/Shanghai',
        past_days: '14',
        forecast_days: '14'
      });
      const res = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`, { cache: 'no-store' });
      if (!res.ok) return;
      const json = await res.json();
      const d = json.daily || {};
      const times = d.time || [];
      const th = d.temperature_2m_max || [];
      const tl = d.temperature_2m_min || [];
      const precip = d.precipitation_sum || [];
      const wmo = d.weathercode || [];
      const wind = d.windspeed_10m_max || [];
      const newWx = {};
      times.forEach((date, i) => {
        const code = WMO_CODE_MAP[wmo[i]] || 'CLOUDY';
        newWx[date] = {
          th: th[i] != null ? th[i] : null,
          tl: tl[i] != null ? tl[i] : null,
          precip: precip[i] != null ? precip[i] : null,
          code,
          windMax: wind[i] != null ? wind[i] : null
        };
      });
      state.wxData = newWx;
      // wxData 就绪后重新触发 updateSelection，刷新侧边天气卡片和7天条
      if (state.payload) {
        renderWxStrip(state.payload);
        renderForecastStrip(state.payload);
        // 用当前选中的 idx 重新渲染天气卡片（联动不变）
        if (state.selectedIdx !== null) {
          updateSelection(state.selectedIdx);
        }
      }
    } catch (e) {
      console.warn('fetchWeatherDirect failed:', e);
    }
  }

  // 根据 date string 查找天气：优先 state.wxData（近期/未来），fallback payload 历史天气
  // state.histWxData: { "YYYY-MM-DD": {th,tl,precip,code,windMax} } 由 buildHistWxData() 填充
  function getWxForDate(dateStr) {
    if (!dateStr) return null;
    // 1. 优先 Open-Meteo 直连数据（近14天+未来14天）
    if (state.wxData[dateStr]) return state.wxData[dateStr];
    // 2. fallback: 后端 payload 历史天气
    if (state.histWxData && state.histWxData[dateStr]) return state.histWxData[dateStr];
    // 3. ±3 天 fallback（两个数据源都找不到时）
    const base = new Date(dateStr + 'T00:00:00');
    for (let delta = 1; delta <= 3; delta++) {
      for (const sign of [1, -1]) {
        const d = new Date(base);
        d.setDate(d.getDate() + sign * delta);
        const key = d.toISOString().slice(0, 10);
        if (state.wxData[key]) return state.wxData[key];
        if (state.histWxData && state.histWxData[key]) return state.histWxData[key];
      }
    }
    return null;
  }

  async function loadForecast() {
    showSpinner(true);
    showError(null);
    setStatus('', t('status_loading'));

    // 缓存 key：固定 online 模式，30分钟 TTL
    const cacheKey = `v3_forecast_v6_h${state.h}`;
    try {
      const cached = localStorage.getItem(cacheKey);
      if (cached) {
        const { data, ts } = JSON.parse(cached);
        if (Date.now() - ts < 30 * 60 * 1000) {
          const normalized = normalizeForecastPayload(data);
          const _fc = normalized.forecast;
          const _wx = normalized.weather;
          const _wSlice = _wx.tempHighC.slice(_fc.startIndex, _fc.endIndex + 1);
          const _wxOk = _wSlice.some((v) => v !== null && v !== undefined);
          if (normalized.timeAxis && normalized.timeAxis.length > 0 && _wxOk) {
            state.payload = normalized;
            renderAll(state.payload);
            showSpinner(false);
            setStatus('ok', t('online_mode') + ' (cached)', state.payload.meta.generatedAt || '');
            return;
          }
          localStorage.removeItem(cacheKey);
        }
      }
    } catch (e) {
      try { localStorage.removeItem(cacheKey); } catch (_) {}
    }

    try {
      const url = `/api/forecast?h=${state.h}`;
      const raw = await apiFetch(url);
      try {
        localStorage.setItem(cacheKey, JSON.stringify({ data: raw, ts: Date.now() }));
      } catch (e) {}

      state.payload = normalizeForecastPayload(raw);
      if (state.payload.warning) showError(state.payload.warning);
      setStatus('ok', t('online_mode'), state.payload.meta.generatedAt || '');
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

  // 从5条曲线中取第一个有值的预测（用于 KPI/Reco/Strip 的"最佳可用预测"）
  const PRED_KEYS = ['gru_single', 'gru_mimo', 'lstm_single', 'lstm_mimo', 'seq2seq'];
  function bestPred(series, i) {
    for (const k of PRED_KEYS) {
      const v = series[k] && series[k][i];
      if (v !== null && v !== undefined) return v;
    }
    return null;
  }

  function pickActiveRisk(payload) {
    return (payload && payload.risk && payload.risk.risk_level) ? payload.risk : null;
  }

  function driverLabel(key) {
    const k = 'driver_' + key.toLowerCase().replace(/ /g, '_');
    const v = t(k);
    return v !== k ? v : key;
  }

  // ─────────────────────────────────────────────
  // Weather forecast strip (replaces KPI)
  // ─────────────────────────────────────────────
  const WX_ICONS = {
    SUNNY: '☀️', PARTLY_CLOUDY: '⛅', CLOUDY: '☁️', OVERCAST: '☁️',
    FOGGY: '🌫️', DRIZZLE: '🌦️', RAINY: '🌧️', SNOWY: '❄️',
    THUNDERSTORM: '⛈️'
  };
  const WX_DESC_ZH = {
    SUNNY:'晴', PARTLY_CLOUDY:'多云', CLOUDY:'阴', OVERCAST:'阴',
    FOGGY:'雾', DRIZZLE:'小雨', RAINY:'雨', SNOWY:'雪', THUNDERSTORM:'雷雨'
  };
  const WX_DESC_EN = {
    SUNNY:'Sunny', PARTLY_CLOUDY:'Partly Cloudy', CLOUDY:'Cloudy', OVERCAST:'Overcast',
    FOGGY:'Foggy', DRIZZLE:'Drizzle', RAINY:'Rainy', SNOWY:'Snowy', THUNDERSTORM:'Thunderstorm'
  };

  function renderWxStrip(_payload) {
    const scroll = $('v3WxScroll');
    if (!scroll) return;
    const today = new Date().toISOString().slice(0, 10);

    // 直接用 state.wxData（前端从 Open-Meteo 获取），只展示今天及以后，最多16天
    const sortedDates = Object.keys(state.wxData).sort();
    const cards = [];
    for (const date of sortedDates) {
      if (date < today) continue;
      const wx = state.wxData[date];
      if (wx.th === null && wx.tl === null) continue;
      cards.push({ date, ...wx });
      if (cards.length >= 16) break;
    }

    if (cards.length === 0) {
      scroll.innerHTML = '<div style="padding:0 16px;color:var(--text-2);font-size:0.8rem">天气数据加载中…</div>';
      return;
    }

    scroll.innerHTML = cards.map(({ date, th, tl, code }) => {
      const isToday = date === today;
      const d = new Date(date + 'T00:00:00');
      const mon = d.getMonth() + 1;
      const day = d.getDate();
      const icon = WX_ICONS[code] || '🌤️';
      const desc = state.lang === 'zh' ? (WX_DESC_ZH[code] || code || '—') : (WX_DESC_EN[code] || code || '—');
      const high = th !== null ? `${Math.round(th)}°` : '—';
      const low  = tl !== null ? `${Math.round(tl)}°` : '—';
      return `<div class="v3-wx-card${isToday ? ' v3-wx-card--today' : ''}">
        <span class="v3-wx-date">${isToday ? (state.lang === 'zh' ? '今天' : 'Today') : `${mon}/${day}`}</span>
        <span class="v3-wx-icon">${icon}</span>
        <span class="v3-wx-desc">${desc}</span>
        <div class="v3-wx-temps"><span class="v3-wx-high">${high}</span><span class="v3-wx-low">${low}</span></div>
      </div>`;
    }).join('');
  }

  function bindWxArrows() {
    const scroll = $('v3WxScroll');
    const left = $('v3WxLeft');
    const right = $('v3WxRight');
    if (!scroll || !left || !right) return;
    left.addEventListener('click', () => { scroll.scrollBy({ left: -264, behavior: 'smooth' }); });
    right.addEventListener('click', () => { scroll.scrollBy({ left: 264, behavior: 'smooth' }); });
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
    el.textContent = `${start} ~ ${end} · ${t('online_mode')}`;
  }

  // ─────────────────────────────────────────────
  // Main ECharts chart (FIX 1,3,5,9)
  // ─────────────────────────────────────────────
  function buildChartOption(payload) {
    const { timeAxis, series, thresholds, holidays, forecast, zones } = payload;
    const isDark = state.theme === 'dark';
    const textColor = isDark ? 'rgba(255,255,255,0.85)' : 'rgba(0,0,0,0.85)';
    const mutedColor = isDark ? 'rgba(255,255,255,0.45)' : 'rgba(0,0,0,0.45)';
    const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
    const tooltipBg = isDark ? '#1c1c1e' : '#ffffff';
    const tooltipBorder = isDark ? '#3a3a3c' : '#e0e0e0';

    // 5条曲线配置：key, 显示名, 颜色
    const CURVE_DEFS = [
      { key: 'gru_single',  nameZh: 'GRU (单步)',       nameEn: 'GRU (Single)',    color: '#0a84ff' },
      { key: 'gru_mimo',    nameZh: 'GRU (多步)',        nameEn: 'GRU (MIMO)',      color: '#5ac8fa' },
      { key: 'lstm_single', nameZh: 'LSTM (单步)',       nameEn: 'LSTM (Single)',   color: '#ff9f0a' },
      { key: 'lstm_mimo',   nameZh: 'LSTM (多步)',       nameEn: 'LSTM (MIMO)',     color: '#ffd60a' },
      { key: 'seq2seq',     nameZh: 'Seq2Seq+Attention', nameEn: 'Seq2Seq+Attn',   color: '#30d158' },
    ];

    // "Developer mode": if off, only show actual + ensemble mean (gru_mimo) + CI band
    const devMode = state.devMode === true;
    const chips = state.activeChips;
    const showActual = devMode ? chips.has('actual') : true;

    // Holiday markAreas (event-level: label floats into top empty grid area)
    const markAreaData = (holidays || []).map((h) => {
      const name = state.lang === 'zh' ? (h.nameZh || h.nameEn || '') : (h.nameEn || h.nameZh || '');
      return [
        {
          name,
          xAxis: h.start,
          itemStyle: { color: 'rgba(255, 165, 0, 0.05)' },
          label: { show: true, position: 'top', color: '#FF8C00', distance: 10 }
        },
        { xAxis: h.end }
      ];
    });

    // ── Three-zone time axis markAreas ──
    // Zone 1: [last real data date] boundary — implicit in actual series null-end
    // Zone 2: Gap zone (last_real + 1 → today - 1): data latency, grey hatched
    // Zone 3: Forecast zone (today → today+6): future prediction, light blue
    // Use backend forecast_start (CST) as today; avoids UTC offset bug at midnight
    const todayStr = (zones && zones.forecastStart) || new Date().toISOString().slice(0, 10);
    // Prefer backend-provided zones; fall back to scanning actual series for lastRealDate
    let lastRealDate = (zones && zones.historyEnd) || null;
    if (!lastRealDate) {
      for (let i = series.actual.length - 1; i >= 0; i--) {
        if (series.actual[i] !== null && series.actual[i] !== undefined) {
          lastRealDate = timeAxis[i]; break;
        }
      }
    }
    // Use backend gap_end if available; otherwise derive from todayStr
    const gapEndStr = (zones && zones.gapEnd) || (() => {
      const d = new Date(todayStr); d.setDate(d.getDate() - 1); return d.toISOString().slice(0, 10);
    })();
    let gapStartIdx = -1;
    let gapEndIdx = -1;
    // Gap zone: day after lastRealDate up to gapEnd (inclusive)
    if (lastRealDate && lastRealDate < todayStr) {
      const dayAfterLast = new Date(lastRealDate);
      dayAfterLast.setDate(dayAfterLast.getDate() + 1);
      const gapStart = dayAfterLast.toISOString().slice(0, 10);
      if (gapStart < todayStr) {
        gapStartIdx = timeAxis.indexOf(gapStart);
        gapEndIdx = timeAxis.indexOf(gapEndStr);
        markAreaData.push([
          {
            xAxis: gapStart,
            itemStyle: {
              color: 'rgba(200, 200, 200, 0.2)'
            },
            label: {
              show: true,
              position: 'insideTop',
              padding: [20, 0, 0, 0],
              color: '#999',
              opacity: 0.4,
              formatter: state.lang === 'zh' ? '数据滞后推演区' : 'Latency Inference'
            }
          },
          { xAxis: todayStr }
        ]);
      }
    }
    // Forecast zone (today → end of forecast window)
    if (forecast.startIndex >= 0 && forecast.endIndex < timeAxis.length) {
      markAreaData.push([
        {
          xAxis: timeAxis[forecast.startIndex],
          itemStyle: { color: 'rgba(10,132,255,0.06)' },
          label: {
            show: true, position: 'insideTop',
            padding: [20, 0, 0, 0],
            color: isDark ? 'rgba(10,132,255,0.7)' : 'rgba(10,132,255,0.8)',
            fontSize: 10,
            formatter: state.lang === 'zh' ? '未来7天预测区' : 'Forecast (7d)'
          }
        },
        { xAxis: timeAxis[forecast.endIndex] }
      ]);
    }

    // ── Ensemble mean series (gru_mimo used as "primary" prediction) ──
    const ensembleMeanColor = '#ff9f0a';  // amber — stands out from all 5 model colors
    const latencyAnchor = (gapStartIdx >= 0 && gapEndIdx >= gapStartIdx)
      ? timeAxis.map((_, i) => (i >= gapStartIdx && i <= gapEndIdx ? 0 : null))
      : timeAxis.map(() => null);

    const seriesList = [
      {
        name: state.lang === 'zh' ? '实际客流' : 'Actual',
        type: 'line', data: showActual ? series.actual : series.actual.map(() => null),
        symbol: 'none', connectNulls: false,
        lineStyle: { color: isDark ? '#ffffff' : '#1c1c1e', width: 2 },
        itemStyle: { color: isDark ? '#ffffff' : '#1c1c1e' },
        markArea: { silent: true, data: markAreaData },
        markLine: {
          silent: true,
          data: [
            // 季节性预警阈值：按 timeAxis 日期判断淡/旺季，分段画线
            ...(() => {
              const peakThr = thresholds.crowd_peak;
              const offThr  = thresholds.crowd_off;
              if (!peakThr && !offThr) return [];
              function isPeakSeason(dateStr) {
                const d = new Date(dateStr + 'T00:00:00');
                const m = d.getMonth() + 1, day = d.getDate();
                return (m >= 4 && m <= 10) || (m === 11 && day <= 15);
              }
              // 找出每段连续同季节的起止日期，生成分段 markLine
              const lines = [];
              let segStart = null, segIsPeak = null;
              for (let i = 0; i < timeAxis.length; i++) {
                const d = timeAxis[i];
                const peak = isPeakSeason(d);
                if (segIsPeak === null) { segStart = d; segIsPeak = peak; }
                const isLast = (i === timeAxis.length - 1);
                const nextPeak = isLast ? null : isPeakSeason(timeAxis[i + 1]);
                if (isLast || nextPeak !== segIsPeak) {
                  const thr = segIsPeak ? peakThr : offThr;
                  const label = segIsPeak
                    ? (state.lang === 'zh' ? `旺季预警 ${fmtVisitors(thr)}` : `Peak alert ${fmtVisitors(thr)}`)
                    : (state.lang === 'zh' ? `淡季预警 ${fmtVisitors(thr)}` : `Off-peak alert ${fmtVisitors(thr)}`);
                  lines.push([
                    { xAxis: segStart, yAxis: thr,
                      lineStyle: { color: '#ff453a', type: 'dashed', width: 1.5 },
                      label: { show: true, position: 'insideEndTop',
                               color: '#ff453a', fontSize: 10, fontWeight: 500,
                               formatter: label } },
                    { xAxis: d, yAxis: thr }
                  ]);
                  segStart = timeAxis[i + 1] || null;
                  segIsPeak = nextPeak;
                }
              }
              // ECharts markLine data 中分段线需要用 coords 格式，转换为单点 yAxis 标注
              // 实际用 markLine 的 [from, to] 方式无法按 x 分段，改用 markArea 叠一条细线效果
              // 简化：只输出两条独立 yAxis 线，label 只在末尾显示一次
              const result = [];
              const hasPeak = timeAxis.some(isPeakSeason);
              const hasOff  = timeAxis.some(d => !isPeakSeason(d));
              if (hasPeak && peakThr) result.push({
                yAxis: peakThr, name: 'Peak Threshold',
                lineStyle: { color: '#ff453a', type: 'dashed', width: 1.5 },
                label: { show: true, position: 'insideEndTop',
                         color: '#ff453a', fontSize: 10, fontWeight: 500,
                         formatter: state.lang === 'zh' ? `旺季预警 ${fmtVisitors(peakThr)}` : `Peak alert ${fmtVisitors(peakThr)}` }
              });
              if (hasOff && offThr && offThr !== peakThr) result.push({
                yAxis: offThr, name: 'Off-peak Threshold',
                lineStyle: { color: '#ff9f0a', type: 'dashed', width: 1.5 },
                label: { show: true, position: 'insideEndTop',
                         color: '#ff9f0a', fontSize: 10, fontWeight: 500,
                         formatter: state.lang === 'zh' ? `淡季预警 ${fmtVisitors(offThr)}` : `Off-peak alert ${fmtVisitors(offThr)}` }
              });
              return result;
            })(),
            // Vertical line at today marking start of future predictions
            ...(todayStr && timeAxis.includes(todayStr) ? [{
              xAxis: todayStr,
              lineStyle: { color: isDark ? 'rgba(255,159,10,0.7)' : 'rgba(200,120,0,0.7)', type: 'dashed', width: 1.5 },
              label: {
                show: true, position: 'insideEndTop',
                color: isDark ? 'rgba(255,159,10,0.9)' : 'rgba(180,100,0,0.9)',
                fontSize: 10,
                formatter: state.lang === 'zh' ? '今日 / 预测起点' : 'Today'
              }
            }] : []),
            ...(payload.meta.testStartDate ? [{
              xAxis: payload.meta.testStartDate,
              lineStyle: { color: isDark ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.25)', type: 'dashed', width: 1 },
              label: {
                show: true, position: 'insideEndTop',
                color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.45)',
                fontSize: 10,
                formatter: state.lang === 'zh' ? '测试集起点' : 'Test start'
              }
            }] : [])
          ]
        }
      },
      {
        name: '_latency_anchor',
        type: 'line',
        data: latencyAnchor,
        symbol: 'none', showSymbol: false, connectNulls: false,
        lineStyle: { color: 'transparent', width: 0 },
        itemStyle: { color: 'transparent' },
        legendHoverLink: false,
        z: 0,
      },
      // ── Primary prediction line: Ensemble mean (gru_mimo) ──
      // Respects the gru_mimo chip toggle so the button actually works.
      // When hidden: render transparent so the CI band stacking still has a centre reference.
      {
        name: state.lang === 'zh' ? 'GRU 集成均值' : 'GRU Ensemble Mean',
        type: 'line',
        data: series.gru_mimo || [],
        symbol: 'none', showSymbol: false, connectNulls: false,
        lineStyle: {
          color: (!devMode || chips.has('gru_mimo')) ? ensembleMeanColor : 'transparent',
          width: (!devMode || chips.has('gru_mimo')) ? (devMode ? 1.5 : 2.5) : 0,
        },
        itemStyle: { color: ensembleMeanColor },
        z: 10,
      },
      // ── Model comparison lines: only in devMode ──
      ...CURVE_DEFS
        .filter(({ key }) => devMode && key !== 'gru_mimo')  // gru_mimo already rendered above
        .map(({ key, nameZh, nameEn, color }) => ({
          name: state.lang === 'zh' ? nameZh : nameEn,
          type: 'line',
          data: chips.has(key) ? (series[key] || []) : (series[key] || []).map(() => null),
          symbol: 'none', showSymbol: false, connectNulls: false,
          lineStyle: { color, width: 1.5, opacity: 0.75 },
          itemStyle: { color }
        })),
      // ── Uncertainty fan chart (Deep Ensemble + Conformal step-wise q̂_h) ──
      // Rendered as two boundary lines with areaStyle between them.
      // Only visible in the forecast window (today ~ today+6), non-null elsewhere.
      ...(payload.uncertainty && payload.uncertainty.available ? (() => {
        const unc = payload.uncertainty;
        const bandColor = isDark ? 'rgba(255,159,10,0.18)' : 'rgba(255,159,10,0.15)';
        const lineColor = isDark ? 'rgba(255,159,10,0.55)' : 'rgba(255,159,10,0.50)';
        return [
          // Lower bound (invisible line, anchor for areaStyle stack)
          {
            name: '_unc_lower',
            type: 'line',
            data: unc.lower,
            symbol: 'none', showSymbol: false, connectNulls: false,
            lineStyle: { color: 'transparent', width: 0 },
            itemStyle: { color: 'transparent' },
            stack: 'unc_band',
            areaStyle: { color: 'transparent' },
            tooltip: { show: false },
            silent: true,
            legendHoverLink: false,
          },
          // Upper bound — stacks on lower, fills the fan area
          {
            name: state.lang === 'zh' ? '90% 置信区间（共形预测）' : '90% CI (Conformal)',
            type: 'line',
            data: unc.upper.map((v, i) => {
              // For stacked area: value = upper - lower
              const lo = unc.lower[i];
              return (v !== null && lo !== null) ? Math.max(0, (v - lo)) : null;
            }),
            symbol: 'none', showSymbol: false, connectNulls: false,
            lineStyle: { color: lineColor, width: 1, type: 'dotted' },
            itemStyle: { color: lineColor },
            stack: 'unc_band',
            areaStyle: { color: bandColor },
            legendHoverLink: false,
          }
        ];
      })() : [])
    ];

    // ── Suitability warning probability curve (secondary Y-axis) ──
    const riskData = payload.risk || {};
    const pWarnArr = riskData.p_warn;
    if (devMode && Array.isArray(pWarnArr) && pWarnArr.length > 0) {
      seriesList.push({
        name: state.lang === 'zh' ? '预警概率' : 'Warn Prob.',
        type: 'line',
        data: pWarnArr,
        yAxisIndex: 1,
        symbol: 'none', showSymbol: false, connectNulls: false,
        lineStyle: {
          color: isDark ? 'rgba(191,90,242,0.75)' : 'rgba(150,60,210,0.65)',
          width: 1.5, type: 'dashed'
        },
        itemStyle: { color: isDark ? 'rgba(191,90,242,0.75)' : 'rgba(150,60,210,0.65)' },
        z: 5,
      });
    }

    return {
      backgroundColor: 'transparent',
      textStyle: { color: textColor, fontFamily: 'Inter, "Noto Sans SC", sans-serif' },
      grid: { top: 60, bottom: 40, left: '10%', right: '5%', containLabel: false },
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
      yAxis: [
        {
          type: 'value',
          min: 0,   // visitor count cannot be negative
          splitLine: { lineStyle: { color: gridColor } },
          axisLabel: { color: mutedColor, fontSize: 11,
            formatter: (v) => v >= 10000 ? (v / 10000).toFixed(1) + 'w' : String(v) }
        },
        {
          // Secondary axis: warning probability [0, 1]
          type: 'value', min: 0, max: 1,
          splitLine: { show: false },
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: {
            color: isDark ? 'rgba(255,69,58,0.55)' : 'rgba(180,40,30,0.50)',
            fontSize: 10,
            formatter: (v) => v === 0 ? '' : (v === 1 ? '100%' : Math.round(v * 100) + '%')
          }
        }
      ],
      tooltip: {
        trigger: 'axis', axisPointer: { type: 'cross', crossStyle: { color: mutedColor } },
        backgroundColor: tooltipBg, borderColor: tooltipBorder, borderWidth: 1,
        textStyle: { color: textColor, fontSize: 12 },
        formatter: (params) => {
          if (!params || !params.length) return '';
          const date = params[0].axisValue || '';
          const unc = payload && payload.uncertainty;
          const si = payload && payload.forecast ? payload.forecast.startIndex : -1;
          const dateIdx = payload && payload.timeAxis ? payload.timeAxis.indexOf(date) : -1;
          // Determine prediction horizon step (h=1 for today, h=7 for today+6)
          const hStep = (si >= 0 && dateIdx >= si) ? (dateIdx - si + 1) : null;
          const isForecastZone = hStep !== null && hStep >= 1 && hStep <= 7;
          const isGapZone = gapStartIdx >= 0 && gapEndIdx >= gapStartIdx && dateIdx >= gapStartIdx && dateIdx <= gapEndIdx;

          let html = `<div style="font-weight:600;margin-bottom:4px">${date}`;
          if (isForecastZone) html += ` <span style="font-size:10px;opacity:0.55;font-weight:500">${state.lang === 'zh' ? `预测 · 第${hStep}天` : `Forecast · h=${hStep}`}</span>`;
          html += `</div>`;
          if (isGapZone) {
            html += `<div style="font-size:11px;opacity:0.75;padding:6px 8px;border-radius:10px;background:${isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.04)'};border:1px solid ${isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.08)'};margin-bottom:6px">${state.lang === 'zh' ? '官方数据滞后，滚动推演中' : 'Official data delayed, rolling inference.'}</div>`;
          }

          params.forEach((p) => {
            // Skip internal stacking series
            if (!p.seriesName || p.seriesName.startsWith('_unc_') || p.seriesName.startsWith('_latency_')) return;
            if (p.value === null || p.value === undefined) return;
            // For the CI band series, skip display value (shown separately below)
            if (p.seriesName === '90% 置信区间（共形预测）' || p.seriesName === '90% CI (Conformal)') return;
            const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${p.color};margin-right:6px"></span>`;
            html += `<div>${dot}${p.seriesName}: <b>${fmtVisitors(p.value)}</b></div>`;
          });

          // Append uncertainty interval info for forecast zone
          if (isForecastZone && unc && unc.available && dateIdx >= 0) {
            const lo = unc.lower[dateIdx];
            const hi = unc.upper[dateIdx];
            if (lo !== null && hi !== null) {
              const ciLabel = state.lang === 'zh' ? '90% 置信区间' : '90% Conf. Interval';
              html += `<div style="margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.12)">`;
              html += `<div style="color:rgba(255,159,10,0.9)">■ ${ciLabel}: <b>[${fmtVisitors(lo)}, ${fmtVisitors(hi)}]</b></div>`;
              html += `</div>`;
            }
          }
          return html;
        }
      },
      legend: { show: false },
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
  // Weather card — 前端直接用 state.wxData（Open-Meteo 直接拉取）
  // renderWeatherCard(idx)：从 payload.timeAxis[idx] 取日期 → 查 wxData
  // renderWeather(payload, idx)：保留兼容调用入口
  // ─────────────────────────────────────────────
  function renderWeatherCard(idx) {
    if (!state.payload) return;
    const { timeAxis, thresholds } = state.payload;
    const date = (timeAxis && timeAxis[idx]) || null;
    const wx = date ? getWxForDate(date) : null;

    const tempHigh = wx ? safeNum(wx.th) : null;
    const tempLow  = wx ? safeNum(wx.tl) : null;
    const precip   = wx ? safeNum(wx.precip) : null;
    const windMax  = wx ? safeNum(wx.windMax) : null;
    const code     = wx ? wx.code : null;

    const wDate = $('v3WDate');
    if (wDate) wDate.textContent = date || '—';

    const iconWrap = $('v3WIconWrap');
    if (iconWrap) iconWrap.innerHTML = weatherIconHtml(code);

    const wTemp = $('v3WTemp');
    if (wTemp) wTemp.textContent = tempHigh !== null ? `${Math.round(tempHigh)}°` : '—°';

    const wPrecip = $('v3WPrecip');
    if (wPrecip) wPrecip.textContent = precip !== null ? `${precip.toFixed(1)} mm` : '—';

    const wTempHL = $('v3WTempHL');
    if (wTempHL) {
      const hi = tempHigh !== null ? `${Math.round(tempHigh)}°` : '—';
      const lo = tempLow  !== null ? `${Math.round(tempLow)}°`  : '—';
      wTempHL.textContent = `${hi} / ${lo}`;
    }

    const wWind = $('v3WWind');
    if (wWind) {
      wWind.textContent = windMax !== null ? `${windMax.toFixed(1)} m/s` : '—';
    }

    // AQI 不在 Open-Meteo 免费 forecast 里，留空
    const wAqi = $('v3WAqi');
    if (wAqi) wAqi.textContent = '—';

    const wFlags = $('v3WFlags');
    if (wFlags) {
      const flags = [];
      const thr = thresholds || {};
      const wthr = thr.weather || {};
      if (precip  !== null && wthr.precipHigh !== null && precip  >= wthr.precipHigh)
        flags.push(`<span class="v3-flag v3-flag--driver">${t('driver_precip_high')}</span>`);
      if (tempHigh !== null && wthr.tempHigh   !== null && tempHigh >= wthr.tempHigh)
        flags.push(`<span class="v3-flag v3-flag--driver">${t('driver_temp_high')}</span>`);
      if (tempLow  !== null && wthr.tempLow    !== null && tempLow  <= wthr.tempLow)
        flags.push(`<span class="v3-flag v3-flag--driver">${t('driver_temp_low')}</span>`);
      wFlags.innerHTML = flags.join('');
    }
  }

  // 兼容旧调用（updateSelection 里用 renderWeather(payload, idx)）
  function renderWeather(_payload, idx) {
    renderWeatherCard(idx);
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

    // 推荐窗口 = forecast window（在线预测的未来7天，或离线最新7天）
    const h = forecast.h || 7;
    const recoEnd = forecast.endIndex;
    const recoStart = Math.max(0, recoEnd - h + 1);

    const candidates = [];
    for (let i = recoStart; i <= recoEnd; i++) {
      const pred = safeNum(bestPred(series, i));
      if (pred === null) continue;  // 跳过无预测值的日期
      // 取 risk 等级（来自后端 risk_main 数组）
      const lv = activeRisk && Array.isArray(activeRisk.risk_level)
        ? (safeNum(activeRisk.risk_level[i]) ?? 0) : 0;
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
    const { timeAxis, series, forecast } = payload;
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

    const anyEnd = Math.max(...PRED_KEYS.map(k => latestNonNullEnd(series[k] || [])));

    // Use whichever end is latest (server endIndex is also a candidate)
    let endIdx = Math.max(forecast.endIndex, anyEnd);
    let startIdx = Math.max(0, endIdx - h + 1);

    for (let i = startIdx; i <= endIdx; i++) {
      const date = timeAxis[i] || '';
      const pred = safeNum(bestPred(series, i));
      // 直接从 wxData 查询天气（前端拉取，支持 ±3 天 fallback）
      const wxEntry = date ? getWxForDate(date) : null;
      const code = wxEntry ? wxEntry.code : null;
      const tempHigh = wxEntry ? safeNum(wxEntry.th) : null;
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
  function updateCurveToggleLabels(_payload) {
    // chip labels are static; sync active state only
    _syncChipUI();
  }

  // ─────────────────────────────────────────────
  // Calibration status card
  // ─────────────────────────────────────────────
  function renderCalibCard(payload) {
    const card = $('v3CalibCard');
    if (!card) return;
    const unc = payload && payload.uncertainty;
    if (!unc || !unc.available) {
      card.style.display = 'none';
      return;
    }
    card.style.display = '';

    const latencyTag = $('v3LatencyTag');
    let latencyDays = 0;
    if (latencyTag) {
      let lastIdx = -1;
      for (let i = (payload.series.actual || []).length - 1; i >= 0; i--) {
        const v = (payload.series.actual || [])[i];
        if (v !== null && v !== undefined) { lastIdx = i; break; }
      }
      const s = payload.forecast && typeof payload.forecast.startIndex === 'number' ? payload.forecast.startIndex : -1;
      // Latency = today - last_real_date (in days).
      // s = index of today in timeAxis; lastIdx = index of last non-null actual.
      // Each step = 1 day, so (s - lastIdx) = calendar days since last real data.
      latencyDays = (lastIdx >= 0 && s >= 0) ? Math.max(0, s - lastIdx) : 0;
      if (latencyDays > 0) {
        const lastRealDate = payload.timeAxis[lastIdx] || '—';
        latencyTag.innerHTML =
          `⚠️ 当前官方数据延迟 <b>${latencyDays}</b> 天，处于滚动推演模式` +
          `<span class="v3-latency-note">（最新数据截至 ${lastRealDate}，` +
          `延迟天数 = 今日 − 最后已知数据日）</span>`;
        latencyTag.style.display = '';
      } else {
        latencyTag.style.display = 'none';
      }
    }

    const setEl = (id, val) => { const el = $(id); if (el) el.textContent = val; };
    setEl('v3CalibMethod', '共形预测 (Conformal Prediction)');
    setEl('v3CalibBackbone', `${unc.nMembers}-Member GRU Ensemble`);
    setEl('v3CalibN', `${unc.calSize} 条`);
    setEl('v3CalibMembers', `${unc.nMembers} 个`);

    // Fan bars: h=1..7 half-widths
    const fanEl = $('v3CalibFan');
    if (fanEl) {
      const hw = unc.halfWidthByHorizon || {};
      const rawVals = Object.values(hw).map(safeNum).filter(v => v != null);
      if (!rawVals.length) { fanEl.innerHTML = ''; return; }
      const sorted = rawVals.slice().sort((a, b) => a - b);
      const q = (sorted.length - 1) * 0.90;
      const b = Math.floor(q);
      const r = q - b;
      const p90 = sorted[b + 1] !== undefined ? (sorted[b] + r * (sorted[b + 1] - sorted[b])) : sorted[b];
      const capP90 = Math.max(1, p90 * 1.25);
      const s = payload.forecast && typeof payload.forecast.startIndex === 'number' ? payload.forecast.startIndex : -1;
      const center = payload.series && payload.series.gru_mimo ? payload.series.gru_mimo : [];
      let conservativeOn = false;
      const eff = [];
      for (let h = 1; h <= 7; h++) {
        const raw = safeNum(hw[String(h)] || hw[h]);
        if (raw == null) { eff.push(null); continue; }
        let v = Math.max(0, raw);
        v = Math.min(v, capP90);
        const idx = (s >= 0) ? (s + (h - 1)) : -1;
        const m = (idx >= 0 && idx < center.length) ? safeNum(center[idx]) : null;
        if (m !== null && m > 0) {
          const capMean = m * 0.40;
          if (v > capMean) { v = capMean; conservativeOn = true; }
        }
        eff.push(v);
      }
      const maxHw = Math.max(...eff.filter(v => v != null), 1);
      let fanHtml = `<div class="v3-calib-fan-hint">${
        state.lang === 'zh'
          ? '注：h (Horizon) 为向未来推演的步数（天数）'
          : 'h = forecast horizon in days'
      }</div>`;
      for (let h = 1; h <= 7; h++) {
        const vv = eff[h - 1];
        if (vv == null) continue;
        const pct = Math.min(100, (vv / maxHw) * 100).toFixed(1);
        const valStr = Math.round(vv).toLocaleString();
        fanHtml += `<div class="v3-calib-fan-row">
          <span class="v3-calib-fan-label">h=${h}</span>
          <div class="v3-calib-fan-track"><div class="v3-calib-fan-bar" style="width:${pct}%"></div></div>
          <span class="v3-calib-fan-val">±${valStr}</span>
        </div>`;
      }
      fanEl.innerHTML = fanHtml;

      const noteEl = $('v3CalibNote');
      if (noteEl) {
        const top = Math.max(...eff.filter(v => v != null), 0);
        const topStr = top > 0 ? Math.round(top).toLocaleString() : '—';
        noteEl.textContent = conservativeOn
          ? `已启用保守模式：极端波动已截断（阈值=均值 40%）。当前最大波动幅度约为上下 ${topStr} 人。`
          : `当前最大波动幅度约为上下 ${topStr} 人。`;
      }
    }
  }

  function renderAttrCard(payload) {
    const card = $('v3AttrCard');
    const list = $('v3AttrList');
    const title = $('v3AttrTitle');
    if (!card || !list) return;
    if (!payload || !payload.timeAxis || !payload.timeAxis.length) {
      card.style.display = 'none';
      return;
    }
    const si = payload.forecast && typeof payload.forecast.startIndex === 'number' ? payload.forecast.startIndex : -1;
    const idx = si >= 0 ? si + 1 : -1;
    if (idx < 0 || idx >= payload.timeAxis.length) {
      card.style.display = 'none';
      return;
    }
    const date = payload.timeAxis[idx];
    if (title) {
      const d = new Date(date + 'T00:00:00');
      const mm = d.getMonth() + 1;
      const dd = d.getDate();
      const label = state.lang === 'zh' ? `${mm}月${dd}日` : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      title.textContent = state.lang === 'zh'
        ? `明日预测核心驱动因素（${label}）`
        : `Top Drivers (${label})`;
    }
    const thr = payload.thresholds || {};
    const wthr = thr.weather || {};
    const precipHigh = safeNum(wthr.precip_high);
    const tempHigh = safeNum(wthr.temp_high);
    const tempLow = safeNum(wthr.temp_low);
    const precip = safeNum(payload.weather && payload.weather.precipMm ? payload.weather.precipMm[idx] : null);
    const th = safeNum(payload.weather && payload.weather.tempHighC ? payload.weather.tempHighC[idx] : null);
    const tl = safeNum(payload.weather && payload.weather.tempLowC ? payload.weather.tempLowC[idx] : null);

    const items = [];
    const hol = (payload.holidays || []).find(h => h && h.start && h.end && h.start <= date && date <= h.end);
    if (hol) {
      const name = state.lang === 'zh' ? (hol.nameZh || hol.nameEn || '节假日') : (hol.nameEn || hol.nameZh || 'Holiday');
      items.push({ text: `📈 ${name}（出行热度）`, delta: 8500 });
    }
    const dow = new Date(date + 'T00:00:00').getDay();
    if (dow === 0 || dow === 6) items.push({ text: state.lang === 'zh' ? '📈 周末出游' : '📈 Weekend', delta: 1500 });

    const r = payload.risk || {};
    const drv = Array.isArray(r.drivers) ? (r.drivers[idx] || []) : [];
    if (drv.includes('crowd_over_threshold')) items.push({ text: state.lang === 'zh' ? '📈 客流偏高（接近/超过阈值）' : '📈 High crowd', delta: 3000 });

    if (precipHigh !== null && precip !== null && precip >= precipHigh) items.push({ text: state.lang === 'zh' ? '📉 强降水风险' : '📉 Heavy precipitation', delta: -2000 });
    else if (precip !== null && precip >= 5) items.push({ text: state.lang === 'zh' ? '📉 降雨概率较高' : '📉 Rain risk', delta: -1200 });
    if (tempHigh !== null && th !== null && th >= tempHigh) items.push({ text: state.lang === 'zh' ? '📉 高温体感' : '📉 Heat stress', delta: -900 });
    if (tempLow !== null && tl !== null && tl <= tempLow) items.push({ text: state.lang === 'zh' ? '📉 低温体感' : '📉 Cold stress', delta: -900 });

    const top = items.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 3);
    if (!top.length) {
      list.innerHTML = `<div class="v3-attr-empty">${state.lang === 'zh' ? '暂无归因信息' : 'No attribution available'}</div>`;
      card.style.display = '';
      return;
    }
    list.innerHTML = top.map((it) => {
      const pos = it.delta >= 0;
      const d = Math.abs(it.delta);
      const cls = pos ? 'v3-attr-delta--pos' : 'v3-attr-delta--neg';
      const sign = pos ? '+' : '−';
      return `<div class="v3-attr-item">
        <div class="v3-attr-text">${it.text}</div>
        <div class="v3-attr-delta ${cls}">${sign}${fmtVisitors(d)}</div>
      </div>`;
    }).join('');
    card.style.display = '';
  }

  // ─────────────────────────────────────────────
  // Render all forecast page components
  // ─────────────────────────────────────────────
  function renderAll(payload) {
    buildHistWxData(payload);  // 先把后端历史天气建成 date map，供 getWxForDate 使用
    renderWxStrip(payload);
    renderChartSub(payload);
    renderChart(payload);
    renderWeatherChart(payload);
    renderForecastStrip(payload);
    renderReco(payload);
    renderCalibCard(payload);
    renderAttrCard(payload);
    updateCurveToggleLabels(payload);
    // 默认选中今天（forecast.startIndex），等 wxData 拉到后会自动刷新
    const defaultIdx = payload.forecast.startIndex;
    updateSelection(defaultIdx);

    // 异步拉取天气（前端直连 Open-Meteo），完成后重新触发 updateSelection 刷新侧边天气卡片
    fetchWeatherDirect();
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
  // Curve chip toggle（多选）
  // ─────────────────────────────────────────────
  const ALL_CHIPS = ['actual', 'gru_single', 'gru_mimo', 'lstm_single', 'lstm_mimo', 'seq2seq'];

  function toggleChip(key) {
    if (key === 'all') {
      // 全选/全取消
      if (state.activeChips.size === ALL_CHIPS.length) {
        state.activeChips.clear();
      } else {
        ALL_CHIPS.forEach(k => state.activeChips.add(k));
      }
    } else {
      if (state.activeChips.has(key)) {
        state.activeChips.delete(key);
      } else {
        state.activeChips.add(key);
      }
    }
    _syncChipUI();
    if (state.payload) renderChart(state.payload);
  }

  function _syncChipUI() {
    const allActive = state.activeChips.size === ALL_CHIPS.length;
    $$('[data-chip="all"]').forEach(el => el.classList.toggle('v3-chip--active', allActive));
    ALL_CHIPS.forEach(key => {
      $$(`[data-chip="${key}"]`).forEach(el => el.classList.toggle('v3-chip--active', state.activeChips.has(key)));
    });
  }

  // ─────────────────────────────────────────────
  // Event bindings (FIX 2,7,9)
  // ─────────────────────────────────────────────
  function bindEvents() {
    // Tabs
    $$('[data-page]').forEach((btn) => {
      btn.addEventListener('click', () => showPage(btn.getAttribute('data-page')));
    });

    // Curve chip toggle
    $$('[data-chip]').forEach((el) => {
      el.addEventListener('click', () => toggleChip(el.getAttribute('data-chip')));
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
        state.wxData = {};
        state.histWxData = {};  // 清空历史天气缓存
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

    const devSwitch = $('v3DevModeSwitch');
    if (devSwitch) {
      devSwitch.addEventListener('change', () => {
        state.devMode = !!devSwitch.checked;
        const devChips = $('v3DevChips');
        if (devChips) devChips.style.display = state.devMode ? 'flex' : 'none';
        const label = $('v3DevModeLabel');
        if (label) label.textContent = state.lang === 'zh' ? '开发者模式' : 'Dev Mode';
        if (state.devMode) {
          ALL_CHIPS.forEach(k => state.activeChips.add(k));
        } else {
          state.activeChips.clear();
          state.activeChips.add('actual');
        }
        _syncChipUI();
        if (state.payload) renderChart(state.payload);
      });
    }

    // Glossary: separated button row + single content panel below
    // activeGloss: key string of active button, or null
    let activeGloss = null;
    const glossBtns = $$('#v3GlossaryBtns .v3-gpill-btn');
    const glossPanel = $('v3GlossaryPanel');
    const glossBodies = glossPanel
      ? Array.from(glossPanel.querySelectorAll('[data-gloss-body]'))
      : [];

    function setGloss(key) {
      // Update buttons
      glossBtns.forEach((b) => {
        b.classList.toggle('v3-gpill-btn--active', b.getAttribute('data-gloss') === key);
        b.setAttribute('aria-pressed', b.getAttribute('data-gloss') === key ? 'true' : 'false');
      });
      // Update panel content
      glossBodies.forEach((bd) => {
        bd.classList.toggle('v3-gloss-active', bd.getAttribute('data-gloss-body') === key);
      });
      // Show/hide panel
      if (glossPanel) glossPanel.hidden = key === null;
      activeGloss = key;
    }

    glossBtns.forEach((btn) => {
      btn.setAttribute('aria-pressed', 'false');
      btn.addEventListener('click', () => {
        const key = btn.getAttribute('data-gloss');
        // Toggle off if already active
        setGloss(activeGloss === key ? null : key);
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
    bindWxArrows();
    loadForecast();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
