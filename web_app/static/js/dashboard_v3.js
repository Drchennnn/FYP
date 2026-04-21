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
      tab_forecast: '预测',
      chart_title: '客流预测', risk_title: '适宜性预警',
      risk_normal: '正常', risk_watch: '关注', risk_warning: '预警', risk_high: '高风险',
      reco_title: '推荐出行窗口', reco_loading: '加载中…',
      w_precip: '降水', w_temphl: '温度区间', w_wind: '风力', w_aqi: 'AQI',
      status_loading: '加载中', status_source: '数据来源：九寨沟官网 · Open-Meteo',
      online_mode: '在线预测',
      risk_lv_0: '正常', risk_lv_1: '关注', risk_lv_2: '预警', risk_lv_3: '高风险',
      driver_crowd_over_threshold: '客流超阈值', driver_precip_high: '强降水',
      driver_temp_high: '高温', driver_temp_low: '低温',
      no_data: '暂无数据',
      warn_fallback: '在线预测失败。',
      show_precip: '降水', show_temp: '温度', weather_sub_title: '降水 / 温度',
      reco_reason_low: '预测客流较少，适合出游',
      reco_reason_normal: '客流正常，天气适宜',
      reco_reason_watch: '客流偏多，建议错峰',
      ds_title: '数据集划分',
      ds_sub: '原始数据过滤至 date ≥ 2023-06-01，共 1,050 行；各模型序列样本数因 look_back 偏移略有差异',
      ds_train_label: '训练集  ~80%',
      ds_train_meta: 'Transformer: 2023-07-16 ~ 2025-09-28（805 条）· GRU: 2023-07-01 ~ 2025-09-12（804 条）· XGBoost: 2023-06-01 ~ 2025-09-18（840 条）',
      ds_val_label: '验证集  ~10%',
      ds_val_meta: 'Transformer: 2025-09-29 ~ 2026-01-06（100 条）· GRU: 2025-09-13 ~ 2026-01-04（114 条）· XGBoost: 2025-09-19 ~ 2026-01-01（105 条）',
      ds_test_label: '测试集  ~10%',
      ds_test_meta: 'Transformer: 2026-01-07 ~ 2026-04-16（100 条）· GRU: 2026-01-05 ~ 2026-04-16（102 条）· XGBoost: 2026-01-02 ~ 2026-04-16（105 条）',
      ds_events_cap: '完整客流时序（2016–2026）：地震封闭（红色）与新冠管控（橙色）期间数据结构性断裂，绿色虚线为建模起点',
      ds_note_title: '为何只用 2023 年以来的数据？',
      ds_note_body: '九寨沟于 2017 年 8 月发生地震，景区关闭至 2019 年底；2020–2022 年受新冠疫情反复影响，景区多次限流或封闭。这两段时期的客流模式与政策稳定后的正常运营期存在本质差异，纳入训练会引入分布噪声。2023 年起客流恢复常态、旺淡季规律稳定，因此以 2023-06-01 为建模起点，保证训练数据的同质性与代表性。',
      analysis_title: '模型性能分析',
      analysis_sub: '基于测试集（2026-01-07 ~ 2026-04-16，100 天）· Crowd Alert 以旺淡季动态阈值计算（旺季 32,800 / 淡季 18,400）',
      analysis_row_regression: '回归误差',
      analysis_row_regression_hint: 'MAE 为平均绝对误差（人次/日），NRMSE 为归一化均方根误差，数值越低越好',
      analysis_cap_mae: '集成预测（GRU 10% + Transformer 20% + XGBoost 70%）MAE 最低，达 2,222 人次/日，较表现最差的 GRU（2,853）提升约 22%，较 XGBoost 单模型（2,389）再降 6.9%，验证了加权集成对系统性偏差的修正效果。',
      analysis_cap_nrmse: 'NRMSE 对极端值更敏感，XGBoost 以 0.1176 领先单模型，集成进一步降至 0.1147。GRU 的 NRMSE（0.1557）偏高，说明其在春节/黄金周等客流峰值时段存在较大低估。',
      analysis_row_crowd: '预警性能 Crowd Alert',
      analysis_row_crowd_hint: 'F1 综合精确率与召回率；Recall 反映超载日被正确预警的比例，对景区管理尤为重要；阈值：旺季 32,800 / 淡季 18,400',
      analysis_cap_f1: 'XGBoost（F1=0.842）与 Transformer（F1=0.800）在精确率与召回率之间取得良好平衡；集成（F1=0.842）与 XGBoost 持平。GRU F1 仅 0.533，预警极为保守，大量超载日被漏报。',
      analysis_cap_recall: 'Recall 直接关系景区安全管理：漏报超载日的代价远大于误报。GRU Recall 仅 36.4%，意味着超过六成超载日未被及时预警；集成以 70% 权重偏向 XGBoost（Recall 0.909），将集成 Recall 拉至 0.818，实现高覆盖率。',
      analysis_row_loss: '训练收敛',
      analysis_row_loss_hint: 'Loss = MinMax 缩放后的 MSE，越低说明拟合越精准；Train/Val 曲线同步下降且差距收窄表明无过拟合；Early Stopping 在 Val Loss 连续 20 轮不改善时终止训练',
      analysis_cap_gru_loss: 'GRU 在第 36 轮达到最优 val_loss=0.0082，共训练 56 轮后早停。Train/Val 曲线高度重合，模型泛化良好，无过拟合迹象。GRU 结构简单（单层 64 单元），收敛速度最快。',
      analysis_cap_tf_loss: 'Transformer 在第 49 轮达到最优 val_loss=0.0094，共训练 69 轮后早停（look_back=45）。Val Loss 在第 30 轮后趋于稳定，但多头注意力机制使其对长期季节依赖的捕捉优于 GRU，体现在更高的预警 F1 上。',
      analysis_row_feat: '特征重要性 XGBoost',
      analysis_row_feat_hint: '基于 Gain 指标（每次特征被用于分裂时的平均信息增益）；反映模型对各输入变量的依赖程度',
      analysis_cap_feat: 'is_peak_season（旺淡季标志）以 58.1% 的 Gain 占绝对主导，印证了九寨沟客流的强季节周期性。visitor_count_lag_1（昨日客流）贡献 22.7%，说明短期惯性显著。rolling_7d_mean（7日均值）提供趋势平滑（9.0%）。三者合计 89.8%，气象与节假日特征提供剩余 10.2% 的辅助修正。'
    },
    en: {
      tab_forecast: 'Forecast',
      chart_title: 'Visitor Forecast', risk_title: 'Suitability Warning',
      risk_normal: 'Normal', risk_watch: 'Watch', risk_warning: 'Warning', risk_high: 'High Risk',
      reco_title: 'Best Visit Window', reco_loading: 'Loading…',
      w_precip: 'Precip', w_temphl: 'Temp Range', w_wind: 'Wind', w_aqi: 'AQI',
      status_loading: 'Loading', status_source: 'Source: Jiuzhaigou Official · Open-Meteo',
      online_mode: 'Online forecast',
      risk_lv_0: 'Normal', risk_lv_1: 'Watch', risk_lv_2: 'Warning', risk_lv_3: 'High Risk',
      driver_crowd_over_threshold: 'Crowd over threshold', driver_precip_high: 'Heavy precipitation',
      driver_temp_high: 'High temperature', driver_temp_low: 'Low temperature',
      no_data: 'No data',
      warn_fallback: 'Online forecast failed.',
      show_precip: 'Precip', show_temp: 'Temp', weather_sub_title: 'Precip / Temp',
      reco_reason_low: 'Low predicted crowd, good for visiting',
      reco_reason_normal: 'Normal crowd, suitable weather',
      reco_reason_watch: 'Higher crowd, consider off-peak timing',
      ds_title: 'Dataset Split',
      ds_sub: 'Raw data filtered to date ≥ 2023-06-01 (1,050 rows); sequence count varies slightly by model look_back',
      ds_train_label: 'Train  ~80%',
      ds_train_meta: 'Transformer: 2023-07-16 ~ 2025-09-28 (805) · GRU: 2023-07-01 ~ 2025-09-12 (804) · XGBoost: 2023-06-01 ~ 2025-09-18 (840)',
      ds_val_label: 'Validation  ~10%',
      ds_val_meta: 'Transformer: 2025-09-29 ~ 2026-01-06 (100) · GRU: 2025-09-13 ~ 2026-01-04 (114) · XGBoost: 2025-09-19 ~ 2026-01-01 (105)',
      ds_test_label: 'Test  ~10%',
      ds_test_meta: 'Transformer: 2026-01-07 ~ 2026-04-16 (100) · GRU: 2026-01-05 ~ 2026-04-16 (102) · XGBoost: 2026-01-02 ~ 2026-04-16 (105)',
      ds_events_cap: 'Full visitor flow 2016–2026: earthquake closure (red) and COVID lockdowns (orange) caused structural breaks; green dashed line = modelling start',
      ds_note_title: 'Why use data from 2023 onward only?',
      ds_note_body: 'Jiuzhaigou was closed after the August 2017 earthquake until late 2019. COVID lockdowns (2020–2022) further caused repeated closures and visitor caps. Visitor patterns during these periods are fundamentally different from stable post-recovery operations. Starting from 2023-06-01 ensures the training data is homogeneous and representative of the current operating regime.',
      analysis_title: 'Model Performance Analysis',
      analysis_sub: 'Test set: 2026-01-07 ~ 2026-04-16 (100 days) · Crowd Alert uses seasonal thresholds (peak 32,800 / off-peak 18,400)',
      analysis_row_regression: 'Regression Error',
      analysis_row_regression_hint: 'MAE = mean absolute error (visitors/day); NRMSE = normalised RMSE. Lower is better.',
      analysis_cap_mae: 'Ensemble (GRU 10% + Transformer 20% + XGBoost 70%) achieves the lowest MAE of 2,222 visitors/day — 22% better than GRU (2,853) and 6.9% better than XGBoost alone (2,389), confirming that weighted blending corrects systematic bias across models.',
      analysis_cap_nrmse: 'NRMSE penalises large errors more heavily. XGBoost leads single models at 0.1176; ensemble reduces it to 0.1147. GRU\'s elevated NRMSE (0.1557) reflects consistent under-prediction during peak-season spikes (Spring Festival, Golden Week).',
      analysis_row_crowd: 'Crowd Alert Performance',
      analysis_row_crowd_hint: 'F1 balances precision and recall; Recall = fraction of overload days correctly flagged — the more safety-critical metric; thresholds: peak 32,800 / off-peak 18,400',
      analysis_cap_f1: 'XGBoost (F1=0.842) and Transformer (F1=0.800) balance precision and recall well; ensemble matches XGBoost at F1=0.842. GRU scores only 0.533 — severely under-flagging overload days.',
      analysis_cap_recall: 'Recall directly governs park safety: a missed overload day is far costlier than a false alarm. GRU\'s 36.4% Recall means over 60% of overload days go unflagged. The ensemble\'s 70% weight on XGBoost (Recall 0.909) lifts ensemble Recall to 0.818, achieving high coverage.',
      analysis_row_loss: 'Training Convergence',
      analysis_row_loss_hint: 'Loss = MSE on MinMax-scaled data; lower = better fit. Parallel Train/Val descent with a narrow gap indicates no overfitting. Early Stopping halts training after 20 consecutive epochs without Val Loss improvement.',
      analysis_cap_gru_loss: 'GRU reaches best val_loss=0.0082 at epoch 36; training stopped at epoch 56. Train and Val curves converge closely — strong generalisation with no overfitting. GRU\'s single-layer 64-unit architecture converges fastest among the three models.',
      analysis_cap_tf_loss: 'Transformer reaches best val_loss=0.0094 at epoch 49; training stopped at epoch 69 (look_back=45). Val Loss stabilises after epoch 30, but multi-head attention enables better long-range seasonal dependency capture — reflected in its superior F1 vs GRU.',
      analysis_row_feat: 'XGBoost Feature Importance',
      analysis_row_feat_hint: 'Gain = average information gain each time a feature is used in a split — measures how much each variable drives prediction decisions',
      analysis_cap_feat: 'is_peak_season dominates at 58.1% Gain, confirming Jiuzhaigou\'s strong seasonal periodicity. visitor_count_lag_1 (yesterday\'s count) contributes 22.7%, reflecting strong short-term momentum. rolling_7d_mean adds trend smoothing (9.0%). Together these three account for 89.8%; weather and holiday features provide the remaining 10.2% fine-grained correction.'
    }
  };

  // ─────────────────────────────────────────────
  // State
  // ─────────────────────────────────────────────
  const state = {
    h: 3,
    mode: 'online',
    lang: 'zh',
    theme: 'dark',
    selectedIdx: null,
    payload: null,
    chart: null,
    weatherChart: null,
    wxData: {},
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
        gru: [], transformer: [], xgboost: [], ensemble: []
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
    out.series.gru         = safeArr(s.gru_pred || s.gru_single_pred || [], n, safeNum);
    out.series.transformer = safeArr(s.transformer_pred || [], n, safeNum);
    out.series.xgboost     = safeArr(s.xgboost_pred || [], n, safeNum);
    out.series.ensemble    = safeArr(s.ensemble_pred || [], n, safeNum);

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
    out.risk = r;
    out.warning = raw.warning || null;
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
      // 优先走后端代理（避免中国大陆直连 Open-Meteo 被阻断）
      let json = null;
      try {
        const res = await fetch('/api/weather', { cache: 'no-store' });
        if (res.ok) json = await res.json();
      } catch (_) {}
      // fallback: 直连 Open-Meteo
      if (!json || json.error) {
        const params = new URLSearchParams({
          latitude: '33.2', longitude: '103.9',
          daily: 'temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode,windspeed_10m_max',
          timezone: 'Asia/Shanghai',
          past_days: '14',
          forecast_days: '14'
        });
        const res2 = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`, { cache: 'no-store' });
        if (!res2.ok) return;
        json = await res2.json();
      }
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
    const cacheKey = `v3_forecast_v12_h${state.h}`;
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
          const _xgbSlice = (normalized.series && normalized.series.xgboost)
            ? normalized.series.xgboost.slice(_fc.startIndex, _fc.endIndex + 1)
            : [];
          const _xgbOk = _xgbSlice.some((v) => v !== null && v !== undefined);
          if (normalized.timeAxis && normalized.timeAxis.length > 0 && _wxOk && _xgbOk) {
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

  // ─────────────────────────────────────────────
  // Weather icon
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

  // 单模型 fallback 顺序（按测试集性能排序：XGBoost > Transformer > GRU）
  const PRED_KEYS = ['xgboost', 'transformer', 'gru'];
  // 优先用集成预测；集成无值时按 PRED_KEYS 顺序 fallback
  function bestPred(series, i) {
    const ens = series.ensemble && series.ensemble[i];
    if (ens !== null && ens !== undefined) return ens;
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

    // 4条曲线配置
    const CURVE_DEFS = [
      { key: 'gru',         nameZh: 'GRU',         nameEn: 'GRU',         color: '#0a84ff' },
      { key: 'transformer', nameZh: 'Transformer',  nameEn: 'Transformer', color: '#30d158' },
      { key: 'xgboost',     nameZh: 'XGBoost',      nameEn: 'XGBoost',     color: '#ffd60a' },
      { key: 'ensemble',    nameZh: '集成预测',      nameEn: 'Ensemble',    color: '#bf5af2' },
    ];

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
    let gapStartIdx = -1;
    let gapEndIdx = -1;
    // Gap zone: day after lastRealDate up to today (exclusive)
    // ECharts category axis requires exact string match — find nearest axis date
    function nearestAxisDate(dateStr, after) {
      // after=true: find first timeAxis entry >= dateStr; false: last entry <= dateStr
      if (after) {
        return timeAxis.find(d => d >= dateStr) || null;
      } else {
        let result = null;
        for (const d of timeAxis) { if (d <= dateStr) result = d; else break; }
        return result;
      }
    }
    if (lastRealDate && lastRealDate < todayStr) {
      const dayAfterLast = new Date(lastRealDate + 'T00:00:00');
      dayAfterLast.setDate(dayAfterLast.getDate() + 1);
      const gapStart = dayAfterLast.toISOString().slice(0, 10);
      if (gapStart < todayStr) {
        // Use nearest axis dates so markArea renders even when gap days not in axis
        const axisGapStart = nearestAxisDate(gapStart, true);
        const axisGapEnd = nearestAxisDate(todayStr, false) || lastRealDate;
        gapStartIdx = axisGapStart ? timeAxis.indexOf(axisGapStart) : -1;
        gapEndIdx = axisGapEnd ? timeAxis.indexOf(axisGapEnd) : -1;
        // Draw from lastRealDate to todayStr using exact strings (ECharts interpolates)
        markAreaData.push([
          {
            xAxis: lastRealDate,
            itemStyle: { color: 'rgba(200, 200, 200, 0.2)' },
            label: {
              show: true, position: 'insideTop',
              padding: [20, 0, 0, 0], color: '#999', opacity: 0.4,
              formatter: state.lang === 'zh' ? '数据滞后区' : 'Data Latency'
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
            formatter: state.lang === 'zh' ? '未来3天预测区' : 'Forecast (3d)'
          }
        },
        { xAxis: timeAxis[forecast.endIndex] }
      ]);
    }

    const seriesList = [
      {
        name: state.lang === 'zh' ? '实际客流' : 'Actual',
        type: 'line', data: series.actual,
        symbol: 'none', connectNulls: false,
        lineStyle: { color: isDark ? '#ffffff' : '#1c1c1e', width: 2 },
        itemStyle: { color: isDark ? '#ffffff' : '#1c1c1e' },
        markArea: { silent: true, data: markAreaData },
        markLine: {
          silent: true,
          data: [
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
      // ── model prediction lines ──
      ...CURVE_DEFS.map(({ key, nameZh, nameEn, color }) => ({
        name: state.lang === 'zh' ? nameZh : nameEn,
        type: 'line',
        z: 10,
        data: series[key] || [],
        symbol: 'none', showSymbol: false, connectNulls: false,
        lineStyle: { color, width: 2 },
        itemStyle: { color }
      })),
      // ── 分段阈值线：每个日期对应当天季节的阈值，旺淡季各自只在对应区间显示 ──
      ...(() => {
        const peakThr = thresholds.crowd_peak;
        const offThr  = thresholds.crowd_off;
        if (!peakThr && !offThr) return [];
        function isPeak(dateStr) {
          const d = new Date(dateStr + 'T00:00:00');
          const m = d.getMonth() + 1, day = d.getDate();
          return (m >= 4 && m <= 10) || (m === 11 && day <= 15);
        }
        // 旺季线：只在旺季日期有值，淡季日期为null
        const peakData = timeAxis.map(d => isPeak(d) ? peakThr : null);
        // 淡季线：只在淡季日期有值，旺季日期为null
        const offData  = timeAxis.map(d => isPeak(d) ? null : offThr);
        const result = [];
        if (peakThr && peakData.some(v => v !== null)) result.push({
          name: state.lang === 'zh' ? `旺季预警 ${fmtVisitors(peakThr)}` : `Peak alert ${fmtVisitors(peakThr)}`,
          type: 'line', z: 5,
          data: peakData,
          symbol: 'none', connectNulls: false,
          lineStyle: { color: '#ff453a', type: 'dashed', width: 1.5 },
          itemStyle: { color: '#ff453a' },
          tooltip: { show: false },
          label: { show: false },
          silent: true,
        });
        if (offThr && offData.some(v => v !== null) && offThr !== peakThr) result.push({
          name: state.lang === 'zh' ? `淡季预警 ${fmtVisitors(offThr)}` : `Off-peak alert ${fmtVisitors(offThr)}`,
          type: 'line', z: 5,
          data: offData,
          symbol: 'none', connectNulls: false,
          lineStyle: { color: '#ff9f0a', type: 'dashed', width: 1.5 },
          itemStyle: { color: '#ff9f0a' },
          tooltip: { show: false },
          label: { show: false },
          silent: true,
        });
        return result;
      })()
    ];

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
          min: 0,
          splitLine: { lineStyle: { color: gridColor } },
          axisLabel: { color: mutedColor, fontSize: 11,
            formatter: (v) => v >= 10000 ? (v / 10000).toFixed(1) + 'w' : String(v) }
        }
      ],
      tooltip: {
        trigger: 'axis', axisPointer: { type: 'cross', crossStyle: { color: mutedColor } },
        backgroundColor: tooltipBg, borderColor: tooltipBorder, borderWidth: 1,
        textStyle: { color: textColor, fontSize: 12 },
        formatter: (params) => {
          if (!params || !params.length) return '';
          const date = params[0].axisValue || '';
          const si = payload && payload.forecast ? payload.forecast.startIndex : -1;
          const dateIdx = payload && payload.timeAxis ? payload.timeAxis.indexOf(date) : -1;
          const hStep = (si >= 0 && dateIdx >= si) ? (dateIdx - si + 1) : null;
          const isForecastZone = hStep !== null && hStep >= 1 && hStep <= 3;
          const isGapZone = gapStartIdx >= 0 && gapEndIdx >= gapStartIdx && dateIdx >= gapStartIdx && dateIdx <= gapEndIdx;

          let html = `<div style="font-weight:600;margin-bottom:4px">${date}`;
          if (isForecastZone) html += ` <span style="font-size:10px;opacity:0.55;font-weight:500">${state.lang === 'zh' ? `预测 · 第${hStep}天` : `Forecast · h=${hStep}`}</span>`;
          html += `</div>`;
          if (isGapZone) {
            html += `<div style="font-size:11px;opacity:0.75;padding:6px 8px;border-radius:10px;background:${isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.04)'};border:1px solid ${isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.08)'};margin-bottom:6px">${state.lang === 'zh' ? '官方数据滞后区' : 'Official data delayed.'}</div>`;
          }

          params.forEach((p) => {
            if (!p.seriesName) return;
            if (p.value === null || p.value === undefined) return;
            // 过滤阈值线（不在tooltip里显示）
            if (p.seriesName.includes('预警') || p.seriesName.includes('alert')) return;
            const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${p.color};margin-right:6px"></span>`;
            html += `<div>${dot}${p.seriesName}: <b>${fmtVisitors(p.value)}</b></div>`;
          });
          return html;
        }
      },
      legend: {
        show: false,
        data: [
          state.lang === 'zh' ? '实际客流' : 'Actual',
          ...CURVE_DEFS.map(d => state.lang === 'zh' ? d.nameZh : d.nameEn)
        ]
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
    // 强制恢复可见，避免上一轮 legendUnSelect 导致曲线隐藏
    [state.lang === 'zh' ? '实际客流' : 'Actual', 'GRU', 'Transformer', 'XGBoost', state.lang === 'zh' ? '集成预测' : 'Ensemble']
      .forEach((name) => state.chart.dispatchAction({ type: 'legendSelect', name }));

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

    // 推荐窗口：优先用在线预测未来3天，若未来无数据则回退到最后有预测值的3天
    const h = forecast.h || 3;

    function latestNonNullEndReco(arr) {
      for (let i = arr.length - 1; i >= 0; i--) { if (arr[i] !== null) return i; }
      return -1;
    }
    const anyEndReco = Math.max(...PRED_KEYS.map(k => latestNonNullEndReco(series[k] || [])));
    // 优先用服务端endIndex，若未来窗口全是null则退到最后有数据的索引
    const serverEnd = forecast.endIndex;
    const hasFutureData = PRED_KEYS.some(k => {
      const arr = series[k] || [];
      return arr.slice(Math.max(0, serverEnd - h + 1), serverEnd + 1).some(v => v !== null);
    });
    const recoEnd = hasFutureData ? serverEnd : (anyEndReco >= 0 ? anyEndReco : serverEnd);
    const recoStart = Math.max(0, recoEnd - h + 1);

    const candidates = [];
    for (let i = recoStart; i <= recoEnd; i++) {
      const pred = safeNum(bestPred(series, i));
      if (pred === null) continue;
      const lv = activeRisk && Array.isArray(activeRisk.risk_level)
        ? (safeNum(activeRisk.risk_level[i]) ?? 0) : 0;
      const drivers = activeRisk && Array.isArray(activeRisk.drivers)
        ? (activeRisk.drivers[i] || []) : [];
      candidates.push({ idx: i, date: timeAxis[i], lv, pred, drivers });
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
      // 主要原因文字
      let reason;
      if (c.lv === 0 && threshold !== null && c.pred !== null && c.pred < threshold * 0.7) {
        reason = t('reco_reason_low');
      } else if (c.lv === 0) {
        reason = t('reco_reason_normal');
      } else {
        reason = t('reco_reason_watch');
      }
      // 影响因素标签
      const driverTags = (c.drivers || []).map(k => {
        const label = driverLabel(k);
        return `<span class="v3-reco-driver">${label}</span>`;
      }).join('');
      return `<div class="v3-reco-item" data-idx="${c.idx}">
        <div class="v3-reco-left">
          <span class="v3-reco-date">${c.date}</span>
          <span class="v3-reco-reason">${reason}${driverTags ? ' · ' + driverTags : ''}</span>
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
    const h = forecast.h || 3;

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
  function initLegendToggles() {
    // Map data-series-name → ECharts series name (language-aware)
    function resolveSeriesName(key) {
      if (key === 'actual') return state.lang === 'zh' ? '实际客流' : 'Actual';
      if (key === 'ensemble') return state.lang === 'zh' ? '集成预测' : 'Ensemble';
      return key; // GRU / Transformer / XGBoost — same in both langs
    }
    const btns = document.querySelectorAll('.v3-chart-key__item--toggle');
    btns.forEach((btn) => {
      if (btn._legendBound) return;
      btn._legendBound = true;
      btn.addEventListener('click', () => {
        const key = btn.getAttribute('data-series-name');
        if (!key || !state.chart) return;
        const seriesName = resolveSeriesName(key);
        const active = btn.classList.toggle('v3-chart-key__item--active');
        state.chart.dispatchAction({
          type: active ? 'legendSelect' : 'legendUnSelect',
          name: seriesName
        });
      });
    });
  }

  function renderAll(payload) {
    buildHistWxData(payload);
    renderWxStrip(payload);
    renderChartSub(payload);
    renderChart(payload);
    initLegendToggles();
    renderWeatherChart(payload);
    renderForecastStrip(payload);
    renderReco(payload);
    const defaultIdx = payload.forecast.startIndex;
    updateSelection(defaultIdx);
    fetchWeatherDirect();
  }


  // ─────────────────────────────────────────────
  // Tab navigation
  // ─────────────────────────────────────────────
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
  }

  // ─────────────────────────────────────────────
  // Event bindings
  // ─────────────────────────────────────────────
  function bindEvents() {
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

    // Sub-chart toggles
    ['v3ShowPrecip', 'v3ShowTemp'].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener('change', () => { if (state.payload) renderWeatherChart(state.payload); });
    });

    // Loss curve tab switcher
    document.querySelectorAll('.v3-loss-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.dataset.loss;
        document.querySelectorAll('.v3-loss-tab').forEach(b => b.classList.remove('v3-loss-tab--active'));
        btn.classList.add('v3-loss-tab--active');
        document.querySelectorAll('.v3-analysis__fig--loss').forEach(fig => {
          fig.classList.toggle('v3-analysis__fig--hidden', fig.id !== 'loss' + target.charAt(0).toUpperCase() + target.slice(1));
        });
      });
    });

  }

  // ─────────────────────────────────────────────
  // Init
  // ─────────────────────────────────────────────
  const CURRENT_CACHE_VER = 'v15';
  function init() {
    console.log('dashboard_v3.js loaded');
    // 清除旧版本 forecast 缓存
    try {
      const oldKeys = [];
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.startsWith('v3_forecast_') && !k.startsWith(`v3_forecast_${CURRENT_CACHE_VER}_`)) {
          oldKeys.push(k);
        }
      }
      oldKeys.forEach(k => localStorage.removeItem(k));
      if (oldKeys.length) console.log('[cache] cleared old keys:', oldKeys);
    } catch (_) {}
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
