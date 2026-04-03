/* Shared helpers for test pages.
 * These helpers are intended to be reused later by dashboard_mvp modules.
 */

(function (global) {
  'use strict';

  function toISODate(d) {
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }

  function clamp(n, lo, hi) {
    return Math.max(lo, Math.min(hi, n));
  }

  function findIndexByDate(timeAxis, dateStr) {
    if (!Array.isArray(timeAxis) || !dateStr) return -1;
    return timeAxis.findIndex((d) => String(d) === String(dateStr));
  }

  async function fetchForecast(opts) {
    const h = clamp(parseInt(opts?.h ?? 7, 10) || 7, 1, 14);
    const includeAll = opts?.includeAll ?? 1;
    const mode = opts?.mode ?? 'offline';

    const params = new URLSearchParams();
    params.set('h', String(h));
    params.set('include_all', includeAll ? '1' : '0');
    params.set('mode', String(mode));

    const url = `/api/forecast?${params.toString()}`;
    const resp = await fetch(url, { headers: { 'Accept': 'application/json' } });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Forecast fetch failed (${resp.status}): ${text}`);
    }
    return await resp.json();
  }

  function getMockForecast(h) {
    const horizon = clamp(parseInt(h ?? 7, 10) || 7, 1, 14);

    const base = new Date();
    base.setHours(0, 0, 0, 0);

    const time_axis = [];
    const actual = [];
    const champion_pred = [];
    const runner_pred = [];

    const precip_mm = [];
    const temp_high_c = [];
    const temp_low_c = [];
    const weather_code_en = [];
    const wind_level = [];
    const wind_dir_en = [];
    const wind_max = [];
    const aqi_value = [];
    const aqi_level_en = [];

    const risk_champion = [];

    for (let i = 0; i < horizon; i++) {
      const d = new Date(base.getTime() + i * 86400000);
      const ds = toISODate(d);
      time_axis.push(ds);

      const v = 8000 + Math.round(500 * Math.sin(i / 2));
      actual.push(v);
      champion_pred.push(v + Math.round(200 * Math.cos(i / 2)));
      runner_pred.push(v + Math.round(300 * Math.sin(i / 3)));

      const p = i % 3 === 0 ? 6.5 : (i % 5 === 0 ? null : 0.3);
      precip_mm.push(p);

      const th = 18 + Math.round(4 * Math.sin(i / 3));
      const tl = th - (6 + (i % 3));
      temp_high_c.push(th);
      temp_low_c.push(tl);

      weather_code_en.push(p && p > 3 ? 'rain' : 'clear');

      wind_level.push(i % 4 === 0 ? 5 : 2);
      wind_dir_en.push(i % 2 === 0 ? 'NE' : 'SW');
      wind_max.push(8 + i);

      aqi_value.push(i % 6 === 0 ? null : (55 + i * 2));
      aqi_level_en.push(i % 6 === 0 ? null : (i < 3 ? 'Good' : 'Moderate'));

      const riskScore = clamp(Math.round(30 + i * 8 + (p ? 10 : 0)), 0, 100);
      const riskLevel = riskScore >= 80 ? 'High' : (riskScore >= 50 ? 'Medium' : 'Low');
      const drivers = [];
      if (p && p > 3) drivers.push('Heavy precipitation');
      if (th >= 22) drivers.push('High temperature');
      if (v >= 8400) drivers.push('High crowd baseline');
      if (drivers.length === 0) drivers.push('No strong drivers');

      risk_champion.push({
        date: ds,
        risk_score: riskScore,
        risk_level: riskLevel,
        drivers
      });
    }

    return {
      meta: {
        generated_at: new Date().toISOString(),
        forecast_mode: 'mock'
      },
      time_axis,
      forecast: {
        h: horizon,
        start_index: 0,
        end_index: horizon - 1
      },
      series: {
        actual,
        champion_pred,
        runner_pred
      },
      weather: {
        precip_mm,
        temp_high_c,
        temp_low_c,
        weather_code_en,
        wind_level,
        wind_dir_en,
        wind_max,
        aqi_value,
        aqi_level_en
      },
      risk: {
        champion: risk_champion,
        runner_up: null
      }
    };
  }

  function pickWeatherByIndex(forecastJson, idx) {
    const fx = forecastJson || {};
    const t = fx.time_axis || [];
    if (idx < 0 || idx >= t.length) return null;

    const w = fx.weather || {};

    return {
      date: t[idx],
      precip_mm: w.precip_mm?.[idx],
      temp_high_c: w.temp_high_c?.[idx],
      temp_low_c: w.temp_low_c?.[idx],
      weather_code_en: w.weather_code_en?.[idx],
      wind_level: w.wind_level?.[idx],
      wind_dir_en: w.wind_dir_en?.[idx],
      wind_max: w.wind_max?.[idx],
      aqi_value: w.aqi_value?.[idx],
      aqi_level_en: w.aqi_level_en?.[idx]
    };
  }

  function pickRiskByIndex(forecastJson, idx) {
    const fx = forecastJson || {};
    const t = fx.time_axis || [];
    if (idx < 0 || idx >= t.length) return null;

    const r = fx.risk?.champion;

    // API shape: risk.champion is an object of aligned arrays
    //   { risk_level: [], risk_score: [], drivers: [] ... }
    if (r && !Array.isArray(r) && typeof r === 'object') {
      const driversArr = r.drivers;
      return {
        date: t[idx],
        risk_level: Array.isArray(r.risk_level) ? r.risk_level[idx] : null,
        risk_score: Array.isArray(r.risk_score) ? r.risk_score[idx] : null,
        drivers: Array.isArray(driversArr) ? (driversArr[idx] || []) : []
      };
    }

    // Mock (or future UI shape): array of objects aligned to time_axis
    const row = Array.isArray(r) ? r[idx] : null;
    if (row && typeof row === 'object') {
      return {
        date: t[idx],
        risk_level: row.risk_level ?? row.level ?? null,
        risk_score: row.risk_score ?? row.score ?? null,
        drivers: row.drivers ?? row.reasons ?? []
      };
    }

    return { date: t[idx], risk_level: null, risk_score: null, drivers: [] };
  }

  global.TestShared = {
    toISODate,
    findIndexByDate,
    fetchForecast,
    getMockForecast,
    pickWeatherByIndex,
    pickRiskByIndex
  };
})(window);
