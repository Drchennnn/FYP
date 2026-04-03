(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const state = {
    source: 'mock',
    forecast: null
  };

  function setActiveSourceButtons() {
    const btnMock = $('btnSourceMock');
    const btnApi = $('btnSourceApi');
    if (!btnMock || !btnApi) return;
    btnMock.classList.toggle('btn-primary', state.source === 'mock');
    btnMock.classList.toggle('btn-outline-primary', state.source !== 'mock');
    btnApi.classList.toggle('btn-primary', state.source === 'api');
    btnApi.classList.toggle('btn-outline-primary', state.source !== 'api');
  }

  function setRowVisible(rowId, visible) {
    const el = $(rowId);
    if (!el) return;
    el.style.display = visible ? '' : 'none';
  }

  function fmt(v, digits) {
    if (v === null || v === undefined) return null;
    if (typeof v === 'number' && Number.isFinite(v)) {
      if (typeof digits === 'number') return v.toFixed(digits);
      return String(v);
    }
    const s = String(v);
    return s.trim() ? s : null;
  }

  function renderWeatherCard(row) {
    if (!row) return;

    $('wTitle').textContent = 'Weather';
    $('wSubtitle').textContent = row.date || '';
    $('wCode').textContent = row.weather_code_en ?? '--';

    const th = fmt(row.temp_high_c, 1);
    const tl = fmt(row.temp_low_c, 1);
    const pm = fmt(row.precip_mm, 1);

    $('wTempHigh').textContent = th ?? '--';
    $('wTempLow').textContent = tl ?? '--';
    $('wPrecip').textContent = pm ?? '--';

    setRowVisible('rowTempHigh', th !== null);
    setRowVisible('rowTempLow', tl !== null);
    setRowVisible('rowPrecip', pm !== null);

    const windParts = [];
    const wl = fmt(row.wind_level);
    const wd = fmt(row.wind_dir_en);
    const wm = fmt(row.wind_max, 1);
    if (wl !== null) windParts.push(`L${wl}`);
    if (wd !== null) windParts.push(String(wd));
    if (wm !== null) windParts.push(`max ${wm}`);
    const wind = windParts.length ? windParts.join(' / ') : null;

    $('wWind').textContent = wind ?? '--';
    setRowVisible('rowWind', wind !== null);

    const aqiV = fmt(row.aqi_value, 0);
    const aqiL = fmt(row.aqi_level_en);
    const aqi = (aqiV !== null || aqiL !== null) ? [aqiV, aqiL].filter((x) => x !== null).join(' ') : null;
    $('wAqi').textContent = aqi ?? '--';
    setRowVisible('rowAqi', aqi !== null);
  }

  async function loadForecast() {
    setActiveSourceButtons();

    if (state.source === 'api') {
      state.forecast = await window.TestShared.fetchForecast({ h: 7, includeAll: 1 });
    } else {
      state.forecast = window.TestShared.getMockForecast(7);
    }

    const timeAxis = state.forecast.time_axis || [];
    if (!timeAxis.length) return;

    const picker = $('datePicker');
    if (picker) {
      // Default: first date in time_axis
      picker.value = timeAxis[0];
    }

    renderByPicker();
  }

  function renderByPicker() {
    const picker = $('datePicker');
    const dateStr = picker?.value;
    const idx = window.TestShared.findIndexByDate(state.forecast?.time_axis, dateStr);

    const row = window.TestShared.pickWeatherByIndex(state.forecast, idx);
    renderWeatherCard(row);
  }

  function bindUI() {
    $('btnSourceMock')?.addEventListener('click', async () => {
      state.source = 'mock';
      await loadForecast();
    });

    $('btnSourceApi')?.addEventListener('click', async () => {
      state.source = 'api';
      try {
        await loadForecast();
      } catch (e) {
        console.error(e);
        alert(String(e.message || e));
        state.source = 'mock';
        await loadForecast();
      }
    });

    $('datePicker')?.addEventListener('change', () => {
      renderByPicker();
    });
  }

  document.addEventListener('DOMContentLoaded', async () => {
    bindUI();
    await loadForecast();
  });
})();
