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

  function clamp(n, lo, hi) {
    return Math.max(lo, Math.min(hi, n));
  }

  function renderRisk(row) {
    if (!row) return;

    $('rDate').textContent = row.date || '';
    $('riskLevel').textContent = row.risk_level ?? '--';
    $('riskScore').textContent = (row.risk_score ?? '--');

    const ul = $('riskDrivers');
    if (ul) {
      ul.innerHTML = '';
      const drivers = Array.isArray(row.drivers) ? row.drivers : [];
      if (!drivers.length) {
        const li = document.createElement('li');
        li.textContent = 'No drivers';
        ul.appendChild(li);
      } else {
        for (const d of drivers) {
          const li = document.createElement('li');
          li.textContent = String(d);
          ul.appendChild(li);
        }
      }
    }

    const score = typeof row.risk_score === 'number' ? row.risk_score : null;
    const fill = $('thermoFill');
    const text = $('thermoText');
    if (fill && text) {
      const pct = score === null ? 0 : clamp(score, 0, 100);
      fill.style.height = `${pct}%`;
      text.textContent = score === null ? '--' : `${pct}/100`;
    }
  }

  function renderByPicker() {
    const picker = $('datePicker');
    const dateStr = picker?.value;
    const idx = window.TestShared.findIndexByDate(state.forecast?.time_axis, dateStr);

    const row = window.TestShared.pickRiskByIndex(state.forecast, idx);
    renderRisk(row);
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
      picker.value = timeAxis[0];
    }

    renderByPicker();
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
