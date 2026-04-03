(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const state = {
    source: 'mock',
    forecast: null,
    chart: null
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

  function renderChart(fx) {
    const el = $('chart');
    if (!el) return;

    if (!state.chart) {
      state.chart = echarts.init(el);
      window.addEventListener('resize', () => state.chart && state.chart.resize());

      state.chart.on('click', (params) => {
        const ds = params?.name;
        if (ds) {
          $('selectedDate').textContent = ds;
        }
      });
    }

    const timeAxis = fx?.time_axis || [];
    const y = fx?.series?.champion_pred || [];

    const option = {
      tooltip: {
        trigger: 'axis'
      },
      xAxis: {
        type: 'category',
        data: timeAxis
      },
      yAxis: {
        type: 'value'
      },
      series: [
        {
          name: 'Champion pred',
          type: 'line',
          data: y,
          showSymbol: true,
          symbolSize: 8,
          smooth: false
        }
      ]
    };

    state.chart.setOption(option, true);
  }

  async function loadData() {
    setActiveSourceButtons();

    if (state.source === 'api') {
      state.forecast = await window.TestShared.fetchForecast({ h: 7, includeAll: 1 });
    } else {
      state.forecast = window.TestShared.getMockForecast(7);
    }

    renderChart(state.forecast);
  }

  function bindUI() {
    $('btnSourceMock')?.addEventListener('click', async () => {
      state.source = 'mock';
      await loadData();
    });

    $('btnSourceApi')?.addEventListener('click', async () => {
      state.source = 'api';
      try {
        await loadData();
      } catch (e) {
        console.error(e);
        alert(String(e.message || e));
        state.source = 'mock';
        await loadData();
      }
    });
  }

  document.addEventListener('DOMContentLoaded', async () => {
    bindUI();
    setActiveSourceButtons();
    await loadData();
  });
})();
