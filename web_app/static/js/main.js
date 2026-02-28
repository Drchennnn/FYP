// 初始化 ECharts
const chartDom = document.getElementById('trafficChart');
const myChart = echarts.init(chartDom);
let chartData = {
    dates: [],
    trueVals: [],
    predVals: []
};

// 国际化配置
let currentLang = 'zh';
const translations = {
    zh: {
        nav_title: '基于 LSTM 的九寨沟景区客流动态预测系统',
        system_status: '系统在线',
        chart_title: '实时客流监测与趋势预测',
        stat_last_date: '最新数据日期',
        stat_last_pred: '最新预测客流',
        btn_1day: '1天',
        btn_3days: '3天',
        btn_7days: '7天',
        label_or_date: '或指定日期：',
        btn_run: '立即运算',
        modal_title: '预测完成',
        modal_desc: '预计游客人次',
        btn_close: '关闭',
        chart_legend_real: '真实客流',
        chart_legend_pred: 'AI 预测客流',
        modal_generated: '已生成趋势',
        modal_total_days: '共 {n} 天预测数据',
        swal_error_title: '数据加载失败',
        swal_warning_title: '请选择日期或点击天数按钮',
        swal_predict_error: '预测失败',
        loading_text: 'AI 模型运算中...'
    },
    en: {
        nav_title: 'LSTM-based Jiuzhaigou Tourist Flow Prediction System',
        system_status: 'SYSTEM ONLINE',
        chart_title: 'Real-time Monitoring & Trend Prediction',
        stat_last_date: 'Latest Data Date',
        stat_last_pred: 'Latest Forecast',
        btn_1day: '1 Day',
        btn_3days: '3 Days',
        btn_7days: '7 Days',
        label_or_date: 'Or Specific Date:',
        btn_run: 'Run Prediction',
        modal_title: 'Prediction Complete',
        modal_desc: 'Estimated Visitors',
        btn_close: 'Close',
        chart_legend_real: 'Real Visitor Flow',
        chart_legend_pred: 'AI Predicted Flow',
        modal_generated: 'Trend Generated',
        modal_total_days: 'Total {n} days of data',
        swal_error_title: 'Data Load Failed',
        swal_warning_title: 'Please select a date or click a button',
        swal_predict_error: 'Prediction Failed',
        loading_text: 'AI MODEL COMPUTING...'
    }
};

// 切换语言函数
function toggleLanguage() {
    currentLang = currentLang === 'zh' ? 'en' : 'zh';
    const t = translations[currentLang];
    
    // 1. 更新按钮文本
    const btn = document.getElementById('langSwitch');
    btn.innerHTML = `<i class="fas fa-globe me-1"></i> ${currentLang === 'zh' ? 'EN' : '中文'}`;
    
    // 2. 更新所有带有 data-i18n 属性的元素
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (t[key]) {
            el.innerText = t[key];
        }
    });

    // 3. 更新 Loading 文本
    const loadingText = document.querySelector('#loadingOverlay .text-white');
    if (loadingText) loadingText.innerText = t.loading_text;

    // 4. 更新图表
    updateChart();
}

// 窗口大小改变时重绘
window.addEventListener('resize', () => myChart.resize());

// 初始化页面
document.addEventListener('DOMContentLoaded', async () => {
    // 设置日期选择器默认值为明天
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    document.getElementById('predictDate').valueAsDate = tomorrow;
    
    // 初始应用中文
    // toggleLanguage(); // 如果想默认英文可以调用一次，或者保持默认中文
    
    await fetchInitData();
});

// 获取初始数据
async function fetchInitData() {
    try {
        const response = await fetch('/api/data');
        const data = await response.json();
        
        if (data.error) throw new Error(data.error);

        chartData = {
            dates: data.dates,
            trueVals: data.true_vals,
            predVals: data.pred_vals,
            holidayRanges: data.holiday_ranges || [] 
        };
        
        updateChart();
        updateStats();

    } catch (error) {
        console.error('Error:', error);
        const t = translations[currentLang];
        Swal.fire({
            icon: 'error',
            title: t.swal_error_title,
            text: error.message,
            background: '#111927',
            color: '#fff'
        });
    }
}

// 更新图表
function updateChart() {
    const t = translations[currentLang];
    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(17, 25, 39, 0.9)',
            borderColor: '#1f2d40',
            textStyle: { color: '#fff' },
            axisPointer: {
                type: 'line',
                lineStyle: { color: '#4ade80', type: 'dashed' }
            }
        },
        legend: {
            data: [t.chart_legend_real, t.chart_legend_pred], // 使用翻译后的图例
            textStyle: { color: '#94a3b8' },
            top: 10
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '10%',
            top: '15%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: chartData.dates,
            axisLine: { lineStyle: { color: '#1f2d40' } },
            axisLabel: { color: '#94a3b8' }
        },
        yAxis: {
            type: 'value',
            splitLine: { lineStyle: { color: '#1f2d40', type: 'dashed' } },
            axisLabel: { color: '#94a3b8' }
        },
        dataZoom: [
            { type: 'inside', start: 80, end: 100 },
            { type: 'slider', bottom: 0, height: 20, borderColor: '#1f2d40', textStyle: { color: '#94a3b8' } }
        ],
        series: [
            {
                name: t.chart_legend_real, // 翻译后的名称
                type: 'line',
                smooth: true,
                showSymbol: false,
                lineStyle: { width: 2, color: '#00f2ff' },
                areaStyle: {
                    opacity: 0.2,
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#00f2ff' },
                        { offset: 1, color: 'rgba(0, 242, 255, 0.01)' }
                    ])
                },
                markArea: {
                    itemStyle: { color: 'rgba(255, 69, 0, 0.15)' },
                    label: {
                        show: true,
                        color: '#ff9d00',
                        position: 'top',
                        fontSize: 12
                    },
                    data: chartData.holidayRanges ? chartData.holidayRanges.map(range => {
                        let color = 'rgba(255, 69, 0, 0.15)';
                        let name = range.name || '';
                        let displayName = name;

                        // 节假日中英映射
                        if (currentLang === 'en') {
                            if (name.includes('春节')) displayName = 'Spring Festival';
                            else if (name.includes('国庆')) displayName = 'National Day';
                            else if (name.includes('中秋')) displayName = 'Mid-Autumn';
                            else if (name.includes('五一') || name.includes('劳动')) displayName = 'Labor Day';
                            else if (name.includes('元旦')) displayName = 'New Year';
                            else if (name.includes('清明')) displayName = 'Tomb Sweeping';
                            else if (name.includes('端午')) displayName = 'Dragon Boat';
                            else if (name.includes('暑假')) displayName = 'Summer Holiday';
                            else if (name.includes('寒假')) displayName = 'Winter Holiday';
                        }
                        
                        if (name.includes('春节') || name.includes('New Year')) {
                            color = 'rgba(220, 20, 60, 0.25)';
                        } else if (name.includes('国庆') || name.includes('National')) {
                            color = 'rgba(255, 0, 0, 0.2)';
                        } else if (name.includes('中秋') || name.includes('Mid-Autumn')) {
                            color = 'rgba(255, 215, 0, 0.2)';
                        } else if (name.includes('五一') || name.includes('Labor')) {
                            color = 'rgba(0, 191, 255, 0.2)';
                        } else if (name.includes('暑假') || name.includes('Summer')) {
                            color = 'rgba(0, 255, 127, 0.05)'; // Spring Green, very transparent
                        } else if (name.includes('寒假') || name.includes('Winter')) {
                            color = 'rgba(0, 255, 255, 0.05)'; // Cyan, very transparent
                        }
                        
                        return [
                            { name: displayName, xAxis: range.start, itemStyle: { color: color } },
                            { xAxis: range.end }
                        ];
                    }) : []
                },
                data: chartData.trueVals
            },
            {
                name: t.chart_legend_pred, // 翻译后的名称
                type: 'line',
                smooth: true,
                showSymbol: false,
                lineStyle: { width: 3, color: '#ff9d00', type: 'dashed' },
                itemStyle: { color: '#ff9d00' },
                data: chartData.predVals
            }
        ]
    };
    myChart.setOption(option); // ECharts 会自动合并更新配置
}

// 更新统计卡片
function updateStats() {
    if (chartData.dates.length > 0) {
        let lastRealIdx = -1;
        for (let i = chartData.trueVals.length - 1; i >= 0; i--) {
            if (chartData.trueVals[i] !== null && chartData.trueVals[i] !== undefined && chartData.trueVals[i] !== 0) {
                lastRealIdx = i;
                break;
            }
        }

        if (lastRealIdx !== -1) {
            document.getElementById('lastDate').innerText = chartData.dates[lastRealIdx];
            const val = chartData.trueVals[lastRealIdx];
            document.getElementById('lastValue').innerText = val ? val.toLocaleString() : '--';
        } else {
            const lastIdx = chartData.dates.length - 1;
            document.getElementById('lastDate').innerText = chartData.dates[lastIdx];
            const val = chartData.predVals[lastIdx] || chartData.trueVals[lastIdx];
            document.getElementById('lastValue').innerText = val ? val.toLocaleString() : '--';
        }
    }
}

function setPredictDays(days) {
    document.getElementById('predictDate').value = '';
    runPrediction(days);
}

// 执行预测
async function runPrediction(days = null) {
    const t = translations[currentLang];
    let payload = {};
    
    if (days) {
        payload = { days: days };
    } else {
        const dateInput = document.getElementById('predictDate').value;
        if (!dateInput) {
            Swal.fire({
                icon: 'warning',
                title: t.swal_warning_title,
                background: '#111927',
                color: '#fff'
            });
            return;
        }
        payload = { future_date: dateInput };
    }

    document.getElementById('loadingOverlay').style.display = 'flex';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        document.getElementById('loadingOverlay').style.display = 'none';

        if (result.error) throw new Error(result.error);

        if (result.predictions && result.predictions.length > 1) {
            document.getElementById('modalDate').innerText = `${result.start_date} ${currentLang === 'zh' ? '至' : 'to'} ${result.end_date}`;
            document.getElementById('modalValue').innerText = t.modal_generated;
            document.querySelector('#resultModal .text-white-50.mt-2').innerText = t.modal_total_days.replace('{n}', result.predictions.length);
        } else if (result.predictions && result.predictions.length === 1) {
            document.getElementById('modalDate').innerText = result.predictions[0].date;
            document.getElementById('modalValue').innerText = result.predictions[0].value.toLocaleString();
            document.querySelector('#resultModal .text-white-50.mt-2').innerText = t.modal_desc;
        }
        
        new bootstrap.Modal(document.getElementById('resultModal')).show();
        await fetchInitData();

    } catch (error) {
        document.getElementById('loadingOverlay').style.display = 'none';
        console.error('Prediction Error:', error);
        Swal.fire({
            icon: 'error',
            title: t.swal_predict_error,
            text: error.message,
            background: '#111927',
            color: '#fff'
        });
    }
}
