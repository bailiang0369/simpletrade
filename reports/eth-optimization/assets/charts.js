// assets/charts.js — ETH 优化报告图表
(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var good = '#1a7f37';

  function baseAxes(axisColor, textColor) {
    return {
      axisLine: { lineStyle: { color: rule } },
      axisTick: { lineStyle: { color: rule } },
      axisLabel: { color: textColor, fontFamily: 'JetBrainsMono' },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    };
  }

  // ---- 图3: mv 分月准确率 ----
  var mvEl = document.getElementById('chart-mv-month');
  if (mvEl) {
    var mvCat = ['2024-10','2024-11','2024-12','2025-01','2025-02','2025-03','2025-04','2025-05','2025-06','2025-07','2025-08','2025-09'];
    var mvVal = [0.584, 0.532, 0.549, 0.518, 0.586, 0.578, 0.482, 0.459, 0.746, 0.572, 0.529, 0.529];
    var c1 = echarts.init(mvEl, null, { renderer: 'svg' });
    c1.setOption({
      animation: false,
      color: [accent],
      tooltip: { appendToBody: true, trigger: 'axis', formatter: function (ps) { return ps[0].axisValue + ' acc=' + (ps[0].value * 100).toFixed(1) + '%'; } },
      grid: { left: 42, right: 20, top: 26, bottom: 40 },
      xAxis: Object.assign({ type: 'category', data: mvCat }, baseAxes(null, ink)),
      yAxis: Object.assign({ type: 'value', min: 0.40, max: 0.80, axisLabel: { formatter: function (v) { return (v * 100) + '%'; }, color: ink, fontFamily: 'JetBrainsMono' } }, baseAxes(null, ink)),
      series: [{
        type: 'bar', data: mvVal,
        itemStyle: { color: function (p) { return (p.value >= 0.55 ? good : accent); }, borderRadius: [3, 3, 0, 0] },
        label: { show: true, position: 'top', formatter: function (p) { return (p.value * 100).toFixed(1) + '%'; }, color: muted, fontFamily: 'JetBrainsMono', fontSize: 10 }
      }],
      markLine: {}
    });
    window.addEventListener('resize', function () { c1.resize(); });
  }

  // ---- 图4: test 分月准确率 + 65% 目标线 ----
  var teEl = document.getElementById('chart-te-month');
  if (teEl) {
    var teCat = ['2025-09','2025-10','2025-11','2025-12','2026-01','2026-02','2026-03','2026-04','2026-05','2026-06','2026-07','2026-08'];
    var teVal = [0.700, 0.677, 0.557, 0.519, 0.647, 0.542, 0.713, 0.535, 0.677, 0.631, 0.625, 0.667];
    var c2 = echarts.init(teEl, null, { renderer: 'svg' });
    c2.setOption({
      animation: false,
      color: [accent2],
      tooltip: { appendToBody: true, trigger: 'axis', formatter: function (ps) { return ps[0].axisValue + ' acc=' + (ps[0].value * 100).toFixed(1) + '%'; } },
      grid: { left: 42, right: 20, top: 26, bottom: 40 },
      xAxis: Object.assign({ type: 'category', data: teCat }, baseAxes(null, ink)),
      yAxis: Object.assign({ type: 'value', min: 0.40, max: 0.80, axisLabel: { formatter: function (v) { return (v * 100) + '%'; }, color: ink, fontFamily: 'JetBrainsMono' } }, baseAxes(null, ink)),
      series: [{
        type: 'bar', data: teVal,
        itemStyle: { color: function (p) { return (p.value >= 0.65 ? good : accent2); }, borderRadius: [3, 3, 0, 0] },
        label: { show: true, position: 'top', formatter: function (p) { return (p.value * 100).toFixed(1) + '%'; }, color: muted, fontFamily: 'JetBrainsMono', fontSize: 10 },
        markLine: {
          silent: true, symbol: 'none',
          lineStyle: { color: accent, type: 'dashed' },
          data: [{ yAxis: 0.65, label: { show: true, formatter: '65% 目标', position: 'insideEndTop', color: accent, fontFamily: 'JetBrainsMono', fontSize: 11 } }]
        }
      }]
    });
    window.addEventListener('resize', function () { c2.resize(); });
  }
})();