<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Chart } from 'chart.js/auto';
  import { livePrices } from '$lib/state';

  export let symbol: string = 'SPY';
  export let height: number = 200;

  const MAX_TICKS = 60;
  let canvas: HTMLCanvasElement;
  let chart:  Chart;
  let ticks:  number[] = [];
  let labels: string[] = [];

  onMount(() => {
    chart = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: symbol,
          data:  ticks,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,.06)',
          borderWidth: 1.5,
          fill: true,
          tension: 0.2,
          pointRadius: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: false },
          tooltip: { backgroundColor: '#1c1f26', titleColor: '#e2e8f0', bodyColor: '#94a3b8', borderColor: '#2d3142', borderWidth: 1 },
        },
        scales: {
          x: { display: false },
          y: {
            grid:   { color: 'rgba(45,49,66,.6)' },
            ticks:  { color: '#64748b', font: { size: 9 }, maxTicksLimit: 5 },
          },
        },
      },
    });
  });

  onDestroy(() => chart?.destroy());

  // Push new tick when livePrices updates for this symbol.
  $: {
    const price = $livePrices[symbol];
    if (price !== undefined && chart) {
      ticks.push(price);
      labels.push(new Date().toLocaleTimeString());
      if (ticks.length > MAX_TICKS) {
        ticks.shift();
        labels.shift();
      }
      chart.data.labels              = labels;
      chart.data.datasets[0].data    = ticks;
      chart.data.datasets[0].label   = symbol;
      chart.update('none');
    }
  }
</script>

<div style="margin-bottom:.5rem;display:flex;align-items:center;justify-content:space-between;">
  <span style="font-size:.78rem;font-weight:600;color:var(--text-2)">{symbol}</span>
  <span style="font-size:.85rem;font-weight:700;color:var(--green)">
    {$livePrices[symbol]?.toFixed(2) ?? '—'}
  </span>
</div>
<div class="chart-wrap" style="height:{height}px">
  <canvas bind:this={canvas}></canvas>
</div>
