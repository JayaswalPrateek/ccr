<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Chart, type ChartConfiguration } from 'chart.js/auto';
  import type { BacktestObservation } from '$lib/types';
  import { fmtNum } from '$lib/fmt';

  export let pfeProfile: number[]              = [];
  export let realised:   BacktestObservation[] = [];
  export let coveragePct: number               = 100;
  export let height:     number                = 280;

  let canvas: HTMLCanvasElement;
  let chart:  Chart;

  $: peakPfe = pfeProfile.length ? Math.max(...pfeProfile) : 0;

  $: if (chart && realised.length) {
    chart.data.labels = realised.map((r) => r.date);
    chart.data.datasets[0].data = realised.map((r) => r.exposure);
    chart.data.datasets[0].pointBackgroundColor = realised.map((r) =>
      r.breach ? '#ef4444' : '#22c55e'
    );
    chart.data.datasets[1].data = realised.map(() => peakPfe);
    chart.update('none');
  }

  onMount(() => {
    const cfg: ChartConfiguration = {
      type: 'line',
      data: {
        labels: realised.map((r) => r.date),
        datasets: [
          {
            label: 'Realised Exposure',
            data:  realised.map((r) => r.exposure),
            borderColor: '#22c55e',
            borderWidth: 1.5,
            fill: false,
            tension: 0.2,
            pointRadius: 4,
            pointBackgroundColor: realised.map((r) => r.breach ? '#ef4444' : '#22c55e'),
            pointBorderColor:     realised.map((r) => r.breach ? '#ef4444' : '#22c55e'),
          },
          {
            label: `PFE Peak (${fmtNum(peakPfe, 0)})`,
            data:  realised.map(() => peakPfe),
            borderColor: '#3b82f6',
            borderDash: [6, 4],
            borderWidth: 1.5,
            fill: false,
            tension: 0,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const v = ctx.parsed.y;
                if (ctx.datasetIndex === 0) {
                  const obs = realised[ctx.dataIndex];
                  return ` ${fmtNum(v, 0)}${obs?.breach ? ' ⚠ BREACH' : ''}`;
                }
                return ` PFE: ${fmtNum(v, 0)}`;
              },
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color: '#64748b',
              maxTicksLimit: 12,
              font: { size: 9 },
              maxRotation: 35,
            },
            grid: { color: 'rgba(255,255,255,.06)' },
          },
          y: {
            ticks: {
              color: '#64748b',
              font: { size: 10 },
              callback: (v) => fmtNum(Number(v), 0),
            },
            title: { display: true, text: 'Exposure', color: '#64748b', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,.06)' },
          },
        },
      },
    };
    chart = new Chart(canvas, cfg);
  });

  onDestroy(() => { chart?.destroy(); });
</script>

<div style="position:relative">
  <canvas bind:this={canvas} style="height:{height}px"></canvas>
</div>
<div style="display:flex;gap:1rem;margin-top:.5rem;font-size:.75rem;color:var(--muted)">
  <span style="color:#22c55e">● within PFE</span>
  <span style="color:#ef4444">● breach</span>
  <span style="color:#3b82f6">– – PFE peak band</span>
  <span style="margin-left:auto;font-weight:600;color:{coveragePct >= 95 ? 'var(--green)' : coveragePct >= 85 ? 'var(--amber)' : 'var(--red)'}">
    {coveragePct.toFixed(1)}% coverage
  </span>
</div>
