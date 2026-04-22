<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Chart, type ChartConfiguration } from 'chart.js/auto';
  import { fmtNum } from '$lib/fmt';

  export let timeGrid:      number[] = [];
  export let pfeBase:       number[] = [];
  export let pfeStressed:   number[] = [];
  export let height:        number   = 220;
  export let enableJump:    boolean  = false;
  export let isStressed:    boolean  = false;

  let canvas: HTMLCanvasElement;
  let chart:  Chart;

  $: spikeIndex = pfeBase.length > 0 ? pfeBase.indexOf(Math.max(...pfeBase)) : -1;
  $: spikeValue = spikeIndex >= 0 ? pfeBase[spikeIndex] : 0;
  $: spikeTime  = spikeIndex >= 0 ? (timeGrid[spikeIndex] ?? 0) : 0;
  $: pointColors = pfeBase.map((_, i) => i === spikeIndex ? '#ff4d6a' : '#3b82f6');

  const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 } as const,
    plugins: {
      legend:  { labels: { color: '#94a3b8', font: { size: 11 } } },
      tooltip: { backgroundColor: '#1c1f26', titleColor: '#e2e8f0', bodyColor: '#94a3b8', borderColor: '#2d3142', borderWidth: 1 },
    },
    scales: {
      x: {
        grid:   { color: 'rgba(45,49,66,.7)' },
        ticks:  { color: '#64748b', font: { size: 10 } },
        title:  { display: true, text: 'Time (years)', color: '#64748b', font: { size: 10 } },
      },
      y: {
        grid:   { color: 'rgba(45,49,66,.7)' },
        ticks:  { color: '#64748b', font: { size: 10 }, callback: (v: unknown) => fmtNum(Number(v), 0) },
        title:  { display: true, text: 'PFE', color: '#64748b', font: { size: 10 } },
      },
    },
  };

  onMount(() => {
    const cfg: ChartConfiguration = {
      type: 'line',
      data: {
        labels: timeGrid.map((t) => t.toFixed(2)),
        datasets: [
          {
            label: 'PFE (base)',
            data:  pfeBase,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59,130,246,.08)',
            borderWidth: 2,
            fill: true,
            tension: 0.35,
            pointRadius: 3,
            pointBackgroundColor: pointColors,
          },
          {
            label: 'PFE (stressed)',
            data:  pfeStressed,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245,158,11,.06)',
            borderWidth: 2,
            borderDash: [5, 3],
            fill: true,
            tension: 0.35,
            pointRadius: 3,
            pointBackgroundColor: '#f59e0b',
          },
        ],
      },
      options: CHART_DEFAULTS as ChartConfiguration['options'],
    };
    chart = new Chart(canvas, cfg);
  });

  onDestroy(() => chart?.destroy());

  $: if (chart) {
    chart.data.labels                                  = timeGrid.map((t) => t.toFixed(2));
    chart.data.datasets[0].data                        = pfeBase;
    (chart.data.datasets[0] as any).pointBackgroundColor = pointColors;
    chart.data.datasets[1].data                        = pfeStressed;
    chart.data.datasets[1].hidden                      = pfeStressed.length === 0;
    chart.update('none');
  }
</script>

<div class="chart-wrap" style="height:{height}px">
  <canvas bind:this={canvas}></canvas>
</div>

{#if spikeIndex >= 0 && pfeBase.length > 1}
  <div style="margin-top:.5rem;padding:.5rem .75rem;background:var(--surface2);border-left:3px solid var(--red);border-radius:var(--radius-sm);font-size:.76rem">
    <span style="color:var(--red);font-weight:600">Peak PFE:</span>
    <span style="color:var(--text)"> {fmtNum(spikeValue, 0)} at t={spikeTime.toFixed(2)}yr</span>
    <div style="color:var(--muted);margin-top:.2rem">
      {#if isStressed}
        Spike driven by stress scenario — volatility shock amplifies tail exposure at this horizon.
      {:else if enableJump}
        Spike driven by jump-diffusion — a simulated default jump concentrates exposure here.
      {:else}
        Spike driven by high volatility — Monte Carlo paths diverge most at this time step.
      {/if}
    </div>
  </div>
{/if}
