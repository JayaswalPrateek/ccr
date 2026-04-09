<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Chart, type ChartConfiguration } from 'chart.js/auto';

  export let timeGrid:    number[] = [];
  export let epeBase:     number[] = [];
  export let epeStressed: number[] = [];
  export let cva:         number   = 0;
  export let height:      number   = 220;

  let canvas: HTMLCanvasElement;
  let chart:  Chart;

  onMount(() => {
    const cfg: ChartConfiguration = {
      type: 'line',
      data: {
        labels: timeGrid.map((t) => t.toFixed(2)),
        datasets: [
          {
            label: 'EPE (base)',
            data:  epeBase,
            borderColor: '#00d4aa',
            backgroundColor: 'rgba(0,212,170,.08)',
            borderWidth: 2,
            fill: true,
            tension: 0.35,
            pointRadius: 3,
            pointBackgroundColor: '#00d4aa',
          },
          {
            label: 'EPE (stressed)',
            data:  epeStressed,
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
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: {
          legend:  { labels: { color: '#94a3b8', font: { size: 11 } } },
          tooltip: { backgroundColor: '#1c1f26', titleColor: '#e2e8f0', bodyColor: '#94a3b8', borderColor: '#2d3142', borderWidth: 1 },
          ...({ annotation: {} } as Record<string, unknown>),
        },
        scales: {
          x: {
            grid:   { color: 'rgba(45,49,66,.7)' },
            ticks:  { color: '#64748b', font: { size: 10 } },
            title:  { display: true, text: 'Time (years)', color: '#64748b', font: { size: 10 } },
          },
          y: {
            grid:   { color: 'rgba(45,49,66,.7)' },
            ticks:  { color: '#64748b', font: { size: 10 } },
            title:  { display: true, text: 'EPE', color: '#64748b', font: { size: 10 } },
          },
        },
      },
    };
    chart = new Chart(canvas, cfg);
  });

  onDestroy(() => chart?.destroy());

  $: if (chart) {
    chart.data.labels           = timeGrid.map((t) => t.toFixed(2));
    chart.data.datasets[0].data = epeBase;
    chart.data.datasets[1].data = epeStressed;
    chart.data.datasets[1].hidden = epeStressed.length === 0;
    chart.update('none');
  }
</script>

<div class="chart-wrap" style="height:{height}px">
  <canvas bind:this={canvas}></canvas>
</div>
{#if cva > 0}
  <div style="text-align:center;font-size:.72rem;color:var(--muted);margin-top:.25rem;">
    CVA = <strong style="color:var(--green)">{cva.toFixed(4)}</strong>
  </div>
{/if}
