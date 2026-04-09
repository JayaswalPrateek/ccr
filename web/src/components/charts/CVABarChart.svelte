<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Chart } from 'chart.js/auto';
  import type { SimulationHistoryItem } from '$lib/types';

  export let history: SimulationHistoryItem[] = [];
  export let height:  number                  = 200;

  let canvas: HTMLCanvasElement;
  let chart:  Chart;

  function buildData(items: SimulationHistoryItem[]) {
    // Show last 10 base (non-stressed) runs.
    const base    = items.filter((i) => !i.is_stressed).slice(0, 10).reverse();
    const labels  = base.map((i) => new Date(i.time).toLocaleDateString());
    const cvas    = base.map((i) => i.cva);
    const maxCva  = Math.max(...cvas, 1);
    const colors  = cvas.map((c) => (c > maxCva * 0.8 ? '#ff4d6a' : '#3b82f6'));
    return { labels, cvas, colors };
  }

  onMount(() => {
    const { labels, cvas, colors } = buildData(history);
    chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'CVA',
          data:  cvas,
          backgroundColor: colors,
          borderRadius: 3,
          borderWidth: 0,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        plugins: {
          legend: { display: false },
          tooltip: { backgroundColor: '#1c1f26', titleColor: '#e2e8f0', bodyColor: '#94a3b8', borderColor: '#2d3142', borderWidth: 1 },
        },
        scales: {
          x: { grid: { color: 'rgba(45,49,66,.7)' }, ticks: { color: '#64748b', font: { size: 9 } } },
          y: { grid: { display: false },              ticks: { color: '#64748b', font: { size: 9 } } },
        },
      },
    });
  });

  onDestroy(() => chart?.destroy());

  $: if (chart) {
    const { labels, cvas, colors } = buildData(history);
    chart.data.labels                             = labels;
    chart.data.datasets[0].data                   = cvas;
    (chart.data.datasets[0] as unknown as { backgroundColor: string[] }).backgroundColor = colors;
    chart.update('none');
  }
</script>

<div class="chart-wrap" style="height:{height}px">
  <canvas bind:this={canvas}></canvas>
</div>
