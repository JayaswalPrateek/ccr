<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Chart } from 'chart.js/auto';
  import type { AttributionItem } from '$lib/types';

  export let items:  AttributionItem[] = [];
  export let height: number            = 200;

  let canvas: HTMLCanvasElement;
  let chart:  Chart;

  function buildData(data: AttributionItem[]) {
    const labels = data.map((d) => `${d.deriv_type} (${(d.notional / 1_000_000).toFixed(1)}M)`);
    const values = data.map((d) => d.allocated_cva);
    const maxVal = Math.max(...values, 1);
    const colors = values.map((v) => (v >= maxVal * 0.8 ? '#f59e0b' : '#3b82f6'));
    return { labels, values, colors };
  }

  onMount(() => {
    const { labels, values, colors } = buildData(items);
    chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Allocated CVA',
          data:  values,
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
          tooltip: {
            backgroundColor: '#1c1f26', titleColor: '#e2e8f0',
            bodyColor: '#94a3b8', borderColor: '#2d3142', borderWidth: 1,
            callbacks: {
              label: (ctx) => ` CVA: ${Number(ctx.raw).toFixed(6)}`,
            },
          },
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
    const { labels, values, colors } = buildData(items);
    chart.data.labels                             = labels;
    chart.data.datasets[0].data                   = values;
    (chart.data.datasets[0] as unknown as { backgroundColor: string[] }).backgroundColor = colors;
    chart.update('none');
  }
</script>

<div class="chart-wrap" style="height:{height}px">
  <canvas bind:this={canvas}></canvas>
</div>
