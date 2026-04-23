<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Chart, type ChartConfiguration } from 'chart.js/auto';

  export let hz_1y:      number | null = null;
  export let hz_3y:      number | null = null;
  export let hz_5y:      number | null = null;
  export let hz_10y:     number | null = null;
  export let flatRate:   number        = 0.02;
  export let height:     number        = 220;

  let canvas: HTMLCanvasElement;
  let chart:  Chart;

  function lerp(tenors: [number, number][], t: number): number {
    if (tenors.length === 0) return flatRate;
    if (t <= tenors[0][0]) return tenors[0][1];
    if (t >= tenors[tenors.length - 1][0]) return tenors[tenors.length - 1][1];
    for (let i = 0; i < tenors.length - 1; i++) {
      const [t0, h0] = tenors[i];
      const [t1, h1] = tenors[i + 1];
      if (t >= t0 && t <= t1) {
        return h0 + (h1 - h0) * (t - t0) / (t1 - t0);
      }
    }
    return flatRate;
  }

  function buildCurve(tenors: [number, number][], useFlat: boolean): number[] {
    return labels.map((t) => {
      const hz = useFlat ? flatRate : lerp(tenors, t);
      return Math.exp(-hz * t);
    });
  }

  const N = 60;
  const labels = Array.from({ length: N + 1 }, (_, i) => (i * 10) / N);

  $: tenors = ([
    [1,  hz_1y],
    [3,  hz_3y],
    [5,  hz_5y],
    [10, hz_10y],
  ] as [number, number | null][]).filter(([, h]) => h !== null) as [number, number][];

  $: termData = buildCurve(tenors, false);
  $: flatData = buildCurve([], true);

  onMount(() => {
    const cfg: ChartConfiguration = {
      type: 'line',
      data: {
        labels: labels.map((t) => t.toFixed(2)),
        datasets: [
          {
            label: 'Term Structure',
            data:  termData,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59,130,246,.08)',
            borderWidth: 2,
            fill: true,
            tension: 0.35,
            pointRadius: 0,
          },
          {
            label: `Flat λ=${flatRate.toFixed(4)}`,
            data:  flatData,
            borderColor: '#f59e0b',
            borderDash: [5, 4],
            borderWidth: 1.5,
            fill: false,
            tension: 0.35,
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
              label: (ctx) => ` ${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(2)}%`,
            },
          },
        },
        scales: {
          x: {
            ticks: { color: '#64748b', maxTicksLimit: 11, font: { size: 10 } },
            title: { display: true, text: 'Years', color: '#64748b', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,.06)' },
          },
          y: {
            min: 0,
            max: 1,
            ticks: {
              color: '#64748b',
              font: { size: 10 },
              callback: (v) => `${(Number(v) * 100).toFixed(0)}%`,
            },
            title: { display: true, text: 'Survival Probability', color: '#64748b', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,.06)' },
          },
        },
      },
    };
    chart = new Chart(canvas, cfg);
  });

  $: if (chart) {
    chart.data.datasets[0].data = termData;
    chart.data.datasets[1].data = flatData;
    chart.data.datasets[1].label = `Flat λ=${flatRate.toFixed(4)}`;
    chart.update('none');
  }

  onDestroy(() => { chart?.destroy(); });
</script>

<canvas bind:this={canvas} style="height:{height}px"></canvas>
