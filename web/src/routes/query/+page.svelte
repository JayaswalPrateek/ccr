<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Chart } from 'chart.js/auto';
  import { api } from '$lib/api';
  import type {
    Counterparty,
    ExposureRankRow,
    MarginActivityRow,
    PfePeakRow,
    RiskTimelineRow,
    VolCvaRow,
  } from '$lib/types';

  // ── Template definitions ───────────────────────────────────────────────────
  type TemplateId = 'risk-timeline' | 'exposure-ranking' | 'pfe-peaks' | 'margin-activity' | 'vol-cva';

  interface Template {
    id:          TemplateId;
    label:       string;
    description: string;
    icon:        string;
    color:       string;
    chartType:   'line' | 'bar' | 'scatter';
  }

  const TEMPLATES: Template[] = [
    {
      id: 'risk-timeline', label: 'Risk Timeline',
      description: 'CVA and margin required over time for any counterparty. Ideal for spotting trend breaks and regime shifts.',
      icon: 'M2 12 L4 8 L7 10 L10 4 L13 7 L16 2',
      color: '#3b82f6',
      chartType: 'line',
    },
    {
      id: 'exposure-ranking', label: 'Exposure Ranking',
      description: 'Cross-counterparty CVA league table from the latest simulation run per party. Identifies concentration risk at a glance.',
      icon: 'M2 14 L2 10 L6 10 L6 14 M7 14 L7 6 L11 6 L11 14 M12 14 L12 2 L16 2 L16 14',
      color: '#f59e0b',
      chartType: 'bar',
    },
    {
      id: 'pfe-peaks', label: 'PFE Peaks',
      description: 'Maximum Potential Future Exposure extracted from each run\'s full profile. Reveals tail-risk events across history.',
      icon: 'M1 15 L5 7 L8 11 L11 3 L14 8 L16 5',
      color: '#ef4444',
      chartType: 'bar',
    },
    {
      id: 'margin-activity', label: 'Margin Activity',
      description: 'Full margin call funnel — amounts, statuses, and counterparties over a date range. Essential for settlement tracking.',
      icon: 'M2 2 L14 2 L14 14 L2 14 Z M5 6 L11 6 M5 9 L9 9',
      color: '#8b5cf6',
      chartType: 'bar',
    },
    {
      id: 'vol-cva', label: 'Vol vs CVA',
      description: 'Scatter of input volatility (σ) against computed CVA. Shows the sensitivity of credit valuation to market vol assumptions.',
      icon: 'M2 14 L5 8 L7 11 L9 4 L11 10 L14 6 L16 9',
      color: '#10b981',
      chartType: 'scatter',
    },
  ];

  // ── State ──────────────────────────────────────────────────────────────────
  let selectedTemplate: TemplateId | null = null;
  let counterparties: Counterparty[] = [];
  let loadingCPs = true;

  // Shared filters
  let filterCP     = '';
  let filterFrom   = '';
  let filterTo     = '';
  let filterLimit  = 100;

  // Template-specific filters
  let filterMinCva     = 0;
  let filterStatus     = '';
  let filterStressed   = false;

  // Results
  let running = false;
  let rows: unknown[] = [];
  let meta: { row_count: number; executed_at: string } | null = null;
  let summary: Record<string, unknown> | null = null;
  let queryError = '';

  // Canvas for chart
  let chartCanvas: HTMLCanvasElement | null = null;
  let chart: Chart | null = null;

  // Pagination
  let pageSize = 25;
  let currentPage = 0;
  $: totalPages = Math.ceil(rows.length / pageSize);
  $: pagedRows = rows.slice(currentPage * pageSize, (currentPage + 1) * pageSize);

  // Saved queries
  let savedQueries: { id: string; name: string; template: string; filters: Record<string,unknown> }[] = [];
  let showSaveQueryModal = false;
  let saveQueryName = '';

  function loadSavedQueries() {
    try { savedQueries = JSON.parse(localStorage.getItem('ccr_saved_queries') ?? '[]'); } catch { savedQueries = []; }
  }
  function saveQuery() {
    if (!saveQueryName.trim() || !selectedTemplate) return;
    const entry = {
      id: Date.now().toString(), name: saveQueryName.trim(), template: selectedTemplate,
      filters: { counterparty_id: filterCP, from: filterFrom, to: filterTo, limit: filterLimit,
                 min_cva: filterMinCva, status: filterStatus, stressed: filterStressed },
    };
    savedQueries = [...savedQueries, entry];
    try { localStorage.setItem('ccr_saved_queries', JSON.stringify(savedQueries)); } catch {}
    showSaveQueryModal = false; saveQueryName = '';
  }
  function loadSavedQuery(q: typeof savedQueries[0]) {
    selectedTemplate = q.template as any;
    filterCP         = (q.filters.counterparty_id as string) ?? '';
    filterFrom       = (q.filters.from as string) ?? '';
    filterTo         = (q.filters.to as string) ?? '';
    filterLimit      = (q.filters.limit as number) ?? 100;
    filterMinCva     = (q.filters.min_cva as number) ?? 0;
    filterStatus     = (q.filters.status as string) ?? '';
    filterStressed   = (q.filters.stressed as boolean) ?? false;
    rows = []; meta = null;
  }
  function deleteSavedQuery(id: string) {
    savedQueries = savedQueries.filter(q => q.id !== id);
    try { localStorage.setItem('ccr_saved_queries', JSON.stringify(savedQueries)); } catch {}
  }

  onDestroy(() => { if (chart) { chart.destroy(); chart = null; } });

  onMount(async () => {
    loadSavedQueries();
    try {
      counterparties = await api.listCounterparties();
    } catch (_) {
      // non-fatal
    } finally {
      loadingCPs = false;
    }
  });

  function selectTemplate(id: TemplateId) {
    selectedTemplate = id;
    rows = [];
    meta = null;
    summary = null;
    queryError = '';
    if (chart) { chart.destroy(); chart = null; }
  }

  $: template = TEMPLATES.find((t) => t.id === selectedTemplate) ?? null;

  // ── Run query ──────────────────────────────────────────────────────────────
  async function runQuery() {
    if (!selectedTemplate) return;
    running    = true;
    queryError = '';
    rows       = [];
    meta       = null;
    summary    = null;
    if (chart) { chart.destroy(); chart = null; }

    try {
      const common = {
        counterparty_id: filterCP || undefined,
        from:   filterFrom || undefined,
        to:     filterTo   || undefined,
        limit:  filterLimit,
      };

      let res: any;
      if (selectedTemplate === 'risk-timeline') {
        res = await api.queryRiskTimeline({ ...common, stressed_only: filterStressed });
      } else if (selectedTemplate === 'exposure-ranking') {
        res = await api.queryExposureRanking({ from: common.from, to: common.to, min_cva: filterMinCva || undefined, limit: common.limit });
      } else if (selectedTemplate === 'pfe-peaks') {
        res = await api.queryPfePeaks(common);
      } else if (selectedTemplate === 'margin-activity') {
        res = await api.queryMarginActivity({ ...common, status: filterStatus || undefined });
      } else if (selectedTemplate === 'vol-cva') {
        res = await api.queryVolCva({ from: common.from, to: common.to, limit: common.limit });
      }

      if (res) {
        rows    = res.rows ?? [];
        meta    = res.meta ?? null;
        summary = res.summary ?? null;
        currentPage = 0;
      }

      // Render chart after DOM updates
      setTimeout(() => renderChart(), 50);
    } catch (e) {
      queryError = e instanceof Error ? e.message : 'Query failed';
    } finally {
      running = false;
    }
  }

  // ── Chart rendering ────────────────────────────────────────────────────────
  function renderChart() {
    if (!chartCanvas || rows.length === 0 || !template) return;
    if (chart) { chart.destroy(); chart = null; }
    const ctx = chartCanvas.getContext('2d');
    if (!ctx) return;

    let config: any = null;

    if (selectedTemplate === 'risk-timeline') {
      const typed = rows as RiskTimelineRow[];
      config = {
        type: 'line',
        data: {
          labels: typed.map((r) => new Date(r.time).toLocaleDateString()),
          datasets: [
            { label: 'CVA', data: typed.map((r) => r.cva), borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,.1)', tension: 0.3, fill: true, pointRadius: 3 },
            { label: 'WWR-CVA', data: typed.map((r) => r.wwr_cva), borderColor: '#f59e0b', backgroundColor: 'transparent', tension: 0.3, borderDash: [4,2], pointRadius: 3 },
          ],
        },
        options: chartOpts('CVA over Time'),
      };
    } else if (selectedTemplate === 'exposure-ranking') {
      const typed = rows as ExposureRankRow[];
      config = {
        type: 'bar',
        data: {
          labels: typed.map((r) => r.counterparty_name ?? r.counterparty_id.slice(0, 8)),
          datasets: [{ label: 'CVA', data: typed.map((r) => r.cva), backgroundColor: typed.map((r) => r.cva > 0.05 ? '#ef4444' : '#3b82f6') }],
        },
        options: chartOpts('CVA by Counterparty'),
      };
    } else if (selectedTemplate === 'pfe-peaks') {
      const typed = (rows as PfePeakRow[]).slice(0, 30);
      config = {
        type: 'bar',
        data: {
          labels: typed.map((r) => new Date(r.time).toLocaleDateString()),
          datasets: [{ label: 'Peak PFE', data: typed.map((r) => r.peak_pfe), backgroundColor: '#ef4444' }],
        },
        options: chartOpts('Peak PFE per Run'),
      };
    } else if (selectedTemplate === 'margin-activity') {
      const typed = rows as MarginActivityRow[];
      const statuses = ['PENDING', 'ACKNOWLEDGED', 'SETTLED', 'DISPUTED'];
      const colors   = { PENDING: '#f59e0b', ACKNOWLEDGED: '#3b82f6', SETTLED: '#10b981', DISPUTED: '#ef4444' };
      const counts   = statuses.map((s) => typed.filter((r) => r.status === s).length);
      config = {
        type: 'bar',
        data: {
          labels: statuses,
          datasets: [{ label: 'Count', data: counts, backgroundColor: statuses.map((s) => (colors as any)[s]) }],
        },
        options: chartOpts('Margin Call Status Breakdown'),
      };
    } else if (selectedTemplate === 'vol-cva') {
      const typed = (rows as VolCvaRow[]).filter((r) => r.sigma !== null);
      config = {
        type: 'scatter',
        data: {
          datasets: [{
            label: 'σ vs CVA',
            data: typed.map((r) => ({ x: r.sigma, y: r.cva })),
            backgroundColor: 'rgba(16,185,129,.6)',
            pointRadius: 5,
          }],
        },
        options: {
          ...chartOpts('Volatility vs CVA'),
          scales: {
            x: { title: { display: true, text: 'Volatility (σ)', color: '#888' }, grid: { color: '#2a2a3a' }, ticks: { color: '#888' } },
            y: { title: { display: true, text: 'CVA',            color: '#888' }, grid: { color: '#2a2a3a' }, ticks: { color: '#888' } },
          },
        },
      };
    }

    if (config) {
      chart = new Chart(ctx, config) as Chart;
    }
  }

  function chartOpts(title: string) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#c8c8d0', font: { size: 11 } } },
        title:  { display: true, text: title, color: '#c8c8d0', font: { size: 12 } },
      },
      scales: {
        x: { grid: { color: '#2a2a3a' }, ticks: { color: '#888', font: { size: 10 }, maxRotation: 45 } },
        y: { grid: { color: '#2a2a3a' }, ticks: { color: '#888', font: { size: 10 } } },
      },
    };
  }

  // ── Table column helpers ───────────────────────────────────────────────────
  $: columns = getColumns(selectedTemplate);

  function getColumns(t: TemplateId | null): { key: string; label: string; fmt?: (v: any) => string }[] {
    if (t === 'risk-timeline')    return [
      { key: 'time',              label: 'Time',        fmt: (v) => new Date(v).toLocaleString() },
      { key: 'counterparty_name', label: 'Counterparty' },
      { key: 'cva',               label: 'CVA',         fmt: (v) => (+v).toFixed(6) },
      { key: 'wwr_cva',           label: 'WWR-CVA',     fmt: (v) => (+v).toFixed(6) },
      { key: 'margin_required',   label: 'Margin',      fmt: (v) => (+v).toLocaleString(undefined,{maximumFractionDigits:0}) },
      { key: 'is_stressed',       label: 'Stressed',    fmt: (v) => v ? 'Yes' : 'No' },
    ];
    if (t === 'exposure-ranking') return [
      { key: 'counterparty_name', label: 'Counterparty' },
      { key: 'cva',               label: 'CVA',         fmt: (v) => (+v).toFixed(6) },
      { key: 'wwr_cva',           label: 'WWR-CVA',     fmt: (v) => (+v).toFixed(6) },
      { key: 'margin_required',   label: 'Margin',      fmt: (v) => (+v).toLocaleString(undefined,{maximumFractionDigits:0}) },
      { key: 'run_count',         label: 'Runs' },
      { key: 'last_run_time',     label: 'Last Run',    fmt: (v) => new Date(v).toLocaleDateString() },
    ];
    if (t === 'pfe-peaks')        return [
      { key: 'time',              label: 'Time',        fmt: (v) => new Date(v).toLocaleString() },
      { key: 'counterparty_name', label: 'Counterparty' },
      { key: 'peak_pfe',          label: 'Peak PFE',    fmt: (v) => (+v).toFixed(5) },
      { key: 'cva',               label: 'CVA',         fmt: (v) => (+v).toFixed(5) },
    ];
    if (t === 'margin-activity')  return [
      { key: 'issued_at',         label: 'Date',        fmt: (v) => new Date(v).toLocaleString() },
      { key: 'counterparty_name', label: 'Counterparty' },
      { key: 'amount',            label: 'Amount',      fmt: (v) => (+v).toLocaleString(undefined,{maximumFractionDigits:0}) },
      { key: 'excess_exposure',   label: 'Excess Exp',  fmt: (v) => (+v).toLocaleString(undefined,{maximumFractionDigits:0}) },
      { key: 'status',            label: 'Status' },
    ];
    if (t === 'vol-cva')          return [
      { key: 'time',              label: 'Time',        fmt: (v) => new Date(v).toLocaleString() },
      { key: 'sigma',             label: 'σ (Vol)',     fmt: (v) => v != null ? (+v).toFixed(4) : '—' },
      { key: 'num_paths',         label: 'Paths' },
      { key: 'cva',               label: 'CVA',         fmt: (v) => (+v).toFixed(6) },
      { key: 'wwr_cva',           label: 'WWR-CVA',     fmt: (v) => (+v).toFixed(6) },
    ];
    return [];
  }

  function statusBadge(status: string) {
    const map: Record<string, string> = {
      PENDING: 'badge-amber', ACKNOWLEDGED: 'badge-blue', SETTLED: 'badge-green', DISPUTED: 'badge-red',
    };
    return map[status] ?? 'badge-muted';
  }

  function downloadCSV() {
    if (rows.length === 0 || columns.length === 0) return;
    const header = columns.map(c => c.label).join(',');
    const body   = rows.map(row =>
      columns.map(c => {
        const v = (row as any)[c.key];
        const s = c.fmt ? c.fmt(v) : String(v ?? '');
        return `"${s.replace(/"/g, '""')}"`;
      }).join(',')
    ).join('\n');
    const blob = new Blob([header + '\n' + body], { type: 'text/csv' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = `ccr-${selectedTemplate ?? 'query'}.csv`; a.click();
  }
</script>

<svelte:head><title>Query Builder — CCR Engine</title></svelte:head>

<div class="page-header">
  <div>
    <div class="page-title">Query Builder</div>
    <div class="page-sub">Interactive risk analytics — no SQL required</div>
  </div>
  {#if meta}
    <span class="badge badge-muted">{meta.row_count} rows · {new Date(meta.executed_at).toLocaleTimeString()}</span>
  {/if}
</div>

<div style="display:grid;grid-template-columns:260px 1fr;gap:1rem;align-items:start">

  <!-- ── Left panel: template picker + filters ───────────────────────────── -->
  <div style="display:flex;flex-direction:column;gap:.75rem">

    <!-- Template cards -->
    <div class="card" style="padding:.75rem">
      <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:.6rem;font-weight:600">Query Templates</div>
      <div style="display:flex;flex-direction:column;gap:.3rem">
        {#each TEMPLATES as t}
          <button
            class="template-btn"
            class:active={selectedTemplate === t.id}
            style="--accent:{t.color}"
            on:click={() => selectTemplate(t.id)}
          >
            <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"
                 style="width:14px;height:14px;flex-shrink:0;color:{selectedTemplate===t.id?t.color:'var(--muted)'}">
              <path d={t.icon}/>
            </svg>
            <span>{t.label}</span>
          </button>
        {/each}
      </div>
    </div>

    <!-- Saved Queries / Bookmarks -->
    {#if savedQueries.length > 0}
      <div class="card" style="padding:.75rem">
        <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:.4rem;font-weight:600">Bookmarks</div>
        {#each savedQueries as q}
          <div style="display:flex;align-items:center;gap:.3rem;margin-bottom:.25rem">
            <button class="btn btn-ghost btn-sm" style="flex:1;text-align:left;font-size:.75rem" on:click={() => loadSavedQuery(q)}>{q.name}</button>
            <button style="background:none;border:none;cursor:pointer;color:var(--muted);font-size:.75rem" on:click={() => deleteSavedQuery(q.id)}>×</button>
          </div>
        {/each}
      </div>
    {/if}

    <!-- Filters panel -->
    {#if selectedTemplate}
      <div class="card" style="padding:.75rem">
        <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:.6rem;font-weight:600">Filters</div>

        <!-- Counterparty filter -->
        {#if selectedTemplate !== 'exposure-ranking' && selectedTemplate !== 'vol-cva'}
          <div class="form-group" style="margin-bottom:.5rem">
            <label class="form-label" style="font-size:.72rem">Counterparty</label>
            <select class="form-select" bind:value={filterCP}>
              <option value="">All counterparties</option>
              {#each counterparties as cp}
                <option value={cp.id}>{cp.name}</option>
              {/each}
            </select>
          </div>
        {/if}

        <!-- Date range -->
        <div class="form-group" style="margin-bottom:.5rem">
          <label class="form-label" style="font-size:.72rem">From</label>
          <input class="form-input" type="datetime-local" bind:value={filterFrom} style="font-size:.78rem" />
        </div>
        <div class="form-group" style="margin-bottom:.5rem">
          <label class="form-label" style="font-size:.72rem">To</label>
          <input class="form-input" type="datetime-local" bind:value={filterTo} style="font-size:.78rem" />
        </div>

        <!-- Template-specific filters -->
        {#if selectedTemplate === 'exposure-ranking'}
          <div class="form-group" style="margin-bottom:.5rem">
            <label class="form-label" style="font-size:.72rem">Min CVA</label>
            <input class="form-input" type="number" bind:value={filterMinCva} min="0" step="0.001" style="font-size:.78rem" />
          </div>
        {/if}

        {#if selectedTemplate === 'margin-activity'}
          <div class="form-group" style="margin-bottom:.5rem">
            <label class="form-label" style="font-size:.72rem">Status</label>
            <select class="form-select" bind:value={filterStatus}>
              <option value="">All statuses</option>
              <option value="PENDING">Pending</option>
              <option value="ACKNOWLEDGED">Acknowledged</option>
              <option value="SETTLED">Settled</option>
              <option value="DISPUTED">Disputed</option>
            </select>
          </div>
        {/if}

        {#if selectedTemplate === 'risk-timeline'}
          <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem;font-size:.78rem">
            <input type="checkbox" id="stressed" bind:checked={filterStressed} />
            <label for="stressed" style="cursor:pointer;color:var(--text-2)">Stressed runs only</label>
          </div>
        {/if}

        <!-- Limit -->
        <div class="form-group" style="margin-bottom:.75rem">
          <label class="form-label" style="font-size:.72rem">Max rows</label>
          <select class="form-select" bind:value={filterLimit}>
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={200}>200</option>
            <option value={500}>500</option>
          </select>
        </div>

        <button class="btn btn-primary w-full" on:click={runQuery} disabled={running}>
          {#if running}<span class="spinner" style="width:12px;height:12px"></span>{/if}
          Run Query
        </button>
        <button class="btn btn-ghost btn-sm w-full" style="margin-top:.4rem" on:click={() => showSaveQueryModal = true} disabled={!selectedTemplate}>☆ Bookmark</button>
      </div>
    {/if}

    <!-- Template description -->
    {#if template}
      <div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:.75rem;font-size:.76rem;color:var(--muted);line-height:1.5">
        <div style="color:var(--text);font-weight:500;margin-bottom:.3rem">{template.label}</div>
        {template.description}
      </div>
    {/if}

  </div>

  <!-- ── Right panel: results ────────────────────────────────────────────── -->
  <div style="display:flex;flex-direction:column;gap:.75rem">

    {#if !selectedTemplate}
      <!-- Empty state -->
      <div class="card" style="padding:3rem;text-align:center">
        <div style="font-size:2rem;margin-bottom:.75rem;opacity:.3">⬡</div>
        <div style="font-size:1rem;font-weight:600;color:var(--text);margin-bottom:.4rem">Select a query template</div>
        <div style="font-size:.82rem;color:var(--muted);max-width:360px;margin:0 auto;line-height:1.6">
          Choose one of the 5 curated analytics templates on the left, configure your filters, then click Run Query to explore the TimescaleDB risk database.
        </div>
        <div style="margin-top:1.5rem;display:flex;flex-wrap:wrap;gap:.5rem;justify-content:center">
          {#each TEMPLATES as t}
            <button class="btn btn-ghost btn-sm" on:click={() => selectTemplate(t.id)} style="border-color:{t.color}22;color:{t.color}">
              {t.label}
            </button>
          {/each}
        </div>
      </div>

    {:else if queryError}
      <div class="alert alert-error">{queryError}</div>

    {:else if rows.length === 0 && !running}
      <div class="card" style="padding:2rem;text-align:center;color:var(--muted)">
        {#if meta}
          No data matched your filters. Try widening the date range or removing filters.
        {:else}
          Configure filters on the left and click <strong style="color:var(--text)">Run Query</strong>.
        {/if}
      </div>

    {:else}
      <!-- Chart -->
      <div class="card">
        <div class="card-header">
          <span class="card-title" style="color:{template?.color}">{template?.label} — Visualization</span>
          {#if meta}<span class="badge badge-muted">{meta.row_count} rows</span>{/if}
        </div>
        <div style="position:relative;height:240px">
          <canvas bind:this={chartCanvas}></canvas>
          {#if running}
            <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.4)">
              <div class="spinner"></div>
            </div>
          {/if}
        </div>
      </div>

      <!-- Summary bar (margin-activity only) -->
      {#if summary && selectedTemplate === 'margin-activity'}
        <div class="card">
          <div class="card-header"><span class="card-title">Summary</span></div>
          <div style="display:flex;flex-wrap:wrap;gap:1rem">
            <div>
              <div style="font-size:.7rem;color:var(--muted)">Total Amount</div>
              <div style="font-weight:700;font-size:1.1rem">{(+(summary.total_amount ?? 0)).toLocaleString(undefined,{maximumFractionDigits:0})}</div>
            </div>
            {#each Object.entries((summary.status_breakdown as Record<string,number>) ?? {}) as [s, n]}
              <div>
                <div style="font-size:.7rem;color:var(--muted)">{s}</div>
                <div style="font-weight:700">{n}</div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Data table -->
      <div class="card">
        <div class="card-header"><span class="card-title">Data</span>
          <div style="display:flex;align-items:center;gap:.5rem">
            <span style="font-size:.75rem;color:var(--muted)">{rows.length} rows</span>
            <button class="btn btn-ghost btn-sm" on:click={downloadCSV}>⬇ CSV</button>
          </div>
        </div>
        <div class="table-wrap" style="max-height:400px;overflow-y:auto">
          <table>
            <thead>
              <tr>
                {#each columns as col}<th>{col.label}</th>{/each}
              </tr>
            </thead>
            <tbody>
              {#each pagedRows as row}
                <tr>
                  {#each columns as col}
                    {@const raw = (row as any)[col.key]}
                    {@const val = col.fmt ? col.fmt(raw) : String(raw ?? '—')}
                    <td>
                      {#if col.key === 'status'}
                        <span class="badge {statusBadge(val)}">{val}</span>
                      {:else if col.key === 'is_stressed'}
                        {#if raw}<span class="badge badge-amber">Stressed</span>{:else}<span class="badge badge-blue">Base</span>{/if}
                      {:else if col.key === 'cva' && raw > 0.05}
                        <span style="color:var(--red);font-weight:600">{val}</span>
                      {:else}
                        {val}
                      {/if}
                    </td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
        {#if totalPages > 1}
          <div style="display:flex;align-items:center;gap:.5rem;margin-top:.5rem;font-size:.78rem;color:var(--muted)">
            <button class="btn btn-ghost btn-sm" disabled={currentPage===0} on:click={()=>currentPage--}>‹ Prev</button>
            <span>Page {currentPage+1} / {totalPages}</span>
            <button class="btn btn-ghost btn-sm" disabled={currentPage>=totalPages-1} on:click={()=>currentPage++}>Next ›</button>
            <span style="margin-left:auto">Showing {currentPage*pageSize+1}–{Math.min((currentPage+1)*pageSize,rows.length)} of {rows.length}</span>
          </div>
        {/if}
      </div>
    {/if}

  </div>
</div>

{#if showSaveQueryModal}
  <div style="position:fixed;inset:0;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:1000">
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.25rem;width:300px">
      <div style="font-weight:600;margin-bottom:.5rem">Save Query Bookmark</div>
      <input class="form-input" bind:value={saveQueryName} placeholder="Name this query" style="margin-bottom:.5rem" />
      <div style="display:flex;gap:.5rem;justify-content:flex-end">
        <button class="btn btn-ghost btn-sm" on:click={() => showSaveQueryModal = false}>Cancel</button>
        <button class="btn btn-primary btn-sm" on:click={saveQuery} disabled={!saveQueryName.trim()}>Save</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .template-btn {
    display: flex;
    align-items: center;
    gap: .5rem;
    width: 100%;
    padding: .45rem .6rem;
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    background: transparent;
    cursor: pointer;
    font-size: .82rem;
    color: var(--text-2);
    text-align: left;
    transition: var(--transition);
  }
  .template-btn:hover {
    background: var(--surface2);
    color: var(--text);
  }
  .template-btn.active {
    background: rgba(var(--accent-rgb, 59,130,246), .08);
    border-color: var(--accent, #3b82f6);
    color: var(--text);
    border-color: var(--accent);
  }
  .template-btn.active { border-color: var(--accent); }
</style>
