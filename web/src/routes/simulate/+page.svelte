<script lang="ts">
  import { onMount } from 'svelte';
  import { get } from 'svelte/store';
  import { page } from '$app/stores';
  import { authToken, latestMetrics, simProgress, simRunning } from '$lib/state';
  import { api } from '$lib/api';
  import { SimulationWS } from '$lib/ws-client';
  import SimParamsForm from '$components/forms/SimParamsForm.svelte';
  import ProgressBar from '$components/ui/ProgressBar.svelte';
  import PFEChart from '$components/charts/PFEChart.svelte';
  import EPEChart from '$components/charts/EPEChart.svelte';
  import AttributionChart from '$components/charts/AttributionChart.svelte';
  import MetricCard from '$components/ui/MetricCard.svelte';
  import type { AttributionItem, SimulationRequest, SimulationResponse } from '$lib/types';

  let result:            SimulationResponse | null = null;
  let error              = '';
  let simWS:             SimulationWS | null = null;
  let formTrigger        = 0;
  let enableJump         = false;
  let loadedPresetName   = '';

  // Preset loading from URL param
  let initialSimParams: Record<string, unknown> | null = null;
  let initialCpId = '';
  onMount(async () => {
    const presetId = $page.url.searchParams.get('preset_id');
    if (presetId) {
      try {
        const preset = await api.getPreset(presetId);
        initialSimParams = preset.params_json as Record<string, unknown>;
        loadedPresetName = preset.name;
        await api.usePreset(presetId).catch(() => {});
      } catch (_) {}
    }

    const rerunId = $page.url.searchParams.get('rerun_id');
    if (rerunId) {
      try {
        const hist = await api.getSimHistory({ limit: 100 });
        const run = hist.find(h => h.run_id === rerunId && !h.is_stressed);
        if (run?.note) loadedPresetName = `Re-run · ${run.note}`;
        else if (run) loadedPresetName = `Re-run · ${new Date(run.time).toLocaleDateString()}`;
      } catch (_) {}
    }

    // Pre-fill counterparty ID when navigating from counterparty detail page.
    const cpId = $page.url.searchParams.get('cp_id');
    if (cpId) {
      initialCpId = cpId;
      try {
        const cp = await api.getCounterparty(cpId);
        loadedPresetName = cp.name;
        initialSimParams = {
          ...(initialSimParams ?? {}),
          counterparty_id: cp.id,
          counterparty_name: cp.name,
          counterparty_hazard_rate: cp.hazard_rate,
          counterparty_recovery_rate: cp.recovery_rate,
          counterparty_collateral: cp.collateral,
          counterparty_mpor_days: cp.mpor_days,
        };
      } catch (_) {}
    }
  });

  // Save as preset UI state
  let showSaveModal    = false;
  let savePresetName   = '';
  let savePresetDesc   = '';
  let savePresetShared = false;
  let savePresetError  = '';
  let savingPreset     = false;
  let lastSubmittedParams: Record<string, unknown> | null = null;

  $: suggestedCollateral = (result?.base?.margin_required ?? 0) > 0
    ? result!.base.margin_required * 1.10
    : null;

  function handleSubmit(e: CustomEvent<SimulationRequest>) {
    const req   = e.detail;
    const token = get(authToken);
    if (!token) { error = 'Not authenticated'; return; }
    enableJump = req.enable_jump_diffusion ?? false;
    lastSubmittedParams = req.sim_params as unknown as Record<string, unknown>;

    error = '';
    result = null;
    latestRunId   = null;
    fetchingRunId = false;
    attribution   = [];
    simProgress.set(0);
    simRunning.set(true);

    simWS = new SimulationWS();
    simWS.run(
      token,
      req,
      (pct) => simProgress.set(pct),
      (r) => {
        result = r as SimulationResponse;
        latestMetrics.set(r as SimulationResponse);
        simRunning.set(false);
        simProgress.set(100);
      },
      (msg) => {
        error = msg;
        simRunning.set(false);
      },
    );
  }

  async function saveAsPreset() {
    if (!savePresetName.trim() || !lastSubmittedParams) return;
    savingPreset = true; savePresetError = '';
    try {
      await api.createPreset({
        name:        savePresetName.trim(),
        description: savePresetDesc.trim() || undefined,
        params_json: lastSubmittedParams,
        is_shared:   savePresetShared,
      });
      showSaveModal  = false;
      savePresetName = '';
      savePresetDesc = '';
    } catch (e) {
      savePresetError = e instanceof Error ? e.message : 'Save failed';
    } finally {
      savingPreset = false;
    }
  }

  // Export helpers
  function downloadPDF() {
    const runId = getLatestRunId();
    if (!runId) return;
    api.downloadBlob(api.exportPDFUrl(runId)).then((blob) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `ccr-report-${runId.slice(0, 8)}.pdf`;
      a.click();
    });
  }

  function downloadCSV() {
    const runId = getLatestRunId();
    if (!runId) return;
    api.downloadBlob(api.exportCSVUrl(runId)).then((blob) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `ccr-profile-${runId.slice(0,8)}.csv`;
      a.click();
    });
  }

  let latestRunId: string | null = null;
  let attribution: AttributionItem[] = [];
  let fetchingRunId = false;   // prevent reactive re-entry
  function getLatestRunId() { return latestRunId; }

  $: if (result?.success && !latestRunId && !fetchingRunId) {
    fetchingRunId = true;
    api.getSimHistory({ limit: 1 }).then((h) => {
      const runId = h.length > 0 ? h[0].run_id : null;
      latestRunId = runId ?? '';   // '' as sentinel — stops re-triggering
      if (runId) {
        api.getAttribution(runId).then((a) => { attribution = a; }).catch(() => {});
      }
    }).catch(() => {}).finally(() => { fetchingRunId = false; });
  }
</script>

<svelte:head><title>Simulate — CCR Engine</title></svelte:head>

<div class="page-header">
  <div>
    <div class="page-title">Monte Carlo Simulation</div>
    <div class="page-sub">Configure and run a CCR / XVA simulation</div>
  </div>
  <div style="display:flex;gap:.5rem;align-items:center">
    {#if loadedPresetName}
      <span class="badge badge-blue">Preset: {loadedPresetName}</span>
    {/if}
    <a href="/presets" class="btn btn-ghost btn-sm">Presets</a>
    {#if lastSubmittedParams}
      <button class="btn btn-ghost btn-sm" title="Copy params as JSON" on:click={() => {
        navigator.clipboard.writeText(JSON.stringify(lastSubmittedParams, null, 2));
      }}>⎘ Copy</button>
      <button class="btn btn-ghost btn-sm" on:click={() => { showSaveModal = true; savePresetName = ''; }}>
        Save as Preset
      </button>
    {/if}
  </div>
</div>

{#if error}
  <div class="alert alert-error">{error}</div>
{/if}

<div class="sim-layout">
  <!-- Left panel: form -->
  <div class="sim-left card">
    <SimParamsForm trigger={formTrigger} {initialSimParams} on:submit={handleSubmit} />
    <hr class="divider" />
    <button
      class="btn btn-primary w-full"
      disabled={$simRunning}
      on:click={() => formTrigger++}
    >
      {#if $simRunning}<span class="spinner" style="width:14px;height:14px"></span>{/if}
      Run Simulation
    </button>
    <ProgressBar value={$simProgress} visible={$simRunning} />
  </div>

  <!-- Right panel: results -->
  <div class="sim-right">
    {#if result?.success}
      <div class="grid-4 mb-4">
        <MetricCard label="CVA"      value={result.base.cva.toLocaleString(undefined, { maximumFractionDigits: 2 })} />
        <MetricCard label="WWR-CVA"  value={result.base.wwr_cva.toLocaleString(undefined, { maximumFractionDigits: 2 })} />
        <MetricCard label="Margin"   value={result.base.margin_required.toLocaleString(undefined,{maximumFractionDigits:0})} breached={result.base.margin_required > 0} />
        <MetricCard label="Compute"  value={(result.base.compute_time_us/1000).toFixed(1)} unit="ms" subtitle={result.base.arch_used} />
      </div>

      <div class="card mb-4">
        <div class="card-header">
          <span class="card-title">Potential Future Exposure</span>
          {#if latestRunId}
            <div style="display:flex;gap:.5rem">
              <button class="btn btn-ghost btn-sm" on:click={downloadPDF}>Export PDF</button>
              <button class="btn btn-ghost btn-sm" on:click={downloadCSV}>Export CSV</button>
            </div>
          {/if}
        </div>
        <PFEChart
          timeGrid={result.base.time_grid_years}
          pfeBase={result.base.pfe_profile}
          pfeStressed={result.stressed?.pfe_profile ?? []}
          height={240}
          {enableJump}
          isStressed={!!result.stressed}
        />
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">Expected Positive Exposure</span>
        </div>
        <EPEChart
          timeGrid={result.base.time_grid_years}
          epeBase={result.base.epe_profile}
          epeStressed={result.stressed?.epe_profile ?? []}
          cva={result.base.cva}
          height={240}
        />
      </div>

      {#if attribution.length > 0}
        <div class="card mt-3">
          <div class="card-header">
            <span class="card-title">CVA Attribution by Derivative</span>
            <span class="badge badge-muted">Notional-weighted approx.</span>
          </div>
          <AttributionChart items={attribution} height={Math.max(80, attribution.length * 32)} />
          <div style="font-size:.71rem;color:var(--muted);margin-top:.4rem">
            CVA<sub>i</sub> ≈ CVA<sub>total</sub> × (notional<sub>i</sub> × maturity<sub>i</sub>) / Σ(notional<sub>j</sub> × maturity<sub>j</sub>)
          </div>
        </div>
      {/if}

      {#if suggestedCollateral !== null}
        <div class="card mt-3" style="border-left:3px solid var(--green)">
          <div class="card-header">
            <span class="card-title">Suggested Collateral</span>
            <span class="badge badge-green">Recommendation</span>
          </div>
          <div>
            <div style="font-size:1.4rem;font-weight:700;color:var(--green)">
              {suggestedCollateral.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
            <div style="font-size:.78rem;color:var(--muted);margin-top:.4rem">
              Recommended collateral = margin required ({result.base.margin_required.toLocaleString(undefined, { maximumFractionDigits: 0 })}) × 1.10 buffer.
              Posting this amount provides a 10% cushion above the computed margin requirement to absorb intraday exposure moves.
            </div>
          </div>
        </div>
      {/if}

      {#if result.base.overflow_detected}
        <div class="alert alert-warn mt-3">Overflow detected in simulation paths — results may be inaccurate.</div>
      {/if}

    {:else if $simRunning}
      <div class="card" style="padding:3rem;text-align:center">
        <div class="spinner" style="width:32px;height:32px;margin:0 auto 1rem"></div>
        <div style="color:var(--text-2)">Simulation running…</div>
        <div style="color:var(--muted);font-size:.8rem;margin-top:.25rem">{$simProgress.toFixed(1)}% complete</div>
        <ProgressBar value={$simProgress} visible={true} />
      </div>
    {:else}
      <div class="card" style="padding:3rem;text-align:center;color:var(--muted)">
        Configure parameters and click <strong>Run Simulation</strong> to see results.
      </div>
    {/if}
  </div>
</div>

<!-- ── Save Preset Modal ─────────────────────────────────────────────────── -->
{#if showSaveModal}
  <div style="position:fixed;inset:0;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:1000">
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;width:360px">
      <div style="font-weight:600;margin-bottom:.75rem">Save as Preset</div>
      {#if savePresetError}<div class="alert alert-error" style="margin-bottom:.5rem">{savePresetError}</div>{/if}
      <div class="form-group" style="margin-bottom:.5rem">
        <label class="form-label" style="font-size:.75rem">Name *</label>
        <input class="form-input" bind:value={savePresetName} placeholder="e.g. High-vol stress run" maxlength="200" />
      </div>
      <div class="form-group" style="margin-bottom:.5rem">
        <label class="form-label" style="font-size:.75rem">Description</label>
        <input class="form-input" bind:value={savePresetDesc} placeholder="Optional" />
      </div>
      <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.75rem;font-size:.78rem">
        <input type="checkbox" id="sm-shared" bind:checked={savePresetShared} />
        <label for="sm-shared" style="cursor:pointer;color:var(--text-2)">Share with team</label>
      </div>
      <div style="display:flex;gap:.5rem;justify-content:flex-end">
        <button class="btn btn-ghost" on:click={() => showSaveModal = false}>Cancel</button>
        <button class="btn btn-primary" on:click={saveAsPreset} disabled={savingPreset || !savePresetName.trim()}>
          {#if savingPreset}<span class="spinner" style="width:12px;height:12px"></span>{/if}
          Save
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .sim-layout { display: grid; grid-template-columns: 380px 1fr; gap: 1rem; align-items: start; }
  .sim-left   { overflow-y: auto; max-height: calc(100vh - 140px); }
  .mb-4 { margin-bottom: 1rem; }
  .mt-3 { margin-top: .75rem; }
</style>
