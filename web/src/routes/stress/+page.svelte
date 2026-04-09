<script lang="ts">
  import { get } from 'svelte/store';
  import { authToken, latestMetrics, simProgress, simRunning } from '$lib/state';
  import { SimulationWS } from '$lib/ws-client';
  import StressScenarioForm from '$components/forms/StressScenarioForm.svelte';
  import SimParamsForm from '$components/forms/SimParamsForm.svelte';
  import ProgressBar from '$components/ui/ProgressBar.svelte';
  import MetricCard from '$components/ui/MetricCard.svelte';
  import PFEChart from '$components/charts/PFEChart.svelte';
  import EPEChart from '$components/charts/EPEChart.svelte';
  import type { SimulationRequest, SimulationResponse, StressScenarioRequest } from '$lib/types';

  let stress:     StressScenarioRequest | null = null;
  let lastRequest: SimulationRequest | null = null;
  let result:     SimulationResponse | null = null;
  let error       = '';
  let simWS:      SimulationWS | null = null;
  let formTrigger = 0;

  function handleStressApply(e: CustomEvent<StressScenarioRequest>) {
    stress = e.detail;
  }

  function handleStressClear() {
    stress = null;
    result = null;
  }

  function handleFormSubmit(e: CustomEvent<SimulationRequest>) {
    lastRequest = { ...e.detail, stress: stress ?? undefined };
    runSim(lastRequest);
  }

  function runSim(req: SimulationRequest) {
    const token = get(authToken);
    if (!token) { error = 'Not authenticated'; return; }

    error = '';
    result = null;
    simProgress.set(0);
    simRunning.set(true);

    simWS = new SimulationWS();
    simWS.run(
      token, req,
      (pct) => simProgress.set(pct),
      (r) => {
        result = r as SimulationResponse;
        simRunning.set(false);
        simProgress.set(100);
      },
      (msg) => { error = msg; simRunning.set(false); },
    );
  }

  function delta(a: number, b: number) {
    if (!b || b === 0) return 0;
    return ((a - b) / b) * 100;
  }

  $: base    = result?.base;
  $: stressed= result?.stressed;
</script>

<svelte:head><title>Stress Test — CCR Engine</title></svelte:head>

<div class="page-header">
  <div>
    <div class="page-title">Stress Testing</div>
    <div class="page-sub">Apply macroeconomic shock scenarios and compare metrics</div>
  </div>
</div>

{#if error}<div class="alert alert-error">{error}</div>{/if}

<div class="stress-layout">
  <!-- Left column -->
  <div class="stress-left">
    <div class="card" style="margin-bottom:1rem">
      <div class="card-header">
        <span class="card-title">Stress Scenario</span>
        {#if stress}
          <span class="badge badge-amber">{stress.label}</span>
        {:else}
          <span class="badge badge-muted">No stress</span>
        {/if}
      </div>
      <StressScenarioForm
        on:apply={handleStressApply}
        on:clear={handleStressClear}
      />
    </div>

    <div class="card">
      <div class="card-header"><span class="card-title">Simulation Params</span></div>
      <SimParamsForm trigger={formTrigger} on:submit={handleFormSubmit} />
      <hr class="divider" />
      <button class="btn btn-primary w-full" disabled={$simRunning} on:click={() => formTrigger++}>
        {#if $simRunning}<span class="spinner" style="width:14px;height:14px"></span>{/if}
        Run {stress ? 'Stressed' : 'Base'} Simulation
      </button>
      <ProgressBar value={$simProgress} visible={$simRunning} />
    </div>
  </div>

  <!-- Right column: comparison -->
  <div class="stress-right">
    {#if result}
      <!-- Comparison metrics grid -->
      <div class="grid-3" style="margin-bottom:1rem">
        <MetricCard
          label="CVA (base)"
          value={base?.cva.toFixed(5) ?? '—'}
        />
        <MetricCard
          label="CVA (stressed)"
          value={stressed?.cva.toFixed(5) ?? 'N/A'}
          delta={stressed && base ? delta(stressed.cva, base.cva) : 0}
        />
        <div class="card" style="display:flex;flex-direction:column;justify-content:center;align-items:center">
          {#if stressed && base}
            {@const pct = delta(stressed.cva, base.cva)}
            <div style="font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem">CVA Δ</div>
            <div style="font-size:1.6rem;font-weight:700;color:{pct > 0 ? 'var(--red)' : 'var(--green)'}">
              {pct > 0 ? '+' : ''}{pct.toFixed(1)}%
            </div>
          {:else}
            <div style="color:var(--muted);font-size:.8rem">No stressed run</div>
          {/if}
        </div>
      </div>

      <div class="grid-3" style="margin-bottom:1rem">
        <MetricCard label="Margin (base)"    value={base?.margin_required.toLocaleString(undefined,{maximumFractionDigits:0}) ?? '—'} />
        <MetricCard
          label="Margin (stressed)"
          value={stressed?.margin_required.toLocaleString(undefined,{maximumFractionDigits:0}) ?? 'N/A'}
          delta={stressed && base ? delta(stressed.margin_required, base.margin_required) : 0}
        />
        <MetricCard label="WWR-CVA (base)"   value={base?.wwr_cva.toFixed(5) ?? '—'} />
      </div>

      <!-- Charts side-by-side -->
      <div class="grid-2" style="margin-bottom:1rem">
        <div class="card">
          <div class="card-header"><span class="card-title">PFE — Base vs Stressed</span></div>
          <PFEChart
            timeGrid={base?.time_grid_years ?? []}
            pfeBase={base?.pfe_profile ?? []}
            pfeStressed={stressed?.pfe_profile ?? []}
            height={220}
          />
        </div>
        <div class="card">
          <div class="card-header"><span class="card-title">EPE — Base vs Stressed</span></div>
          <EPEChart
            timeGrid={base?.time_grid_years ?? []}
            epeBase={base?.epe_profile ?? []}
            epeStressed={stressed?.epe_profile ?? []}
            cva={base?.cva ?? 0}
            height={220}
          />
        </div>
      </div>

      <!-- Profile diff table -->
      {#if stressed}
        <div class="card">
          <div class="card-header"><span class="card-title">Profile Comparison</span></div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>T (yrs)</th>
                  <th>PFE Base</th><th>PFE Stress</th><th>PFE Δ%</th>
                  <th>EPE Base</th><th>EPE Stress</th><th>EPE Δ%</th>
                </tr>
              </thead>
              <tbody>
                {#each base?.time_grid_years ?? [] as t, i}
                  {@const pfeD = delta(stressed.pfe_profile[i] ?? 0, base?.pfe_profile[i] ?? 0)}
                  {@const epeD = delta(stressed.epe_profile[i] ?? 0, base?.epe_profile[i] ?? 0)}
                  <tr>
                    <td class="text-muted">{t.toFixed(3)}</td>
                    <td>{(base?.pfe_profile[i] ?? 0).toFixed(4)}</td>
                    <td>{(stressed.pfe_profile[i] ?? 0).toFixed(4)}</td>
                    <td style="color:{pfeD > 0 ? 'var(--red)' : 'var(--green)'}">
                      {pfeD > 0 ? '+' : ''}{pfeD.toFixed(1)}%
                    </td>
                    <td>{(base?.epe_profile[i] ?? 0).toFixed(4)}</td>
                    <td>{(stressed.epe_profile[i] ?? 0).toFixed(4)}</td>
                    <td style="color:{epeD > 0 ? 'var(--red)' : 'var(--green)'}">
                      {epeD > 0 ? '+' : ''}{epeD.toFixed(1)}%
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      {/if}

    {:else if $simRunning}
      <div class="card" style="padding:3rem;text-align:center">
        <div class="spinner" style="width:32px;height:32px;margin:0 auto 1rem"></div>
        <div style="color:var(--text-2)">Running…</div>
        <ProgressBar value={$simProgress} visible={true} />
      </div>
    {:else}
      <div class="card" style="padding:3rem;text-align:center;color:var(--muted)">
        Apply a stress scenario and run a simulation to see comparison metrics.
      </div>
    {/if}
  </div>
</div>

<style>
  .stress-layout { display: grid; grid-template-columns: 360px 1fr; gap: 1rem; align-items: start; }
  .stress-left   { overflow-y: auto; max-height: calc(100vh - 140px); }
</style>
