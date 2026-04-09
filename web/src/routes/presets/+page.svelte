<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { api } from '$lib/api';
  import { currentUser } from '$lib/state';
  import type { Counterparty, SimPreset } from '$lib/types';

  let presets:        SimPreset[]      = [];
  let recentPresets:  SimPreset[]      = [];
  let counterparties: Counterparty[]   = [];
  let loading      = true;
  let error        = '';
  let saving       = false;
  let deleteTarget: string | null = null;
  let confirmingDelete = false;

  // ── Edit / Create form ─────────────────────────────────────────────────────
  let editId:     string | null = null;   // null = create new
  let editName    = '';
  let editDesc    = '';
  let editCpId    = '';
  let editShared  = false;
  let editParams  = '{}';
  let editStress  = '';
  let editError   = '';
  let paramsValid = true;

  $: {
    try { JSON.parse(editParams); paramsValid = true; }
    catch { paramsValid = false; }
  }

  onMount(async () => {
    try {
      [presets, counterparties, recentPresets] = await Promise.all([
        api.listPresets(),
        api.listCounterparties(),
        api.recentPresets(5),
      ]);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load';
    } finally {
      loading = false;
    }
  });

  async function reload() {
    try {
      [presets, recentPresets] = await Promise.all([
        api.listPresets(),
        api.recentPresets(5),
      ]);
    } catch (_) {}
  }

  // ── Open edit form ─────────────────────────────────────────────────────────
  function openCreate() {
    editId    = null;
    editName  = '';
    editDesc  = '';
    editCpId  = '';
    editShared= false;
    editParams= '{}';
    editStress= '';
    editError = '';
  }

  function openEdit(p: SimPreset) {
    editId     = p.id;
    editName   = p.name;
    editDesc   = p.description ?? '';
    editCpId   = p.counterparty_id ?? '';
    editShared = p.is_shared;
    editParams = JSON.stringify(p.params_json, null, 2);
    editStress = p.stress_json ? JSON.stringify(p.stress_json, null, 2) : '';
    editError  = '';
  }

  function cancelEdit() {
    editId    = null;
    editName  = '';
    editError = '';
  }

  // ── Save ──────────────────────────────────────────────────────────────────
  async function save() {
    if (!editName.trim()) { editError = 'Name is required'; return; }
    if (!paramsValid)     { editError = 'Parameters JSON is invalid'; return; }

    saving    = true;
    editError = '';
    try {
      const payload = {
        name:            editName.trim(),
        description:     editDesc.trim() || undefined,
        counterparty_id: editCpId || undefined,
        params_json:     JSON.parse(editParams),
        stress_json:     editStress.trim() ? JSON.parse(editStress) : undefined,
        is_shared:       editShared,
      };
      if (editId) {
        await api.updatePreset(editId, payload);
      } else {
        await api.createPreset(payload);
      }
      await reload();
      cancelEdit();
    } catch (e) {
      editError = e instanceof Error ? e.message : 'Save failed';
    } finally {
      saving = false;
    }
  }

  // ── Delete ─────────────────────────────────────────────────────────────────
  async function deletePreset() {
    if (!deleteTarget) return;
    try {
      await api.deletePreset(deleteTarget);
      await reload();
      deleteTarget     = null;
      confirmingDelete = false;
      if (editId === deleteTarget) cancelEdit();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Delete failed';
    }
  }

  // ── Load into Simulator ────────────────────────────────────────────────────
  async function loadIntoSimulator(p: SimPreset) {
    try {
      await api.usePreset(p.id);
    } catch (_) {}
    // Navigate to /simulate with preset_id query param
    goto(`/simulate?preset_id=${p.id}`);
  }

  // ── Export / Import ────────────────────────────────────────────────────────
  function exportPreset(p: SimPreset) {
    const blob = new Blob([JSON.stringify({ name: p.name, description: p.description, params_json: p.params_json, stress_json: p.stress_json }, null, 2)], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = `${p.name.replace(/[^a-z0-9]/gi, '-')}.ccr-preset.json`; a.click();
  }

  let importInput: HTMLInputElement;
  function importPreset() { importInput.click(); }
  async function handleImport(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      editName   = data.name ?? 'Imported Preset';
      editDesc   = data.description ?? '';
      editParams = JSON.stringify(data.params_json ?? {}, null, 2);
      editStress = data.stress_json ? JSON.stringify(data.stress_json, null, 2) : '';
      editId     = null; // create new
    } catch { error = 'Invalid preset file'; }
  }

  // ── Helpers ────────────────────────────────────────────────────────────────
  function cpName(id: string | null) {
    if (!id) return null;
    return counterparties.find((c) => c.id === id)?.name ?? id.slice(0, 8);
  }

  $: isOwner = (p: SimPreset) =>
    $currentUser?.id === p.owner_id || $currentUser?.role === 'ADMIN';

  $: sharedPresets = presets.filter((p) => p.is_shared && p.owner_id !== $currentUser?.id);
  $: myPresets     = presets.filter((p) => p.owner_id === $currentUser?.id);
</script>

<svelte:head><title>Simulation Presets — CCR Engine</title></svelte:head>

<div class="page-header">
  <div>
    <div class="page-title">Simulation Presets</div>
    <div class="page-sub">Save, share, and reuse named simulation scenarios</div>
  </div>
  <div style="display:flex;gap:.5rem">
    <input bind:this={importInput} type="file" accept=".json" style="display:none" on:change={handleImport} />
    <button class="btn btn-ghost btn-sm" on:click={importPreset}>⬆ Import</button>
    <button class="btn btn-primary btn-sm" on:click={openCreate}>+ New Preset</button>
  </div>
</div>

{#if error}<div class="alert alert-error">{error}</div>{/if}

<div style="display:grid;grid-template-columns:1fr 380px;gap:1rem;align-items:start">

  <!-- ── Preset list ──────────────────────────────────────────────────────── -->
  <div style="display:flex;flex-direction:column;gap:.75rem">

    <!-- Recently used -->
    {#if recentPresets.length > 0}
      <div class="card">
        <div class="card-header">
          <span class="card-title">Recently Used</span>
          <span class="badge badge-muted">Quick access</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:.4rem">
          {#each recentPresets as p}
            <button
              class="btn btn-ghost btn-sm"
              style="border-color:rgba(59,130,246,.3)"
              on:click={() => loadIntoSimulator(p)}
              title="Last used: {p.last_used_at ? new Date(p.last_used_at).toLocaleString() : 'never'}"
            >
              {p.name}
              <span class="badge badge-muted" style="margin-left:.3rem;font-size:.6rem">{p.use_count}×</span>
            </button>
          {/each}
        </div>
      </div>
    {/if}

    <!-- My presets -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">My Presets</span>
        <span class="badge badge-muted">{myPresets.length}</span>
      </div>
      {#if loading}
        <div style="text-align:center;padding:1.5rem"><div class="spinner"></div></div>
      {:else if myPresets.length === 0}
        <div style="color:var(--muted);font-size:.82rem;padding:.5rem 0">
          No presets yet. Click <strong style="color:var(--text)">+ New Preset</strong> to create one.
        </div>
      {:else}
        <div style="display:flex;flex-direction:column;gap:.4rem">
          {#each myPresets as p}
            <div class="preset-row" class:active={editId === p.id}>
              <div style="flex:1;min-width:0">
                <div style="display:flex;align-items:center;gap:.4rem;flex-wrap:wrap">
                  <span style="font-weight:500;font-size:.85rem">{p.name}</span>
                  {#if p.is_shared}<span class="badge badge-blue" style="font-size:.6rem">Shared</span>{/if}
                  {#if p.counterparty_id}<span class="badge badge-muted" style="font-size:.6rem">{cpName(p.counterparty_id)}</span>{/if}
                </div>
                {#if p.description}
                  <div style="font-size:.75rem;color:var(--muted);margin-top:.15rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{p.description}</div>
                {/if}
                <div style="font-size:.7rem;color:var(--muted);margin-top:.2rem">
                  Used {p.use_count}× ·
                  {p.last_used_at ? 'Last ' + new Date(p.last_used_at).toLocaleDateString() : 'Never used'} ·
                  Updated {new Date(p.updated_at).toLocaleDateString()}
                </div>
              </div>
              <div style="display:flex;gap:.3rem;flex-shrink:0">
                <button class="btn btn-primary btn-sm" on:click={() => loadIntoSimulator(p)} title="Load into Simulator">
                  Run
                </button>
                <button class="btn btn-ghost btn-sm" on:click={() => openEdit(p)}>Edit</button>
                <button class="btn btn-ghost btn-sm" on:click={() => exportPreset(p)} title="Export as JSON">⬇</button>
                <button class="btn btn-ghost btn-sm" style="color:var(--red)"
                  on:click={() => { deleteTarget = p.id; confirmingDelete = true; }}>×</button>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Shared presets from others -->
    {#if sharedPresets.length > 0}
      <div class="card">
        <div class="card-header">
          <span class="card-title">Shared by Team</span>
          <span class="badge badge-muted">{sharedPresets.length}</span>
        </div>
        <div style="display:flex;flex-direction:column;gap:.4rem">
          {#each sharedPresets as p}
            <div class="preset-row">
              <div style="flex:1;min-width:0">
                <div style="display:flex;align-items:center;gap:.4rem;flex-wrap:wrap">
                  <span style="font-weight:500;font-size:.85rem">{p.name}</span>
                  {#if p.counterparty_id}<span class="badge badge-muted" style="font-size:.6rem">{cpName(p.counterparty_id)}</span>{/if}
                </div>
                {#if p.description}
                  <div style="font-size:.75rem;color:var(--muted);margin-top:.15rem">{p.description}</div>
                {/if}
                <div style="font-size:.7rem;color:var(--muted);margin-top:.2rem">
                  Used {p.use_count}× · Updated {new Date(p.updated_at).toLocaleDateString()}
                </div>
              </div>
              <button class="btn btn-primary btn-sm" on:click={() => loadIntoSimulator(p)}>Run</button>
            </div>
          {/each}
        </div>
      </div>
    {/if}

  </div>

  <!-- ── Edit / Create form ──────────────────────────────────────────────── -->
  <div>
    <!-- Form is visible when user clicked New or Edit -->
    <div class="card" style="position:sticky;top:1rem">
      <div class="card-header">
        <span class="card-title">{editId ? 'Edit Preset' : 'New Preset'}</span>
        {#if editId}<button class="btn btn-ghost btn-sm" on:click={cancelEdit}>Cancel</button>{/if}
      </div>

      {#if editError}<div class="alert alert-error" style="margin-bottom:.75rem">{editError}</div>{/if}

      <div class="form-group" style="margin-bottom:.5rem">
        <label class="form-label" style="font-size:.75rem">Name *</label>
        <input class="form-input" bind:value={editName} placeholder="e.g. Base Case — Acme Corp" maxlength="200" />
      </div>

      <div class="form-group" style="margin-bottom:.5rem">
        <label class="form-label" style="font-size:.75rem">Description</label>
        <textarea class="form-input" bind:value={editDesc} rows="2" placeholder="Optional context for this preset" style="resize:vertical;min-height:52px"></textarea>
      </div>

      <div class="form-group" style="margin-bottom:.5rem">
        <label class="form-label" style="font-size:.75rem">Counterparty scope</label>
        <select class="form-select" bind:value={editCpId}>
          <option value="">Global (all counterparties)</option>
          {#each counterparties as cp}
            <option value={cp.id}>{cp.name}</option>
          {/each}
        </select>
      </div>

      <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem;font-size:.78rem">
        <input type="checkbox" id="shared" bind:checked={editShared} />
        <label for="shared" style="cursor:pointer;color:var(--text-2)">Share with team</label>
      </div>

      <div class="form-group" style="margin-bottom:.5rem">
        <label class="form-label" style="font-size:.75rem;display:flex;justify-content:space-between">
          <span>Simulation Parameters (JSON) *</span>
          {#if !paramsValid}<span style="color:var(--red)">Invalid JSON</span>{/if}
        </label>
        <textarea
          class="form-input"
          bind:value={editParams}
          rows="8"
          style="resize:vertical;min-height:120px;font-family:monospace;font-size:.73rem;color:{paramsValid?'var(--text)':'var(--red)'}"
          spellcheck="false"
        ></textarea>
        <div style="font-size:.68rem;color:var(--muted);margin-top:.2rem">
          Keys: num_paths, num_timesteps, sigma, mu, horizon_years, mode, grid_type, rho_wwr, recovery_rate
        </div>
      </div>

      <div class="form-group" style="margin-bottom:.75rem">
        <label class="form-label" style="font-size:.75rem">Stress Scenario (JSON, optional)</label>
        <textarea
          class="form-input"
          bind:value={editStress}
          rows="4"
          style="resize:vertical;min-height:72px;font-family:monospace;font-size:.73rem"
          placeholder={`{"vol_shock": 0.3, "hazard_rate_shock": 0.5, "label": "Stress"}`}
          spellcheck="false"
        ></textarea>
      </div>

      <div style="display:flex;gap:.5rem">
        <button class="btn btn-primary" style="flex:1" on:click={save} disabled={saving || !paramsValid || !editName.trim()}>
          {#if saving}<span class="spinner" style="width:12px;height:12px"></span>{/if}
          {editId ? 'Save Changes' : 'Create Preset'}
        </button>
        {#if editId}
          <button class="btn btn-ghost" on:click={cancelEdit}>Cancel</button>
        {/if}
      </div>

    </div>
  </div>

</div>

<!-- ── Delete confirmation modal ─────────────────────────────────────────── -->
{#if confirmingDelete}
  <div style="position:fixed;inset:0;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:1000">
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;max-width:360px;width:90%">
      <div style="font-weight:600;margin-bottom:.5rem">Delete Preset</div>
      <div style="font-size:.84rem;color:var(--muted);margin-bottom:1rem">
        Are you sure? This preset will be permanently deleted and cannot be recovered.
      </div>
      <div style="display:flex;gap:.5rem;justify-content:flex-end">
        <button class="btn btn-ghost" on:click={() => { confirmingDelete = false; deleteTarget = null; }}>Cancel</button>
        <button class="btn btn-danger" on:click={deletePreset}>Delete</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .preset-row {
    display: flex;
    align-items: center;
    gap: .5rem;
    padding: .5rem .6rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    transition: var(--transition);
  }
  .preset-row:hover { background: var(--surface2); }
  .preset-row.active { border-color: var(--blue); background: rgba(59,130,246,.05); }
</style>
