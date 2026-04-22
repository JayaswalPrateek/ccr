<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import { currentUser } from '$lib/state';
  import { goto } from '$app/navigation';
  import type { AuditLogItem, User } from '$lib/types';

  let users:     User[]         = [];
  let auditLog:  AuditLogItem[] = [];
  let loading    = true;
  let error      = '';
  let creating   = false;
  let newUser    = { username: '', email: '', password: '', role: 'AUDITOR' };
  let activeTab: 'users' | 'audit' = 'users';

  onMount(async () => {
    if ($currentUser?.role !== 'ADMIN') { goto('/dashboard'); return; }
    try {
      [users, auditLog] = await Promise.all([
        api.listUsers(),
        api.getAuditLog({ limit: 100 }),
      ]);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load';
    } finally {
      loading = false;
    }
  });

  async function createUser() {
    try {
      const u = await api.registerUser(newUser);
      users = [...users, u];
      creating = false;
      newUser = { username: '', email: '', password: '', role: 'AUDITOR' };
    } catch (e) { error = e instanceof Error ? e.message : 'Error'; }
  }

  async function toggleActive(user: User) {
    try {
      const updated = await api.updateUser(user.id, { is_active: !user.is_active });
      users = users.map((u) => (u.id === updated.id ? updated : u));
    } catch (e) { error = e instanceof Error ? e.message : 'Error'; }
  }

  async function changeRole(user: User, role: string) {
    try {
      const updated = await api.updateUser(user.id, { role });
      users = users.map((u) => (u.id === updated.id ? updated : u));
    } catch (e) { error = e instanceof Error ? e.message : 'Error'; }
  }

  async function refreshAudit() {
    auditLog = await api.getAuditLog({ limit: 200 });
  }

  const roleBadge: Record<string, string> = {
    ADMIN: 'badge-red', RISK_MANAGER: 'badge-blue', AUDITOR: 'badge-muted',
  };
</script>

<svelte:head><title>Admin — CCR Engine</title></svelte:head>

<div class="page-header">
  <div>
    <div class="page-title">Administration</div>
    <div class="page-sub">User management and audit log</div>
  </div>
</div>

{#if error}<div class="alert alert-error">{error}</div>{/if}

<!-- Tabs -->
<div style="display:flex;gap:.4rem;margin-bottom:1rem">
  <button class="btn btn-sm {activeTab === 'users' ? 'btn-primary' : 'btn-ghost'}" on:click={() => activeTab = 'users'}>Users</button>
  <button class="btn btn-sm {activeTab === 'audit' ? 'btn-primary' : 'btn-ghost'}" on:click={() => activeTab = 'audit'}>Audit Log</button>
</div>

{#if loading}
  <div style="padding:2rem;text-align:center"><div class="spinner"></div></div>
{:else if activeTab === 'users'}

  <!-- User management -->
  <div class="card" style="margin-bottom:1rem">
    <div class="card-header">
      <span class="card-title">Users ({users.length})</span>
      <button class="btn btn-primary btn-sm" on:click={() => creating = !creating}>
        {creating ? 'Cancel' : '+ New User'}
      </button>
    </div>

    {#if creating}
      <div style="border:1px solid var(--border);border-radius:var(--radius-sm);padding:.75rem;margin-bottom:.75rem;background:var(--surface2)">
        <div class="form-row">
          <div class="form-group"><label class="form-label" for="nu-username">Username</label><input id="nu-username" class="form-input" bind:value={newUser.username} /></div>
          <div class="form-group"><label class="form-label" for="nu-email">Email</label><input id="nu-email" class="form-input" type="email" bind:value={newUser.email} /></div>
        </div>
        <div class="form-row">
          <div class="form-group"><label class="form-label" for="nu-password">Password</label><input id="nu-password" class="form-input" type="password" bind:value={newUser.password} /></div>
          <div class="form-group">
            <label class="form-label" for="nu-role">Role</label>
            <select id="nu-role" class="form-select" bind:value={newUser.role}>
              <option value="AUDITOR">Auditor</option>
              <option value="RISK_MANAGER">Risk Manager</option>
              <option value="ADMIN">Admin</option>
            </select>
          </div>
        </div>
        <button class="btn btn-success btn-sm" on:click={createUser}>Create User</button>
      </div>
    {/if}

    <div class="table-wrap">
      <table>
        <thead><tr><th>Username</th><th>Email</th><th>Role</th><th>Active</th><th>Created</th><th>Actions</th></tr></thead>
        <tbody>
          {#each users as user}
            <tr class:inactive={!user.is_active}>
              <td style="font-weight:500">{user.username}</td>
              <td class="text-muted">{user.email}</td>
              <td>
                <select
                  class="form-select"
                  style="width:auto;padding:.2rem .5rem;font-size:.75rem"
                  on:change={(e) => changeRole(user, (e.target as HTMLSelectElement).value)}
                  disabled={user.id === $currentUser?.id}
                >
                  <option value="AUDITOR" selected={user.role === 'AUDITOR'}>Auditor</option>
                  <option value="RISK_MANAGER" selected={user.role === 'RISK_MANAGER'}>Risk Manager</option>
                  <option value="ADMIN" selected={user.role === 'ADMIN'}>Admin</option>
                </select>
              </td>
              <td>
                <button
                  class="btn btn-sm {user.is_active ? 'btn-ghost' : 'btn-danger'}"
                  on:click={() => toggleActive(user)}
                  disabled={user.id === $currentUser?.id}
                >
                  {user.is_active ? 'Active' : 'Inactive'}
                </button>
              </td>
              <td class="text-muted text-sm">{new Date(user.created_at).toLocaleDateString()}</td>
              <td class="text-muted text-sm">{user.last_login ? new Date(user.last_login).toLocaleString() : '—'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>

{:else}

  <!-- Audit log -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">Audit Log ({auditLog.length})</span>
      <button class="btn btn-ghost btn-sm" on:click={refreshAudit}>Refresh</button>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Time</th><th>User</th><th>Action</th><th>Resource</th><th>Detail</th>
          </tr>
        </thead>
        <tbody>
          {#each auditLog as entry}
            <tr>
              <td class="text-muted text-xs" style="white-space:nowrap">{new Date(entry.time).toLocaleString()}</td>
              <td class="text-sm">{entry.user_id?.slice(0,8) ?? '—'}</td>
              <td>
                <span class="badge badge-blue">{entry.action}</span>
              </td>
              <td class="text-sm text-muted">{entry.resource_type}{entry.resource_id ? ` / ${entry.resource_id.slice(0,8)}` : ''}</td>
              <td class="text-xs text-muted" style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
                {entry.detail ? JSON.stringify(entry.detail).slice(0, 80) : '—'}
              </td>
            </tr>
          {/each}
          {#if auditLog.length === 0}
            <tr><td colspan="5" style="text-align:center;color:var(--muted);padding:1rem">No audit entries</td></tr>
          {/if}
        </tbody>
      </table>
    </div>
  </div>

{/if}

<style>
  tr.inactive { opacity: .5; }
</style>
