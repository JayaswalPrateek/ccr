<script lang="ts">
  import { goto } from '$app/navigation';
  import { login } from '$lib/auth';

  let username = '';
  let password = '';
  let error    = '';
  let loading  = false;

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    error   = '';
    loading = true;
    try {
      await login(username, password);
      goto('/dashboard');
    } catch (err: unknown) {
      error = err instanceof Error ? err.message : 'Login failed';
    } finally {
      loading = false;
    }
  }
</script>

<svelte:head><title>Sign In — CCR Engine</title></svelte:head>

<div class="login-wrap">
  <div class="login-card">
    <div class="login-logo">CCR</div>
    <div class="login-subtitle">Counterparty Credit Risk Platform</div>

    {#if error}
      <div class="alert alert-error">{error}</div>
    {/if}

    <form on:submit={submit}>
      <div class="form-group">
        <label class="form-label" for="username">Username</label>
        <input
          id="username"
          class="form-input"
          type="text"
          bind:value={username}
          autocomplete="username"
          placeholder="admin"
          required
        />
      </div>

      <div class="form-group">
        <label class="form-label" for="password">Password</label>
        <input
          id="password"
          class="form-input"
          type="password"
          bind:value={password}
          autocomplete="current-password"
          placeholder="••••••••"
          required
        />
      </div>

      <button class="btn btn-primary w-full" type="submit" disabled={loading}>
        {#if loading}<span class="spinner" style="width:14px;height:14px"></span>{/if}
        Sign In
      </button>
    </form>

    <div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid var(--border);font-size:.72rem;color:var(--muted);text-align:center;">
      Default credentials: <code>admin / admin123</code>
    </div>
  </div>
</div>

<style>
  .login-wrap {
    min-height: 100vh;
    background: var(--bg);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
  }
  .login-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 2rem;
    width: 100%;
    max-width: 380px;
  }
  .login-logo {
    font-size: 2rem;
    font-weight: 800;
    color: var(--green);
    letter-spacing: .1em;
    text-align: center;
    margin-bottom: .25rem;
  }
  .login-subtitle {
    text-align: center;
    color: var(--muted);
    font-size: .78rem;
    margin-bottom: 1.75rem;
  }
</style>
