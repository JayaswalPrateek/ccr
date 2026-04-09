/**
 * Global Svelte stores — single source of truth for shared UI state.
 */

import { derived, writable } from 'svelte/store';
import type { Counterparty, MarginCall, Portfolio, SimulationResponse, User } from './types';

// ── Auth ──────────────────────────────────────────────────────────────────────

function createTokenStore() {
  const stored = typeof localStorage !== 'undefined'
    ? localStorage.getItem('ccr_token')
    : null;
  const { subscribe, set } = writable<string | null>(stored);

  return {
    subscribe,
    set(val: string | null) {
      if (typeof localStorage !== 'undefined') {
        if (val) localStorage.setItem('ccr_token', val);
        else     localStorage.removeItem('ccr_token');
      }
      set(val);
    },
  };
}

export const authToken      = createTokenStore();
export const currentUser    = writable<User | null>(null);

// ── Entity data ───────────────────────────────────────────────────────────────

export const counterparties = writable<Counterparty[]>([]);
export const portfolios     = writable<Portfolio[]>([]);
export const marginCalls    = writable<MarginCall[]>([]);

// ── Simulation state ──────────────────────────────────────────────────────────

export const latestMetrics  = writable<SimulationResponse | null>(null);
export const simProgress    = writable<number>(0);       // 0–100
export const simRunning     = writable<boolean>(false);

// ── Market / live data ────────────────────────────────────────────────────────

export const livePrices     = writable<Record<string, number>>({});

// ── Derived ───────────────────────────────────────────────────────────────────

/**
 * True when there is at least one PENDING or ACKNOWLEDGED margin call.
 * margin_required > 0 is not a breach — breach = exposure > collateral,
 * which the server evaluates and records as a MarginCall row.
 */
export const marginBreached = derived(marginCalls, ($mc) =>
  $mc.some((m) => m.status === 'PENDING' || m.status === 'ACKNOWLEDGED'),
);

/** Pending margin call count. */
export const pendingMarginCallCount = derived(marginCalls, ($mc) =>
  $mc.filter((m) => m.status === 'PENDING').length,
);

/** Last API round-trip latency in milliseconds. */
export const lastApiLatencyMs = writable<number | null>(null);

// Alert thresholds (persisted to localStorage)
function _loadThresholds() {
  try { return JSON.parse(localStorage.getItem('ccr_alert_thresholds') ?? '{}'); } catch { return {}; }
}
export const alertThresholds = writable<{ cva: number; margin: number }>(
  { cva: 0.05, margin: 0, ..._loadThresholds() }
);
alertThresholds.subscribe((v) => {
  try { localStorage.setItem('ccr_alert_thresholds', JSON.stringify(v)); } catch {}
});
