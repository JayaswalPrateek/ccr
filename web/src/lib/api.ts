/**
 * CCR API client — typed wrapper around all REST endpoints.
 * Singleton exported as `api`.
 */

import type {
  AttributionItem,
  AuditLogItem,
  ConcentrationItem,
  Counterparty,
  Derivative,
  ExposureRankRow,
  MarginActivityRow,
  MarginCall,
  MarketPriceItem,
  PfePeakRow,
  Portfolio,
  PriceHistoryItem,
  RiskTimelineRow,
  SimPreset,
  SimulationHistoryItem,
  SimulationRequest,
  SimulationResponse,
  TokenResponse,
  User,
  VolCvaRow,
} from './types';
import { lastApiLatencyMs } from './state';

// ── Error ─────────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// ── Client ────────────────────────────────────────────────────────────────────

class ApiClient {
  private token: string | null = null;

  setToken(t: string | null) {
    this.token = t;
  }

  private get headers(): Record<string, string> {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.token) h['Authorization'] = `Bearer ${this.token}`;
    return h;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const t0 = Date.now();
    const res = await fetch(path, {
      method,
      headers: this.headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    lastApiLatencyMs.set(Date.now() - t0);

    if (!res.ok) {
      let detail = res.statusText;
      try {
        const json = await res.json();
        detail = json.detail ?? JSON.stringify(json);
      } catch {}
      throw new ApiError(res.status, detail);
    }

    // 204 No Content
    if (res.status === 204) return undefined as unknown as T;
    return res.json() as Promise<T>;
  }

  // ── Auth ───────────────────────────────────────────────────────────────────

  async login(username: string, password: string): Promise<TokenResponse> {
    const form = new URLSearchParams({ username, password });
    const res = await fetch('/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form.toString(),
    });
    if (!res.ok) {
      const json = await res.json().catch(() => ({}));
      throw new ApiError(res.status, json.detail ?? 'Login failed');
    }
    return res.json();
  }

  async me(): Promise<User> {
    return this.request<User>('GET', '/api/v1/auth/me');
  }

  async listUsers(): Promise<User[]> {
    return this.request<User[]>('GET', '/api/v1/auth/users');
  }

  async registerUser(data: {
    username: string;
    email: string;
    password: string;
    role: string;
  }): Promise<User> {
    return this.request<User>('POST', '/api/v1/auth/register', data);
  }

  async updateUser(id: string, data: { role?: string; is_active?: boolean }): Promise<User> {
    return this.request<User>('PUT', `/api/v1/auth/users/${id}`, data);
  }

  // ── Simulation ──────────────────────────────────────────────────────────────

  async runSimulation(req: SimulationRequest): Promise<SimulationResponse> {
    return this.request<SimulationResponse>('POST', '/api/v1/simulate', req);
  }

  async getSimHistory(params?: {
    counterparty_id?: string;
    limit?: number;
  }): Promise<SimulationHistoryItem[]> {
    const qs = new URLSearchParams();
    if (params?.counterparty_id) qs.set('counterparty_id', params.counterparty_id);
    if (params?.limit)           qs.set('limit', String(params.limit));
    return this.request<SimulationHistoryItem[]>(
      'GET', `/api/v1/simulate/history?${qs}`,
    );
  }

  async compareSimulations(run_ids: string[]): Promise<SimulationHistoryItem[]> {
    return this.request<SimulationHistoryItem[]>(
      'POST', '/api/v1/simulate/compare', { run_ids },
    );
  }

  exportPDFUrl(run_id: string): string {
    return `/api/v1/simulate/${run_id}/export/pdf`;
  }

  exportCSVUrl(run_id: string): string {
    return `/api/v1/simulate/${run_id}/export/csv`;
  }

  async downloadBlob(url: string): Promise<Blob> {
    const res = await fetch(url, { headers: this.headers });
    if (!res.ok) throw new ApiError(res.status, 'Download failed');
    return res.blob();
  }

  // ── Counterparties ──────────────────────────────────────────────────────────

  async listCounterparties(): Promise<Counterparty[]> {
    return this.request<Counterparty[]>('GET', '/api/v1/counterparties');
  }

  async createCounterparty(data: Partial<Counterparty>): Promise<Counterparty> {
    return this.request<Counterparty>('POST', '/api/v1/counterparties', data);
  }

  async getCounterparty(id: string): Promise<Counterparty> {
    return this.request<Counterparty>('GET', `/api/v1/counterparties/${id}`);
  }

  async updateCounterparty(id: string, data: Partial<Counterparty>): Promise<Counterparty> {
    return this.request<Counterparty>('PUT', `/api/v1/counterparties/${id}`, data);
  }

  async deleteCounterparty(id: string): Promise<void> {
    return this.request<void>('DELETE', `/api/v1/counterparties/${id}`);
  }

  async getCounterpartySummary(id: string): Promise<{
    total_runs: number; avg_cva: number; latest_cva: number | null;
    total_margin_called: number; pending_calls: number; settled_calls: number; total_derivatives: number;
  }> {
    return this.request('GET', `/api/v1/counterparties/${id}/summary`);
  }

  async triggerAutoRun(): Promise<{ counterparty_id: string; counterparty_name: string; success: boolean; cva: number; margin_required: number; error?: string }[]> {
    return this.request('POST', '/api/v1/simulate/auto-run');
  }

  // ── Portfolios ──────────────────────────────────────────────────────────────

  async listPortfolios(counterparty_id?: string): Promise<Portfolio[]> {
    const qs = counterparty_id ? `?counterparty_id=${counterparty_id}` : '';
    return this.request<Portfolio[]>('GET', `/api/v1/portfolios${qs}`);
  }

  async createPortfolio(data: Partial<Portfolio>): Promise<Portfolio> {
    return this.request<Portfolio>('POST', '/api/v1/portfolios', data);
  }

  async getPortfolio(id: string): Promise<Portfolio> {
    return this.request<Portfolio>('GET', `/api/v1/portfolios/${id}`);
  }

  async updatePortfolio(id: string, data: Partial<Portfolio>): Promise<Portfolio> {
    return this.request<Portfolio>('PUT', `/api/v1/portfolios/${id}`, data);
  }

  async deletePortfolio(id: string): Promise<void> {
    return this.request<void>('DELETE', `/api/v1/portfolios/${id}`);
  }

  // ── Derivatives ─────────────────────────────────────────────────────────────

  async addDerivative(portfolio_id: string, data: Partial<Derivative>): Promise<Derivative> {
    return this.request<Derivative>(
      'POST', `/api/v1/portfolios/${portfolio_id}/derivatives`, data,
    );
  }

  async deleteDerivative(portfolio_id: string, deriv_id: string): Promise<void> {
    return this.request<void>(
      'DELETE', `/api/v1/portfolios/${portfolio_id}/derivatives/${deriv_id}`,
    );
  }

  // ── Margin calls ────────────────────────────────────────────────────────────

  async listMarginCalls(params?: {
    status?: string;
    counterparty_id?: string;
    limit?: number;
  }): Promise<MarginCall[]> {
    const qs = new URLSearchParams();
    if (params?.status)          qs.set('status',          params.status);
    if (params?.counterparty_id) qs.set('counterparty_id', params.counterparty_id);
    if (params?.limit)           qs.set('limit',           String(params.limit));
    return this.request<MarginCall[]>('GET', `/api/v1/margin-calls?${qs}`);
  }

  async acknowledgeMarginCall(id: string): Promise<MarginCall> {
    return this.request<MarginCall>('PUT', `/api/v1/margin-calls/${id}/acknowledge`);
  }

  async settleMarginCall(id: string): Promise<MarginCall> {
    return this.request<MarginCall>('PUT', `/api/v1/margin-calls/${id}/settle`);
  }

  async notifyCounterparty(id: string): Promise<{ status: string; margin_call_id: string }> {
    return this.request('POST', `/api/v1/margin-calls/${id}/notify`);
  }

  // ── Market data ─────────────────────────────────────────────────────────────

  async getMarketPrices(): Promise<MarketPriceItem[]> {
    return this.request<MarketPriceItem[]>('GET', '/api/v1/market/prices');
  }

  async getPriceHistory(symbol: string, hours = 24): Promise<PriceHistoryItem[]> {
    return this.request<PriceHistoryItem[]>(
      'GET', `/api/v1/market/prices/${symbol}/history?hours=${hours}`,
    );
  }

  async triggerMarketRefresh(): Promise<{ status: string }> {
    return this.request<{ status: string }>('POST', '/api/v1/market/refresh');
  }

  // ── Analytics ────────────────────────────────────────────────────────────────

  async getConcentration(limit = 20): Promise<ConcentrationItem[]> {
    return this.request<ConcentrationItem[]>('GET', `/api/v1/analytics/concentration?limit=${limit}`);
  }

  async getAttribution(run_id: string): Promise<AttributionItem[]> {
    return this.request<AttributionItem[]>('GET', `/api/v1/simulate/${run_id}/attribution`);
  }

  async getMyActivity(params?: { since?: string; limit?: number }): Promise<AuditLogItem[]> {
    const qs = new URLSearchParams();
    if (params?.since)  qs.set('since',  params.since);
    if (params?.limit)  qs.set('limit',  String(params.limit));
    return this.request<AuditLogItem[]>('GET', `/api/v1/me/activity?${qs}`);
  }

  // ── Audit log ────────────────────────────────────────────────────────────────

  async getAuditLog(params?: {
    action?: string;
    resource_type?: string;
    from?: string;
    to?: string;
    limit?: number;
  }): Promise<AuditLogItem[]> {
    const qs = new URLSearchParams();
    if (params?.action)        qs.set('action',        params.action);
    if (params?.resource_type) qs.set('resource_type', params.resource_type);
    if (params?.from)          qs.set('from',          params.from);
    if (params?.to)            qs.set('to',            params.to);
    if (params?.limit)         qs.set('limit',         String(params.limit));
    return this.request<AuditLogItem[]>('GET', `/api/v1/audit-log?${qs}`);
  }

  // ── Presets ──────────────────────────────────────────────────────────────────

  async listPresets(params?: { counterparty_id?: string; include_shared?: boolean }): Promise<SimPreset[]> {
    const qs = new URLSearchParams();
    if (params?.counterparty_id)               qs.set('counterparty_id', params.counterparty_id);
    if (params?.include_shared !== undefined)   qs.set('include_shared', String(params.include_shared));
    return this.request<SimPreset[]>('GET', `/api/v1/presets?${qs}`);
  }

  async recentPresets(limit = 5): Promise<SimPreset[]> {
    return this.request<SimPreset[]>('GET', `/api/v1/presets/recent?limit=${limit}`);
  }

  async getPreset(id: string): Promise<SimPreset> {
    return this.request<SimPreset>('GET', `/api/v1/presets/${id}`);
  }

  async createPreset(data: {
    name: string;
    description?: string;
    counterparty_id?: string;
    params_json: Record<string, unknown>;
    stress_json?: Record<string, unknown>;
    is_shared?: boolean;
  }): Promise<SimPreset> {
    return this.request<SimPreset>('POST', '/api/v1/presets', data);
  }

  async updatePreset(id: string, data: {
    name: string;
    description?: string;
    counterparty_id?: string;
    params_json: Record<string, unknown>;
    stress_json?: Record<string, unknown>;
    is_shared?: boolean;
  }): Promise<SimPreset> {
    return this.request<SimPreset>('PUT', `/api/v1/presets/${id}`, data);
  }

  async deletePreset(id: string): Promise<void> {
    return this.request<void>('DELETE', `/api/v1/presets/${id}`);
  }

  async usePreset(id: string): Promise<SimPreset> {
    return this.request<SimPreset>('POST', `/api/v1/presets/${id}/use`);
  }

  // ── Query Builder ─────────────────────────────────────────────────────────────

  async queryRiskTimeline(params: {
    counterparty_id?: string; from?: string; to?: string;
    stressed_only?: boolean; limit?: number;
  }): Promise<{ meta: { row_count: number; executed_at: string }; rows: RiskTimelineRow[] }> {
    const qs = new URLSearchParams();
    if (params.counterparty_id) qs.set('counterparty_id', params.counterparty_id);
    if (params.from)             qs.set('from',            params.from);
    if (params.to)               qs.set('to',              params.to);
    if (params.stressed_only)    qs.set('stressed_only',   'true');
    if (params.limit)            qs.set('limit',           String(params.limit));
    return this.request('GET', `/api/v1/query/risk-timeline?${qs}`);
  }

  async queryExposureRanking(params: {
    from?: string; to?: string; min_cva?: number; limit?: number;
  }): Promise<{ meta: { row_count: number; executed_at: string }; rows: ExposureRankRow[] }> {
    const qs = new URLSearchParams();
    if (params.from)    qs.set('from',    params.from);
    if (params.to)      qs.set('to',      params.to);
    if (params.min_cva) qs.set('min_cva', String(params.min_cva));
    if (params.limit)   qs.set('limit',   String(params.limit));
    return this.request('GET', `/api/v1/query/exposure-ranking?${qs}`);
  }

  async queryPfePeaks(params: {
    counterparty_id?: string; from?: string; to?: string; limit?: number;
  }): Promise<{ meta: { row_count: number; executed_at: string }; rows: PfePeakRow[] }> {
    const qs = new URLSearchParams();
    if (params.counterparty_id) qs.set('counterparty_id', params.counterparty_id);
    if (params.from)             qs.set('from',            params.from);
    if (params.to)               qs.set('to',              params.to);
    if (params.limit)            qs.set('limit',           String(params.limit));
    return this.request('GET', `/api/v1/query/pfe-peaks?${qs}`);
  }

  async queryMarginActivity(params: {
    counterparty_id?: string; from?: string; to?: string;
    status?: string; limit?: number;
  }): Promise<{ meta: { row_count: number; executed_at: string }; rows: MarginActivityRow[]; summary: Record<string, unknown> }> {
    const qs = new URLSearchParams();
    if (params.counterparty_id) qs.set('counterparty_id', params.counterparty_id);
    if (params.from)             qs.set('from',            params.from);
    if (params.to)               qs.set('to',              params.to);
    if (params.status)           qs.set('status',          params.status);
    if (params.limit)            qs.set('limit',           String(params.limit));
    return this.request('GET', `/api/v1/query/margin-activity?${qs}`);
  }

  async queryVolCva(params: {
    from?: string; to?: string; limit?: number;
  }): Promise<{ meta: { row_count: number; executed_at: string }; rows: VolCvaRow[] }> {
    const qs = new URLSearchParams();
    if (params.from)  qs.set('from',  params.from);
    if (params.to)    qs.set('to',    params.to);
    if (params.limit) qs.set('limit', String(params.limit));
    return this.request('GET', `/api/v1/query/vol-cva?${qs}`);
  }

  // ── Health ───────────────────────────────────────────────────────────────────

  async health(): Promise<{ status: string; engine: { arch: string; simd_lanes: number } }> {
    return this.request('GET', '/api/v1/health');
  }
}

export const api = new ApiClient();
