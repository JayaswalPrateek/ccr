// TypeScript interfaces that mirror the Python Pydantic schemas exactly.

// ── Enumerations ──────────────────────────────────────────────────────────────

export enum DerivativeType { IRS = 0, CDS = 1, FX = 2, EQUITY = 3, COMMODITY = 4 }
export enum SimMode        { REGULATORY = 0, STANDARD = 1, APPROX_FAST = 2 }
export enum GridType       { MONTHLY = 0, WEEKLY = 1, DAILY = 2, PARSIMONIOUS = 3 }
export enum CreditRating   { AAA = 0, AA = 1, A = 2, BBB = 3, BB = 4, B = 5, CCC = 6, D = 7 }

export type UserRole = 'ADMIN' | 'RISK_MANAGER' | 'AUDITOR';
export type MarginCallStatus = 'PENDING' | 'ACKNOWLEDGED' | 'SETTLED' | 'DISPUTED';
export type SimStatus = 'RUNNING' | 'DONE' | 'FAILED';
export type TriggerType = 'MANUAL' | 'SCHEDULED' | 'AUTO_RERUN';
export type ParamType = 'SPOT' | 'VOL' | 'RATE' | 'HAZARD';

// ── Auth ──────────────────────────────────────────────────────────────────────

export interface User {
  id: string;
  username: string;
  email: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
  last_login: string | null;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

// ── Simulation request types ──────────────────────────────────────────────────

export interface SimParamsRequest {
  num_paths: number;
  num_timesteps: number;
  num_assets: number;
  mu: number;
  sigma: number;
  rho_wwr: number;
  recovery_rate: number;
  horizon_years: number;
  mode: SimMode;
  grid_type: GridType;
}

export interface CounterpartyRequest {
  id: string;
  name: string;
  credit_rating: CreditRating;
  hazard_rate: number;
  recovery_rate: number;
  collateral: number;
  margin_threshold: number;
  mpor_days: number;
}

export interface DerivativeSpecRequest {
  id: string;
  type: DerivativeType;
  notional: number;
  maturity_years: number;
  underlying_price: number;
  strike: number;
  cash_flow_freq: number;
}

export interface PortfolioRequest {
  id: string;
  counterparty_id: string;
  derivatives: DerivativeSpecRequest[];
  collateral: number;
  net_value: number;
}

export interface StressScenarioRequest {
  vol_shock: number;
  fx_shock: number;
  equity_shock: number;
  interest_rate_shock: number;
  credit_spread_shock: number;
  hazard_rate_shock: number;
  jump_amplitude: number;
  label: string;
}

export interface SimulationRequest {
  sim_params: SimParamsRequest;
  counterparty: CounterpartyRequest;
  portfolio: PortfolioRequest;
  stress?: StressScenarioRequest;
  enable_wwr: boolean;
  enable_jump_diffusion: boolean;
  enable_collateral: boolean;
  deterministic_quantile: boolean;
  log_overflow_warnings: boolean;
  rng_seed: number;
  note?: string;
}

// ── Simulation response types ─────────────────────────────────────────────────

export interface RiskMetricsResponse {
  cva: number;
  wwr_cva: number;
  margin_required: number;
  pfe_profile: number[];
  epe_profile: number[];
  time_grid_years: number[];
  compute_time_us: number;
  overflow_detected: boolean;
  arch_used: string;
  paths_used: number;
}

export interface SimulationResponse {
  base: RiskMetricsResponse;
  stressed?: RiskMetricsResponse;
  success: boolean;
  error_msg: string;
}

export interface SimulationHistoryItem {
  id: string;
  run_id: string | null;
  counterparty_id: string | null;
  cva: number;
  wwr_cva: number;
  margin_required: number;
  is_stressed: boolean;
  compute_time_us: number;
  time: string;
  pfe_profile: number[];
  epe_profile: number[];
  time_grid_years: number[];
  note?: string | null;
}

// ── Entity types ──────────────────────────────────────────────────────────────

export interface Counterparty {
  id: string;
  external_id: string;
  name: string;
  credit_rating: string;
  hazard_rate: number;
  recovery_rate: number;
  collateral: number;
  margin_threshold: number;
  mpor_days: number;
  created_by: string | null;
  created_at: string;
  updated_at: string;
  portfolios?: Portfolio[];
}

export interface Portfolio {
  id: string;
  external_id: string;
  counterparty_id: string;
  collateral: number;
  net_value: number;
  auto_run: boolean;
  created_at: string;
  updated_at: string;
  derivatives?: Derivative[];
}

export interface Derivative {
  id: string;
  external_id: string;
  portfolio_id: string;
  deriv_type: string;
  notional: number;
  maturity_years: number;
  underlying_price: number;
  strike: number;
  cash_flow_freq: number;
  created_at: string;
}

export interface MarginCall {
  id: string;
  counterparty_id: string;
  simulation_run_id: string | null;
  amount: number;
  excess_exposure: number;
  status: MarginCallStatus;
  reason: string;
  issued_at: string;
  acknowledged_at: string | null;
  settled_at: string | null;
  issued_by: string | null;
  counterparty?: Counterparty;
}

// ── Market data ───────────────────────────────────────────────────────────────

export interface MarketPriceItem {
  symbol: string;
  param_type: ParamType;
  value: number;
  source: string;
  fetched_at: string;
}

export interface PriceHistoryItem {
  ts: string;
  symbol: string;
  price: number;
  source: string;
}

export interface TickData {
  type: 'tick';
  data: Record<string, number>;
  ts: number;
  note: string;
}

// ── Audit log ─────────────────────────────────────────────────────────────────

export interface AuditLogItem {
  id: string;
  time: string;
  user_id: string | null;
  action: string;
  resource_type: string;
  resource_id: string | null;
  detail: Record<string, unknown> | null;
  ip_address: string | null;
}

// ── Analytics ─────────────────────────────────────────────────────────────────

export interface ConcentrationItem {
  counterparty_id:   string;
  counterparty_name: string | null;
  cva:               number;
  margin_required:   number;
  last_run_time:     string;
}

export interface AttributionItem {
  deriv_id:       string;
  deriv_type:     string;
  notional:       number;
  maturity_years: number;
  weight:         number;
  allocated_cva:  number;
}

// ── Presets ───────────────────────────────────────────────────────────────────

export interface SimPreset {
  id:              string;
  name:            string;
  description:     string | null;
  owner_id:        string | null;
  counterparty_id: string | null;
  params_json:     Record<string, unknown>;
  stress_json:     Record<string, unknown> | null;
  is_shared:       boolean;
  use_count:       number;
  last_used_at:    string | null;
  created_at:      string;
  updated_at:      string;
}

// ── Query Builder ─────────────────────────────────────────────────────────────

export interface QueryMeta {
  template:    string;
  row_count:   number;
  executed_at: string;
}

export interface RiskTimelineRow {
  time:              string;
  counterparty_id:   string | null;
  counterparty_name: string | null;
  cva:               number;
  wwr_cva:           number;
  margin_required:   number;
  is_stressed:       boolean;
}

export interface ExposureRankRow {
  counterparty_id:   string;
  counterparty_name: string | null;
  cva:               number;
  wwr_cva:           number;
  margin_required:   number;
  run_count:         number;
  last_run_time:     string;
}

export interface PfePeakRow {
  time:              string;
  simulation_run_id: string | null;
  counterparty_id:   string | null;
  counterparty_name: string | null;
  peak_pfe:          number;
  cva:               number;
}

export interface MarginActivityRow {
  issued_at:         string;
  counterparty_id:   string;
  counterparty_name: string | null;
  amount:            number;
  excess_exposure:   number;
  status:            string;
  reason:            string;
}

export interface VolCvaRow {
  time:             string;
  sigma:            number | null;
  num_paths:        number | null;
  cva:              number;
  wwr_cva:          number;
  counterparty_id:  string | null;
}
