/**
 * WebSocket clients for real-time simulation progress and live price ticks.
 */

// ── SimulationWS ──────────────────────────────────────────────────────────────

export interface ProgressMsg {
  type: 'progress';
  timestep: number;
  total: number;
  pfe_so_far: number;
  pct: number;
}

export interface ResultMsg {
  type: 'result';
  result: {
    base: {
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
    };
    stressed?: ResultMsg['result']['base'];
    success: boolean;
    error_msg: string;
  };
}

export interface ErrorMsg {
  type: 'error';
  detail: string;
}

export class SimulationWS {
  private ws: WebSocket | null = null;

  run(
    token: string,
    request: unknown,
    onProgress: (pct: number, pfeSoFar: number) => void,
    onResult: (result: ResultMsg['result']) => void,
    onError: (msg: string) => void,
  ): void {
    this.close();

    const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/simulate`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      // Message 1: auth token
      this.ws!.send(JSON.stringify({ token }));
      // Message 2: simulation request (server reads these sequentially)
      this.ws!.send(JSON.stringify(request));
      requestSent = true;
    };

    // After the token is sent the server immediately waits for the request
    // (no ack message). Send the request right after open.
    let requestSent = false;

    this.ws.onmessage = (ev) => {
      if (!requestSent) {
        // Should not happen before we send, but handle gracefully.
        return;
      }
      const data = JSON.parse(ev.data) as ProgressMsg | ResultMsg | ErrorMsg;

      if (data.type === 'progress') {
        onProgress(data.pct, data.pfe_so_far);
      } else if (data.type === 'result') {
        onResult(data.result);
        this.close();
      } else if (data.type === 'error') {
        onError(data.detail);
        this.close();
      }
    };

    this.ws.onerror = () => onError('WebSocket connection error');
    this.ws.onclose = (ev) => {
      if (ev.code === 4001) {
        onError('Unauthorized — please log in again');
      } else if (ev.code !== 1000) {
        // Any non-normal close (server error, network drop, etc.) must surface
        // so the caller can reset simRunning; otherwise the UI hangs forever.
        onError(`Connection closed unexpectedly (code ${ev.code})`);
      }
    };
  }

  close(): void {
    if (this.ws) {
      this.ws.onmessage = null;
      this.ws.onerror = null;
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }
}

// ── PriceTickWS ───────────────────────────────────────────────────────────────

const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECTS = 10;

export class PriceTickWS {
  private ws: WebSocket | null = null;
  private token = '';
  private cb: ((symbol: string, price: number, ts: number) => void) | null = null;
  private reconnects = 0;
  private stopped = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  connect(
    token: string,
    onTick: (symbol: string, price: number, ts: number) => void,
  ): void {
    this.token = token;
    this.cb = onTick;
    this.stopped = false;
    this.reconnects = 0;
    this._open();
  }

  private _open(): void {
    if (this.stopped) return;
    const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/prices`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.reconnects = 0;
      this.ws!.send(JSON.stringify({ token: this.token }));
    };

    this.ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'tick') {
          const ts: number = msg.ts ?? Date.now() / 1000;
          for (const [sym, price] of Object.entries(msg.data as Record<string, number>)) {
            this.cb?.(sym, price, ts);
          }
        }
      } catch {}
    };

    this.ws.onerror = () => this._reconnect();
    this.ws.onclose = (ev) => {
      if (!this.stopped && ev.code !== 4001) this._reconnect();
    };
  }

  private _reconnect(): void {
    if (this.stopped || this.reconnects >= MAX_RECONNECTS) return;
    this.reconnects++;
    this.reconnectTimer = setTimeout(
      () => this._open(),
      RECONNECT_DELAY_MS * Math.min(this.reconnects, 4),
    );
  }

  disconnect(): void {
    this.stopped = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.ws) {
      this.ws.onmessage = null;
      this.ws.onerror = null;
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }
}
