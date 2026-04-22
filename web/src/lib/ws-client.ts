/**
 * WebSocket client for real-time simulation progress.
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

