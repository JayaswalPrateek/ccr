"""
GBM-based mock tick generator seeded from the last real yfinance price.

These are NOT real prices.  The UI labels all ticks from this generator
as "Demo Ticks" so the academic / demo nature is transparent.

MockTickGenerator.next_tick(symbol)     → next simulated price (~1.5 s cadence)
MockTickGenerator.next_hazard_rate(id) → mean-reverting CDS spread proxy
"""

from __future__ import annotations

import math
import random
from typing import Dict, List


# One "tick" represents roughly 1.5 seconds of trading time.
# 252 trading days × 390 minutes × 40 ticks/min ≈ 3.9M ticks/year.
_DT = 1.0 / (252.0 * 390.0 * 40.0)

# Assumed annualised vol when none is provided (20 %).
_DEFAULT_SIGMA = 0.20

# Hazard-rate mean-reversion parameters.
_HR_KAPPA   = 2.0   # speed of mean reversion
_HR_THETA   = 0.05  # long-run mean (~5 % default hazard rate)
_HR_SIGMA   = 0.02  # diffusion coefficient
_HR_MIN     = 0.001
_HR_MAX     = 0.30


def _box_muller() -> float:
    """Standard normal variate via Box-Muller (no numpy dependency here)."""
    u1 = random.random() or 1e-10   # avoid log(0)
    u2 = random.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


class MockTickGenerator:
    """Stateful GBM tick generator for a fixed set of symbols."""

    def __init__(
        self,
        symbols:      List[str],
        seed_prices:  Dict[str, float],
        sigmas:       Dict[str, float] | None = None,
    ) -> None:
        self._prices:  Dict[str, float] = {}
        self._sigmas:  Dict[str, float] = {}
        self._hazards: Dict[str, float] = {}   # keyed by counterparty id

        for sym in symbols:
            self._prices[sym] = seed_prices.get(sym, 100.0)
            self._sigmas[sym] = (sigmas or {}).get(sym, _DEFAULT_SIGMA)

    @property
    def symbols(self) -> List[str]:
        return list(self._prices.keys())

    def next_tick(self, symbol: str) -> float:
        """Advance the GBM by one time-step and return the new price.

        S_{t+dt} = S_t * exp((−σ²/2)*dt + σ*√dt*Z)

        The drift term is set to zero (risk-neutral) so tick prices stay
        approximately around the seed price over short horizons.
        """
        s     = self._prices.get(symbol, 100.0)
        sigma = self._sigmas.get(symbol, _DEFAULT_SIGMA)
        z     = _box_muller()
        s_new = s * math.exp((-0.5 * sigma * sigma) * _DT + sigma * math.sqrt(_DT) * z)
        self._prices[symbol] = s_new
        return s_new

    def next_hazard_rate(self, cp_id: str, current: float | None = None) -> float:
        """Euler-Maruyama step of an Ornstein-Uhlenbeck hazard-rate process.

        dh = κ(θ − h)dt + σ_h √dt Z,  bounded to [_HR_MIN, _HR_MAX].
        """
        h = self._hazards.get(cp_id, current if current is not None else _HR_THETA)
        z = _box_muller()
        h_new = (
            h
            + _HR_KAPPA * (_HR_THETA - h) * _DT
            + _HR_SIGMA  * math.sqrt(_DT)  * z
        )
        h_new = max(_HR_MIN, min(_HR_MAX, h_new))
        self._hazards[cp_id] = h_new
        return h_new
