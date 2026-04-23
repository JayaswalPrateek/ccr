"""
SA-CCR (BCBS 279 / Basel III CRE52) — Standardised Approach for Counterparty
Credit Risk.

EAD = 1.4 × (RC + AddOn_aggregate)

RC  = max(V − C, 0)           V = portfolio MtM approximation, C = collateral
AddOn_aggregate = Σ AddOn_i   (simple sum, conservative)
AddOn_i = |notional_i| × SF_i × MF_i
MF_i = sqrt(min(M_i, 1.0))   M_i = maturity in years
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


ALPHA = 1.4  # BCBS 279 alpha multiplier


# Supervisory factors by asset class and maturity bucket (CRE52.72)
# IRS: <1yr, 1-5yr, >5yr
_IRS_SF = {(0.0, 1.0): 0.005, (1.0, 5.0): 0.005, (5.0, 999.0): 0.015}
# CDS
_CDS_SF = {(0.0, 1.0): 0.0038, (1.0, 5.0): 0.0042, (5.0, 999.0): 0.0045}
# Others: fixed
_SF_FIXED = {
    "FX":    0.04,
    "EQ":    0.32,
    "CMDTY": 0.40,
}


def _supervisory_factor(deriv_type: str, maturity_years: float) -> float:
    dt = deriv_type.upper()
    if dt == "IRS":
        for (lo, hi), sf in _IRS_SF.items():
            if lo <= maturity_years < hi:
                return sf
        return 0.015
    if dt == "CDS":
        for (lo, hi), sf in _CDS_SF.items():
            if lo <= maturity_years < hi:
                return sf
        return 0.0045
    return _SF_FIXED.get(dt, _SF_FIXED["CMDTY"])


def _maturity_factor(maturity_years: float) -> float:
    return math.sqrt(min(maturity_years, 1.0))


@dataclass
class AddOnBreakdown:
    deriv_id:  str
    deriv_type: str
    notional:  float
    maturity_years: float
    sf:        float
    mf:        float
    add_on:    float


@dataclass
class SACCRResult:
    ead:               float
    rc:                float
    add_on_aggregate:  float
    alpha:             float = ALPHA
    breakdown:         List[AddOnBreakdown] = field(default_factory=list)


def compute_sa_ccr(
    derivatives: list[dict],   # each: {id, deriv_type, notional, maturity_years}
    collateral:  float,
    margin_required: float,
    mpor_days:   int = 10,
) -> SACCRResult:
    """
    Compute SA-CCR EAD for a list of derivatives.

    Parameters
    ----------
    derivatives     : list of dicts with id, deriv_type, notional, maturity_years
    collateral      : posted collateral (C)
    margin_required : simulation-derived margin (used to approximate V)
    mpor_days       : margin period of risk in days
    """
    # Approximate portfolio MtM from margin_required (reverse the MPOR adjustment)
    v_approx = margin_required / (1 + mpor_days / 360)

    rc = max(v_approx - collateral, 0.0)

    breakdown: list[AddOnBreakdown] = []
    add_on_total = 0.0

    for d in derivatives:
        sf  = _supervisory_factor(d.get("deriv_type", "IRS"), d.get("maturity_years", 1.0))
        mf  = _maturity_factor(d.get("maturity_years", 1.0))
        ao  = abs(d.get("notional", 0.0)) * sf * mf
        add_on_total += ao
        breakdown.append(AddOnBreakdown(
            deriv_id       = d.get("id", ""),
            deriv_type     = d.get("deriv_type", "?"),
            notional       = d.get("notional", 0.0),
            maturity_years = d.get("maturity_years", 1.0),
            sf             = sf,
            mf             = mf,
            add_on         = ao,
        ))

    ead = ALPHA * (rc + add_on_total)

    return SACCRResult(
        ead              = ead,
        rc               = rc,
        add_on_aggregate = add_on_total,
        alpha            = ALPHA,
        breakdown        = breakdown,
    )
