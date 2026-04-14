"""Simulation preset CRUD endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from server.auth.rbac import Role, get_current_user, require_role
from server.core.database import get_db
from server.models.db_models import SimPreset, User
from server.notifications.audit import log_event

preset_router = APIRouter(prefix="/api/v1/presets", tags=["presets"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PresetIn(BaseModel):
    name:            str
    description:     Optional[str] = None
    counterparty_id: Optional[str] = None
    params_json:     Dict[str, Any]
    stress_json:     Optional[Dict[str, Any]] = None
    is_shared:       bool = False


class PresetOut(BaseModel):
    id:              str
    name:            str
    description:     Optional[str]
    owner_id:        Optional[str]
    counterparty_id: Optional[str]
    params_json:     Dict[str, Any]
    stress_json:     Optional[Dict[str, Any]]
    is_shared:       bool
    use_count:       int
    last_used_at:    Optional[datetime]
    created_at:      datetime
    updated_at:      datetime
    model_config = {"from_attributes": True}


# ── List ──────────────────────────────────────────────────────────────────────

@preset_router.get("", response_model=List[PresetOut])
async def list_presets(
    counterparty_id: Optional[str] = Query(None),
    include_shared:  bool          = Query(True),
    db:              AsyncSession  = Depends(get_db),
    user:            User          = Depends(get_current_user),
) -> List[PresetOut]:
    """List own presets plus shared presets, optionally filtered by counterparty."""
    if include_shared:
        visibility = or_(SimPreset.owner_id == user.id, SimPreset.is_shared.is_(True))
    else:
        visibility = SimPreset.owner_id == user.id

    stmt = select(SimPreset).where(visibility).order_by(desc(SimPreset.updated_at))
    if counterparty_id:
        stmt = stmt.where(
            or_(SimPreset.counterparty_id == counterparty_id, SimPreset.counterparty_id.is_(None))
        )
    result = await db.execute(stmt)
    return [PresetOut.model_validate(p) for p in result.scalars().all()]


@preset_router.get("/recent", response_model=List[PresetOut])
async def recent_presets(
    limit: int          = Query(5, ge=1, le=20),
    db:    AsyncSession = Depends(get_db),
    user:  User         = Depends(get_current_user),
) -> List[PresetOut]:
    """Most recently used presets for the current user."""
    stmt = (
        select(SimPreset)
        .where(
            and_(
                SimPreset.owner_id    == user.id,
                SimPreset.last_used_at.isnot(None),
            )
        )
        .order_by(desc(SimPreset.last_used_at))
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [PresetOut.model_validate(p) for p in result.scalars().all()]


# ── Get one ───────────────────────────────────────────────────────────────────

@preset_router.get("/{preset_id}", response_model=PresetOut)
async def get_preset(
    preset_id: str,
    db:        AsyncSession = Depends(get_db),
    user:      User         = Depends(get_current_user),
) -> PresetOut:
    preset = await _get_visible_or_404(db, preset_id, user.id)
    return PresetOut.model_validate(preset)


# ── Create ────────────────────────────────────────────────────────────────────

@preset_router.post("", response_model=PresetOut, status_code=201)
async def create_preset(
    body: PresetIn,
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> PresetOut:
    preset = SimPreset(
        name            = body.name,
        description     = body.description,
        owner_id        = user.id,
        counterparty_id = body.counterparty_id,
        params_json     = body.params_json,
        stress_json     = body.stress_json,
        is_shared       = body.is_shared,
    )
    db.add(preset)
    await log_event(db, action="create_preset", user_id=user.id,
                    resource_type="sim_preset", resource_id=None,
                    detail={"name": body.name, "is_shared": body.is_shared})
    await db.commit()
    await db.refresh(preset)
    return PresetOut.model_validate(preset)


# ── Update ────────────────────────────────────────────────────────────────────

@preset_router.put("/{preset_id}", response_model=PresetOut)
async def update_preset(
    preset_id: str,
    body:      PresetIn,
    db:        AsyncSession = Depends(get_db),
    user:      User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> PresetOut:
    preset = await _get_owned_or_404(db, preset_id, user.id)
    preset.name            = body.name
    preset.description     = body.description
    preset.counterparty_id = body.counterparty_id
    preset.params_json     = body.params_json
    preset.stress_json     = body.stress_json
    preset.is_shared       = body.is_shared
    preset.updated_at      = datetime.now(timezone.utc)
    await log_event(db, action="update_preset", user_id=user.id,
                    resource_type="sim_preset", resource_id=preset_id,
                    detail={"name": body.name})
    await db.commit()
    await db.refresh(preset)
    return PresetOut.model_validate(preset)


# ── Delete ────────────────────────────────────────────────────────────────────

@preset_router.delete("/{preset_id}", status_code=204)
async def delete_preset(
    preset_id: str,
    db:        AsyncSession = Depends(get_db),
    user:      User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> None:
    preset = await _get_owned_or_404(db, preset_id, user.id)
    await db.delete(preset)
    await log_event(db, action="delete_preset", user_id=user.id,
                    resource_type="sim_preset", resource_id=preset_id,
                    detail={"name": preset.name})
    await db.commit()


# ── Mark as used ──────────────────────────────────────────────────────────────

@preset_router.post("/{preset_id}/use", response_model=PresetOut)
async def use_preset(
    preset_id: str,
    db:        AsyncSession = Depends(get_db),
    user:      User         = Depends(get_current_user),
) -> PresetOut:
    """Increment use_count and set last_used_at. Call when loading into simulator."""
    preset = await _get_visible_or_404(db, preset_id, user.id)
    preset.use_count    = (preset.use_count or 0) + 1
    preset.last_used_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(preset)
    return PresetOut.model_validate(preset)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_visible_or_404(db: AsyncSession, preset_id: str, user_id: str) -> SimPreset:
    result = await db.execute(
        select(SimPreset).where(
            and_(
                SimPreset.id == preset_id,
                or_(SimPreset.owner_id == user_id, SimPreset.is_shared.is_(True)),
            )
        )
    )
    preset = result.scalar_one_or_none()
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    return preset


async def _get_owned_or_404(db: AsyncSession, preset_id: str, user_id: str) -> SimPreset:
    result = await db.execute(
        select(SimPreset).where(SimPreset.id == preset_id, SimPreset.owner_id == user_id)
    )
    preset = result.scalar_one_or_none()
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found or not owned by you")
    return preset
