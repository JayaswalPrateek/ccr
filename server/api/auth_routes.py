"""Authentication and user management endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from server.auth.rbac import Role, get_current_user, require_role
from server.auth.security import create_access_token, hash_password, verify_password
from server.core.database import get_db
from server.models.db_models import User

auth_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"


class UserOut(BaseModel):
    id:         str
    username:   str
    email:      str
    role:       str
    is_active:  bool
    created_at: datetime
    last_login: Optional[datetime] = None

    model_config = {"from_attributes": True}


class RegisterRequest(BaseModel):
    username: str
    email:    str
    password: str
    role:     str = Role.AUDITOR


class UpdateUserRequest(BaseModel):
    role:      Optional[str]  = None
    is_active: Optional[bool] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@auth_router.post("/login", response_model=TokenResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db:   AsyncSession              = Depends(get_db),
) -> TokenResponse:
    result = await db.execute(select(User).where(User.username == form.username))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(form.password, user.hashed_pw):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    user.last_login = datetime.now(timezone.utc)
    await db.commit()

    token = create_access_token({"sub": user.id, "role": user.role})
    return TokenResponse(access_token=token)


@auth_router.post("/register", response_model=UserOut, status_code=201)
async def register(
    body:    RegisterRequest,
    db:      AsyncSession = Depends(get_db),
    _caller: User         = Depends(require_role(Role.ADMIN)),
) -> UserOut:
    dup = await db.execute(
        select(User).where(
            (User.username == body.username) | (User.email == body.email)
        )
    )
    existing = dup.scalar_one_or_none()
    if existing:
        field = "Username" if existing.username == body.username else "Email"
        raise HTTPException(status_code=409, detail=f"{field} already exists")

    new_user = User(
        username  = body.username,
        email     = body.email,
        hashed_pw = hash_password(body.password),
        role      = body.role,
    )
    db.add(new_user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Username or email already exists")
    await db.refresh(new_user)
    return UserOut.model_validate(new_user)


@auth_router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(current_user)


@auth_router.get("/users", response_model=List[UserOut])
async def list_users(
    db:      AsyncSession = Depends(get_db),
    _caller: User         = Depends(require_role(Role.ADMIN)),
) -> List[UserOut]:
    result = await db.execute(select(User).order_by(User.created_at))
    return [UserOut.model_validate(u) for u in result.scalars().all()]


@auth_router.put("/users/{user_id}", response_model=UserOut)
async def update_user(
    user_id: str,
    body:    UpdateUserRequest,
    db:      AsyncSession = Depends(get_db),
    _caller: User         = Depends(require_role(Role.ADMIN)),
) -> UserOut:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if body.role is not None:
        user.role = body.role
    if body.is_active is not None:
        user.is_active = body.is_active

    await db.commit()
    await db.refresh(user)
    return UserOut.model_validate(user)
