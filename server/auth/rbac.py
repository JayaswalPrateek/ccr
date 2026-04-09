"""Role-based access control FastAPI dependencies."""

from __future__ import annotations

import enum
from typing import Callable, Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.auth.security import decode_token
from server.core.database import get_db
from server.models.db_models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


class Role(str, enum.Enum):
    ADMIN        = "ADMIN"
    RISK_MANAGER = "RISK_MANAGER"
    AUDITOR      = "AUDITOR"


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Decode Bearer token → DB lookup → return active User ORM object."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        user_id: str = payload.get("sub", "")
        if not user_id:
            raise credentials_exc
    except ValueError:
        raise credentials_exc

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise credentials_exc
    return user


def require_role(*roles: Role) -> Callable:
    """Return a FastAPI dependency that raises 403 if the user's role is not in *roles*."""
    role_values = {r.value for r in roles}

    async def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in role_values:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(role_values)}",
            )
        return current_user

    return _check
