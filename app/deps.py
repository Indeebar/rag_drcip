"""Shared FastAPI dependencies.

Keeping them in one place means route modules stay free of configuration plumbing
and tests can override a single symbol.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]
