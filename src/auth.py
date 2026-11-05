import hmac
import os
from typing import Optional


MIN_TOKEN_LENGTH = 32
VIEWER = "viewer"
OPERATOR = "operator"
PUBLIC_HTTP_PATHS = {"/", "/api/health", "/api/system/overview"}


class AuthenticationConfigurationError(ValueError):
    pass


def configured_tokens() -> dict[str, str]:
    viewer = os.environ.get("TRAFFIC_VIEWER_TOKEN", "").strip()
    operator = os.environ.get("TRAFFIC_OPERATOR_TOKEN", "").strip()
    if not viewer or not operator:
        raise AuthenticationConfigurationError(
            "Configure both TRAFFIC_VIEWER_TOKEN and TRAFFIC_OPERATOR_TOKEN."
        )
    if len(viewer) < MIN_TOKEN_LENGTH or len(operator) < MIN_TOKEN_LENGTH:
        raise AuthenticationConfigurationError(
            f"Access tokens must each contain at least {MIN_TOKEN_LENGTH} characters."
        )
    if hmac.compare_digest(viewer, operator):
        raise AuthenticationConfigurationError(
            "Viewer and operator access tokens must be different."
        )
    return {VIEWER: viewer, OPERATOR: operator}


def authenticate_token(token: Optional[str]) -> Optional[str]:
    if not isinstance(token, str) or not token:
        return None
    tokens = configured_tokens()
    for role, configured in tokens.items():
        if hmac.compare_digest(token, configured):
            return role
    return None


def required_role(method: str) -> str:
    if method.upper() in {"GET", "HEAD", "OPTIONS"}:
        return VIEWER
    return OPERATOR


def role_allows(role: str, required: str) -> bool:
    return role == OPERATOR or (role == VIEWER and required == VIEWER)
