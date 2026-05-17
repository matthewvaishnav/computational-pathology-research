#!/usr/bin/env python3
"""
OAuth 2.0 and OpenID Connect (OIDC) Integration

Provides OAuth 2.0 authorization code flow and OIDC authentication
for integration with enterprise identity providers (Azure AD, Okta, Google, etc.)
"""

import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional
from urllib.parse import urlencode

import requests
from fastapi import HTTPException, Request
from jose import JWTError, jwt

logger = logging.getLogger(__name__)


# ============================================================================
# OAuth 2.0 Configuration
# ============================================================================


class OAuthConfig:
    """OAuth 2.0 / OIDC configuration."""

    def __init__(
        self,
        provider: str = "generic",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        authorization_endpoint: Optional[str] = None,
        token_endpoint: Optional[str] = None,
        userinfo_endpoint: Optional[str] = None,
        jwks_uri: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        scopes: Optional[list[str]] = None,
    ):
        """Initialize OAuth configuration.

        Args:
            provider: Provider name (azure, okta, google, generic)
            client_id: OAuth client ID
            client_secret: OAuth client secret
            authorization_endpoint: OAuth authorization URL
            token_endpoint: OAuth token URL
            userinfo_endpoint: OIDC userinfo URL
            jwks_uri: OIDC JWKS URL for token validation
            redirect_uri: OAuth redirect URI
            scopes: OAuth scopes to request
        """
        self.provider = provider
        self.client_id = client_id or os.getenv("OAUTH_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("OAUTH_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv("OAUTH_REDIRECT_URI")

        # Provider-specific defaults
        if provider == "azure":
            tenant_id = os.getenv("AZURE_TENANT_ID", "common")
            base_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0"
            self.authorization_endpoint = authorization_endpoint or f"{base_url}/authorize"
            self.token_endpoint = token_endpoint or f"{base_url}/token"
            self.userinfo_endpoint = (
                userinfo_endpoint or "https://graph.microsoft.com/oidc/userinfo"
            )
            self.jwks_uri = jwks_uri or f"{base_url}/discovery/keys"
            self.scopes = scopes or ["openid", "profile", "email"]

        elif provider == "okta":
            okta_domain = os.getenv("OKTA_DOMAIN")
            if not okta_domain:
                raise ValueError("OKTA_DOMAIN environment variable required for Okta provider")
            base_url = f"https://{okta_domain}/oauth2/default"
            self.authorization_endpoint = authorization_endpoint or f"{base_url}/v1/authorize"
            self.token_endpoint = token_endpoint or f"{base_url}/v1/token"
            self.userinfo_endpoint = userinfo_endpoint or f"{base_url}/v1/userinfo"
            self.jwks_uri = jwks_uri or f"{base_url}/v1/keys"
            self.scopes = scopes or ["openid", "profile", "email"]

        elif provider == "google":
            self.authorization_endpoint = (
                authorization_endpoint or "https://accounts.google.com/o/oauth2/v2/auth"
            )
            self.token_endpoint = token_endpoint or "https://oauth2.googleapis.com/token"
            self.userinfo_endpoint = (
                userinfo_endpoint or "https://openidconnect.googleapis.com/v1/userinfo"
            )
            self.jwks_uri = jwks_uri or "https://www.googleapis.com/oauth2/v3/certs"
            self.scopes = scopes or ["openid", "profile", "email"]

        else:
            # Generic provider - all endpoints must be provided
            self.authorization_endpoint = authorization_endpoint
            self.token_endpoint = token_endpoint
            self.userinfo_endpoint = userinfo_endpoint
            self.jwks_uri = jwks_uri
            self.scopes = scopes or ["openid", "profile", "email"]

        # Validate required fields
        if not self.client_id:
            raise ValueError("OAuth client_id is required")
        if not self.client_secret:
            raise ValueError("OAuth client_secret is required")
        if not self.redirect_uri:
            raise ValueError("OAuth redirect_uri is required")
        if not self.authorization_endpoint:
            raise ValueError("OAuth authorization_endpoint is required")
        if not self.token_endpoint:
            raise ValueError("OAuth token_endpoint is required")

        logger.info(
            f"OAuth config initialized: provider={provider} client_id={self.client_id[:8]}..."
        )


# ============================================================================
# OAuth 2.0 Client
# ============================================================================


class OAuthClient:
    """OAuth 2.0 / OIDC client implementation."""

    def __init__(self, config: OAuthConfig):
        """Initialize OAuth client."""
        self.config = config
        self.state_store: Dict[str, datetime] = (
            {}
        )  # In-memory state storage (use Redis in production)
        self.jwks_cache: Optional[Dict] = None
        self.jwks_cache_expiry: Optional[datetime] = None

        logger.info(f"OAuth client initialized: provider={config.provider}")

    def get_authorization_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """Generate OAuth authorization URL.

        Returns:
            Tuple of (authorization_url, state)
        """
        # Generate state for CSRF protection
        if not state:
            state = secrets.token_urlsafe(32)

        # Store state with expiry (5 minutes)
        self.state_store[state] = datetime.utcnow() + timedelta(minutes=5)

        # Build authorization URL
        params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "state": state,
        }

        auth_url = f"{self.config.authorization_endpoint}?{urlencode(params)}"

        logger.info(f"Generated authorization URL: state={state[:8]}...")
        return auth_url, state

    def validate_state(self, state: str) -> bool:
        """Validate OAuth state parameter (CSRF protection).

        Args:
            state: State parameter from callback

        Returns:
            True if valid, False otherwise
        """
        if state not in self.state_store:
            logger.warning(f"Invalid state: {state[:8]}... not found")
            return False

        expiry = self.state_store[state]
        if datetime.utcnow() > expiry:
            logger.warning(f"Expired state: {state[:8]}...")
            del self.state_store[state]
            return False

        # Remove used state
        del self.state_store[state]
        return True

    def exchange_code_for_token(self, code: str) -> Dict:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from callback

        Returns:
            Token response dict with access_token, id_token, etc.
        """
        try:
            # Token request
            data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.config.redirect_uri,
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
            }

            response = requests.post(
                self.config.token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
            response.raise_for_status()

            token_data = response.json()

            logger.info("Successfully exchanged code for token")
            return token_data

        except requests.RequestException as e:
            logger.error(f"Token exchange failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to exchange authorization code")

    def get_userinfo(self, access_token: str) -> Dict:
        """Get user information from OIDC userinfo endpoint.

        Args:
            access_token: OAuth access token

        Returns:
            User info dict
        """
        if not self.config.userinfo_endpoint:
            raise ValueError("Userinfo endpoint not configured")

        try:
            response = requests.get(
                self.config.userinfo_endpoint,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
            response.raise_for_status()

            userinfo = response.json()

            logger.info(f"Retrieved userinfo: sub={userinfo.get('sub', 'unknown')}")
            return userinfo

        except requests.RequestException as e:
            logger.error(f"Userinfo request failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve user information")

    def validate_id_token(self, id_token: str) -> Dict:
        """Validate OIDC ID token with proper JWKS-based signature verification.

        Args:
            id_token: JWT ID token from token response

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token validation fails
        """
        if not self.config.jwks_uri:
            logger.error("JWKS URI not configured - cannot validate ID token")
            raise HTTPException(
                status_code=500, detail="ID token validation not configured - JWKS URI required"
            )

        try:
            # Fetch JWKS (with caching)
            jwks = self._get_jwks()

            # Decode header to get key ID (kid)
            unverified_header = jwt.get_unverified_header(id_token)
            kid = unverified_header.get("kid")

            if not kid:
                raise ValueError("No 'kid' in token header")

            # Find matching key in JWKS
            signing_key = None
            for key in jwks.get("keys", []):
                if key.get("kid") == kid:
                    signing_key = key
                    break

            if not signing_key:
                raise ValueError(f"No matching key found for kid: {kid}")

            # Construct RSA public key from JWK
            from jose.backends import RSAKey

            rsa_key = RSAKey(signing_key, algorithm="RS256")

            # Validate token with signature verification
            # Expected issuer varies by provider
            expected_issuer = self._get_expected_issuer()

            payload = jwt.decode(
                id_token,
                rsa_key.to_pem().decode("utf-8"),
                algorithms=["RS256"],
                audience=self.config.client_id,
                issuer=expected_issuer,
                options={
                    "verify_signature": True,
                    "verify_aud": True,
                    "verify_iat": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iss": True,
                    "verify_sub": True,
                    "verify_jti": False,
                    "verify_at_hash": False,
                    "leeway": 0,
                },
            )

            # Additional validation
            if not payload.get("sub"):
                raise ValueError("Missing 'sub' claim")

            # Check token was issued recently (within last hour)
            iat = payload.get("iat")
            if iat:
                issued_at = datetime.utcfromtimestamp(iat)
                if datetime.utcnow() - issued_at > timedelta(hours=1):
                    raise ValueError("Token issued too long ago")

            logger.info(f"ID token validated successfully: sub={payload.get('sub')}")
            return payload

        except JWTError as e:
            logger.error(f"JWT validation failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid ID token")
        except ValueError as e:
            logger.error(f"Token validation failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid ID token")
        except Exception as e:
            logger.error(f"Unexpected error during token validation: {e}")
            raise HTTPException(status_code=500, detail="Token validation error")

    def _get_expected_issuer(self) -> str:
        """Get expected issuer based on provider.

        Returns:
            Expected issuer URL
        """
        if self.config.provider == "azure":
            tenant_id = os.getenv("AZURE_TENANT_ID", "common")
            return f"https://login.microsoftonline.com/{tenant_id}/v2.0"
        elif self.config.provider == "okta":
            okta_domain = os.getenv("OKTA_DOMAIN")
            return f"https://{okta_domain}/oauth2/default"
        elif self.config.provider == "google":
            return "https://accounts.google.com"
        else:
            # For generic provider, derive from authorization endpoint
            return self.config.authorization_endpoint.rsplit("/", 2)[0]

    def _get_jwks(self) -> Dict:
        """Fetch JWKS from provider (with caching).

        Returns:
            JWKS dict
        """
        # Check cache
        if self.jwks_cache and self.jwks_cache_expiry:
            if datetime.utcnow() < self.jwks_cache_expiry:
                return self.jwks_cache

        # Fetch JWKS
        try:
            response = requests.get(self.config.jwks_uri, timeout=10)
            response.raise_for_status()

            jwks = response.json()

            # Cache for 1 hour
            self.jwks_cache = jwks
            self.jwks_cache_expiry = datetime.utcnow() + timedelta(hours=1)

            logger.info("Fetched and cached JWKS")
            return jwks

        except requests.RequestException as e:
            logger.error(f"JWKS fetch failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch JWKS")

    def refresh_token(self, refresh_token: str) -> Dict:
        """Refresh access token using refresh token.

        Args:
            refresh_token: OAuth refresh token

        Returns:
            New token response dict
        """
        try:
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
            }

            response = requests.post(
                self.config.token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
            response.raise_for_status()

            token_data = response.json()

            logger.info("Successfully refreshed token")
            return token_data

        except requests.RequestException as e:
            logger.error(f"Token refresh failed: {e}")
            raise HTTPException(status_code=401, detail="Failed to refresh token")


# ============================================================================
# FastAPI Integration Helpers
# ============================================================================


def create_oauth_client(provider: str = "generic") -> OAuthClient:
    """Create OAuth client from environment variables.

    Args:
        provider: OAuth provider (azure, okta, google, generic)

    Returns:
        Configured OAuthClient
    """
    config = OAuthConfig(provider=provider)
    return OAuthClient(config)


async def oauth_callback_handler(
    request: Request,
    oauth_client: OAuthClient,
) -> Dict:
    """Handle OAuth callback.

    Args:
        request: FastAPI request
        oauth_client: OAuth client instance

    Returns:
        User info dict
    """
    # Get code and state from query params
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")

    # Check for errors
    if error:
        error_description = request.query_params.get("error_description", "Unknown error")
        logger.error(f"OAuth error: {error} - {error_description}")
        raise HTTPException(status_code=400, detail=f"OAuth error: {error_description}")

    # Validate required params
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state parameter")

    # Validate state (CSRF protection)
    if not oauth_client.validate_state(state):
        raise HTTPException(status_code=400, detail="Invalid or expired state parameter")

    # Exchange code for token
    token_data = oauth_client.exchange_code_for_token(code)

    # Get user info
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=500, detail="No access token in response")

    userinfo = oauth_client.get_userinfo(access_token)

    # Validate ID token if present
    id_token = token_data.get("id_token")
    if id_token:
        id_token_claims = oauth_client.validate_id_token(id_token)
        # Merge claims into userinfo
        userinfo.update(id_token_claims)

    return {
        "userinfo": userinfo,
        "access_token": access_token,
        "refresh_token": token_data.get("refresh_token"),
        "expires_in": token_data.get("expires_in"),
    }


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    # Example: Azure AD OAuth
    logging.basicConfig(level=logging.INFO)

    # Set environment variables:
    # export OAUTH_CLIENT_ID="your-client-id"
    # export OAUTH_CLIENT_SECRET="your-client-secret"
    # export OAUTH_REDIRECT_URI="http://localhost:8000/auth/callback"
    # export AZURE_TENANT_ID="your-tenant-id"

    try:
        client = create_oauth_client(provider="azure")

        # Step 1: Get authorization URL
        auth_url, state = client.get_authorization_url()
        logger.info(f"Authorization URL: {auth_url}")
        logger.info(f"State: {state}")

        # Step 2: User visits auth_url, grants permission, redirected to callback
        # Step 3: Extract code from callback URL
        # code = "..."

        # Step 4: Exchange code for token
        # token_data = client.exchange_code_for_token(code)
        # print(f"Access token: {token_data['access_token'][:20]}...")

        # Step 5: Get user info
        # userinfo = client.get_userinfo(token_data['access_token'])
        # print(f"User: {userinfo}")

    except Exception as e:
        logger.error(f"Error: {e}")
