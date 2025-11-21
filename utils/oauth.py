"""
OAuth utilities for Databricks Apps

Handles OAuth token generation using client credentials flow
with the OAuth credentials automatically injected by Databricks Apps.
"""

import os
import requests
import logging
from typing import Optional
import time

logger = logging.getLogger(__name__)

# Cache for token to avoid repeated requests
_token_cache = {
    "token": None,
    "expires_at": 0
}

# Flag to detect if we're running locally vs in Databricks Apps
_is_local_mode = None


def get_databricks_token() -> str:
    """
    Get Databricks OAuth token using client credentials flow (in Apps)
    or profile-based authentication (local testing).

    When running in Databricks Apps:
    - Uses environment variables automatically injected: DATABRICKS_HOST,
      DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET
    - Uses OAuth client credentials flow

    When running locally:
    - Falls back to using Databricks SDK with profile authentication
    - Reads credentials from ~/.databrickscfg

    Returns:
        OAuth access token

    Raises:
        Exception if token cannot be obtained
    """
    global _is_local_mode

    # Check if we have a cached valid token
    current_time = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] > current_time:
        logger.debug("Using cached OAuth token")
        return _token_cache["token"]

    # Get OAuth credentials from environment
    host = os.environ.get("DATABRICKS_HOST")
    client_id = os.environ.get("DATABRICKS_CLIENT_ID")
    client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")

    # Check if we're running locally (no OAuth credentials)
    if not all([host, client_id, client_secret]):
        if _is_local_mode is None:
            _is_local_mode = True
            logger.info("OAuth credentials not available - using local profile authentication")
        return _get_token_from_sdk_profile()

    # Ensure host doesn't have protocol
    if host.startswith("http://") or host.startswith("https://"):
        host = host.split("://")[1]

    # Construct token endpoint
    token_url = f"https://{host}/oidc/v1/token"

    logger.info(f"Requesting OAuth token from {token_url}")

    # Request token using client credentials flow
    try:
        response = requests.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "scope": "all-apis"
            },
            auth=(client_id, client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        response.raise_for_status()
        token_data = response.json()

        access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)  # Default to 1 hour

        if not access_token:
            raise Exception("No access_token in OAuth response")

        # Cache the token (with 5 minute buffer before expiration)
        _token_cache["token"] = access_token
        _token_cache["expires_at"] = current_time + expires_in - 300

        logger.info(f"Successfully obtained OAuth token (expires in {expires_in}s)")
        return access_token

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to obtain OAuth token: {str(e)}", exc_info=True)
        raise Exception(f"OAuth token request failed: {str(e)}")


def _get_token_from_sdk_profile() -> str:
    """
    Get Databricks token using Databricks CLI with profile authentication (for local testing).

    Uses `databricks auth token` command which handles profile-based authentication
    from ~/.databrickscfg.

    Returns:
        Access token from profile authentication

    Raises:
        Exception if token cannot be obtained
    """
    try:
        import subprocess
        import json as json_module
        from databricks.sdk import WorkspaceClient

        # Get the profile name and host
        profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")

        # Use SDK to get the host from the profile
        w = WorkspaceClient()
        host = w.config.host

        logger.info(f"Getting token from Databricks CLI for profile: {profile}, host: {host}")

        # Use databricks CLI to get token
        cmd = ["databricks", "auth", "token", "--host", host, "--profile", profile]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise Exception(f"databricks auth token failed: {result.stderr}")

        # Parse JSON response
        token_data = json_module.loads(result.stdout)
        access_token = token_data.get("access_token")

        if not access_token:
            raise Exception("No access_token in CLI response")

        logger.info("Successfully obtained token from Databricks CLI")

        # Cache the token (set expiry to 1 hour as we don't know exact expiry)
        current_time = time.time()
        _token_cache["token"] = access_token
        _token_cache["expires_at"] = current_time + 3600 - 300  # 1 hour minus 5 min buffer

        return access_token

    except Exception as e:
        logger.error(f"Failed to obtain token from Databricks CLI: {str(e)}", exc_info=True)
        raise Exception(f"Databricks CLI authentication failed: {str(e)}")


def get_user_token_from_streamlit_context() -> Optional[str]:
    """
    Get on-behalf-of-user token from Streamlit context headers.

    NOTE: This function is kept for future use but is NOT currently used.
    All operations currently use the App Service Principal token via get_databricks_token().

    For future use cases where on-behalf-of-user operations are needed
    (e.g., user-specific permissions, audit trails showing actual user actions).

    Returns:
        User access token or None if not available
    """
    try:
        import streamlit as st
        user_token = st.context.headers.get('X-Forwarded-Access-Token')
        if user_token:
            logger.debug("Retrieved user token from Streamlit context")
            return user_token
        else:
            logger.warning("No user token available in Streamlit context")
            return None
    except Exception as e:
        logger.error(f"Failed to get user token from Streamlit context: {e}")
        return None
