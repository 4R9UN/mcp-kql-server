"""
KQL Authentication Module

This module handles Azure authentication for KQL cluster access.
Supports both Azure CLI (local dev) and Service Principal (container/k8s) auth modes.
The auth mode is auto-detected from environment variables.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import subprocess
import platform
import os
import logging
from functools import lru_cache
from typing import Dict, Any, Union
from tenacity import retry, stop_after_attempt, wait_exponential
from azure.identity import AzureCliCredential, ClientSecretCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_auth_mode() -> str:
    """
    Detect auth mode from environment variables.

    Returns 'service_principal' when KUSTO_CLIENT_ID, KUSTO_CLIENT_SECRET,
    and KUSTO_TENANT_ID are all set. Otherwise returns 'az_cli'.
    """
    if all(os.environ.get(k) for k in ("KUSTO_CLIENT_ID", "KUSTO_CLIENT_SECRET", "KUSTO_TENANT_ID")):
        return "service_principal"
    return "az_cli"


@lru_cache(maxsize=1)
def get_azure_credential() -> Union[AzureCliCredential, ClientSecretCredential]:
    """
    Get the Azure credential based on auth mode.

    In service_principal mode, returns ClientSecretCredential.
    In az_cli mode, returns AzureCliCredential.
    """
    if get_auth_mode() == "service_principal":
        logger.info("Using Service Principal authentication")
        return ClientSecretCredential(
            tenant_id=os.environ["KUSTO_TENANT_ID"],
            client_id=os.environ["KUSTO_CLIENT_ID"],
            client_secret=os.environ["KUSTO_CLIENT_SECRET"],
        )
    return AzureCliCredential()


def _authenticate_service_principal() -> Dict[str, Any]:
    """
    Validate Service Principal credentials by requesting a token.
    """
    logger.info("Validating Service Principal credentials...")
    try:
        credential = get_azure_credential()
        credential.get_token("https://kusto.kusto.windows.net/.default")
        logger.info("Service Principal authentication successful.")
        return {"authenticated": True, "message": "Service Principal authentication successful."}
    except Exception as e:
        logger.error("Service Principal authentication failed: %s", str(e))
        return {"authenticated": False, "message": str(e)}


@lru_cache(maxsize=1)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def kql_auth():
    """
    Check if user is authenticated with Azure CLI.

    Returns:
        dict: Authentication status and message
    """
    logger.info("Checking Azure CLI authentication...")
    az_command = "az.cmd" if platform.system() == "Windows" else "az"

    try:
        env = os.environ.copy()
        subprocess.run(
            [az_command, "config", "set", "core.login_experience_v2=off"],
            env=env, capture_output=True, text=True, check=True
        )
        subprocess.run(
            [az_command, "account", "get-access-token"],
            capture_output=True, text=True, check=True
        )
        logger.info("User is authenticated with Azure CLI.")
        return {"authenticated": True, "message": "User is authenticated."}
    except subprocess.CalledProcessError as e:
        logger.warning("User is not authenticated: %s", e.stderr)
        return {"authenticated": False, "message": "User is not authenticated."}
    except Exception as e:
        logger.error("Authentication check failed: %s", str(e))
        return {"authenticated": False, "message": str(e)}

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
def trigger_az_cli_auth():
    """
    Trigger Azure CLI authentication using device code.

    Returns:
        dict: Authentication status and message
    """
    logger.info("Triggering Azure CLI login...")
    az_command = "az.cmd" if platform.system() == "Windows" else "az"
    env = os.environ.copy()
    env["AZURE_CORE_ONLY_SHOW_ERRORS"] = "true"

    try:
        subprocess.run(
            [az_command, "config", "set", "core.login_experience_v2=off"],
            env=env, capture_output=True, text=True, check=True
        )
        auth_result = subprocess.run(
            [az_command, "login", "--use-device-code"],
            capture_output=True, text=True, env=env, timeout=120, check=False
        )
        if auth_result.returncode == 0:
            logger.info("Azure CLI login successful.")
            return {"authenticated": True, "message": "Azure CLI login successful."}
        logger.error("Azure CLI login failed: %s", auth_result.stderr)
        return {"authenticated": False, "message": auth_result.stderr}
    except subprocess.TimeoutExpired:
        logger.error("Authentication timed out.")
        return {"authenticated": False, "message": "Authentication timed out. Please try again."}
    except Exception as e:
        logger.error("Authentication attempt failed: %s", str(e))
        return {"authenticated": False, "message": str(e)}

def authenticate():
    """
    Complete authentication flow with caching and retry logic.

    In service_principal mode, validates credentials directly (no CLI needed).
    In az_cli mode, checks CLI auth and falls back to device code login.

    Returns:
        dict: Final authentication status and message
    """
    logger.info("Starting authentication process...")

    if get_auth_mode() == "service_principal":
        return _authenticate_service_principal()

    auth_status = kql_auth()
    if auth_status["authenticated"]:
        logger.info("Already authenticated to Azure.")
        return auth_status

    logger.info("Not authenticated. Initiating Azure login...")
    auth_status = trigger_az_cli_auth()

    if not auth_status["authenticated"]:
        logger.error("Authentication failed: %s", auth_status["message"])
        logger.info("Troubleshooting tips:")
        logger.info("1. Ensure you have a working internet connection")
        logger.info("2. Verify your Azure account is active")
        logger.info("3. Try running 'az login' directly in your terminal")

    return auth_status

def authenticate_kusto() -> Dict[str, Any]:
    """
    Wrapper function for compatibility with existing code.

    Returns:
        dict: Authentication status and message
    """
    return authenticate()

if __name__ == "__main__":
    result = authenticate()
    print(f"Authentication status: {result['authenticated']}")
    print(f"Message: {result['message']}")
