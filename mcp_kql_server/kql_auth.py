"""
KQL Authentication Module

This module handles Azure CLI authentication for KQL cluster access.
Provides cached authentication checking and device code flow authentication.
Supports AzureCliCredential for async clients.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""
# pylint: disable=broad-exception-caught

import subprocess
import platform
import os
import logging
from functools import lru_cache
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from azure.identity import AzureCliCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUTH_CHECK_TIMEOUT_SECONDS = int(os.environ.get("MCP_KQL_AUTH_CHECK_TIMEOUT_SECONDS", "10"))
AUTH_LOGIN_TIMEOUT_SECONDS = int(os.environ.get("MCP_KQL_AUTH_LOGIN_TIMEOUT_SECONDS", "120"))

@lru_cache(maxsize=1)
def get_azure_credential() -> AzureCliCredential:
    """
    Get the Azure CLI credential for async clients.
    """
    return AzureCliCredential()

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
            env=env, capture_output=True, text=True, check=True, timeout=AUTH_CHECK_TIMEOUT_SECONDS
        )
        subprocess.run(
            [az_command, "account", "get-access-token"],
            env=env, capture_output=True, text=True, check=True, timeout=AUTH_CHECK_TIMEOUT_SECONDS
        )
        logger.info("User is authenticated with Azure CLI.")
        return {"authenticated": True, "message": "User is authenticated.", "auth_method": "azure-cli"}
    except subprocess.TimeoutExpired:
        logger.error("Authentication check timed out.")
        return {"authenticated": False, "message": "Authentication check timed out."}
    except subprocess.CalledProcessError as e:
        logger.warning("User is not authenticated: %s", e.stderr)
        return {"authenticated": False, "message": "User is not authenticated."}
    except (OSError, ValueError) as e:
        logger.error("Authentication check failed: %s", str(e))
        return {"authenticated": False, "message": str(e)}
    except Exception as e:  # Required for test compatibility (test_kql_auth_exception)
        logger.error("Unexpected error during authentication: %s", str(e))
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
            env=env, capture_output=True, text=True, check=True, timeout=AUTH_CHECK_TIMEOUT_SECONDS
        )
        auth_result = subprocess.run(
            [az_command, "login", "--use-device-code"],
            capture_output=True, text=True, env=env, timeout=AUTH_LOGIN_TIMEOUT_SECONDS, check=False
        )
        if auth_result.returncode == 0:
            logger.info("Azure CLI login successful.")
            return {"authenticated": True, "message": "Azure CLI login successful.", "auth_method": "azure-cli"}
        logger.error("Azure CLI login failed: %s", auth_result.stderr)
        return {"authenticated": False, "message": auth_result.stderr}
    except subprocess.TimeoutExpired:
        logger.error("Authentication timed out.")
        return {"authenticated": False, "message": "Authentication timed out. Please try again."}
    except (OSError, ValueError) as e:
        logger.error("Authentication attempt failed: %s", str(e))
        return {"authenticated": False, "message": str(e)}

def authenticate(interactive: bool = True):
    """
    Complete authentication flow with caching and retry logic.

    Args:
        interactive: If False, never starts device-code login. This keeps MCP
            startup and tool discovery fast in hosts that cannot answer prompts.

    Returns:
        dict: Final authentication status and message
    """
    logger.info("Starting authentication process...")

    auth_status = kql_auth()
    if auth_status["authenticated"]:
        logger.info("Already authenticated to Azure.")
        return auth_status

    logger.info("Not authenticated. Initiating Azure login...")
    if not interactive:
        return {
            "authenticated": False,
            "message": "Azure CLI is not authenticated. Run 'az login' in a terminal, then retry.",
            "auth_method": "azure-cli",
            "interactive_required": True,
        }

    auth_status = trigger_az_cli_auth()

    if not auth_status["authenticated"]:
        logger.error("Authentication failed: %s", auth_status["message"])
        logger.info("Troubleshooting tips:")
        logger.info("1. Ensure you have a working internet connection")
        logger.info("2. Verify your Azure account is active")
        logger.info("3. Try running 'az login' directly in your terminal")

    return auth_status

def authenticate_kusto(interactive: bool = True) -> Dict[str, Any]:
    """
    Wrapper function for compatibility with existing code.

    Args:
        interactive: If False, skip device-code login and return a structured
            unauthenticated status instead.

    Returns:
        dict: Authentication status and message
    """
    return authenticate(interactive=interactive)

if __name__ == "__main__":
    result = authenticate()
    print(f"Authentication status: {result['authenticated']}")
    print(f"Message: {result['message']}")
