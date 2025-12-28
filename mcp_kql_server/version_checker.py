"""
Version Checker for MCP KQL Server.

This module provides functionality to check for updates on PyPI
and optionally auto-install or prompt the user to update.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import logging
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple
from packaging import version as pkg_version

logger = logging.getLogger(__name__)

# Package name on PyPI
PYPI_PACKAGE_NAME = "mcp-kql-server"


def get_current_version() -> str:
    """Get the current installed version."""
    from .constants import __version__
    return __version__


def fetch_latest_pypi_version(timeout: int = 5) -> Optional[str]:
    """
    Fetch the latest version from PyPI.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        Latest version string or None if fetch fails.
    """
    try:
        import httpx

        url = f"https://pypi.org/pypi/{PYPI_PACKAGE_NAME}/json"
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get("info", {}).get("version")
    except ImportError:
        # Fallback to requests if httpx not available
        try:
            import requests
            url = f"https://pypi.org/pypi/{PYPI_PACKAGE_NAME}/json"
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get("info", {}).get("version")
        except Exception as e:
            logger.debug("Failed to fetch PyPI version with requests: %s", e)
    except Exception as e:
        logger.debug("Failed to fetch PyPI version: %s", e)

    return None


def compare_versions(current: str, latest: str) -> int:
    """
    Compare two version strings.

    Returns:
        -1 if current < latest (update available)
         0 if current == latest (up to date)
         1 if current > latest (development version)
    """
    try:
        current_v = pkg_version.parse(current)
        latest_v = pkg_version.parse(latest)

        if current_v < latest_v:
            return -1
        elif current_v > latest_v:
            return 1
        else:
            return 0
    except Exception:
        # Fallback to string comparison
        if current < latest:
            return -1
        elif current > latest:
            return 1
        return 0


def check_for_updates(auto_update: bool = False) -> Dict[str, Any]:
    """
    Check for available updates.

    Args:
        auto_update: If True, automatically install updates.

    Returns:
        Dict with update status information.
    """
    current = get_current_version()
    latest = fetch_latest_pypi_version()

    result = {
        "current_version": current,
        "latest_version": latest,
        "update_available": False,
        "update_installed": False,
        "message": "",
        "error": None
    }

    if latest is None:
        result["message"] = "Could not check for updates (network issue or PyPI unavailable)"
        return result

    comparison = compare_versions(current, latest)

    if comparison == 0:
        result["message"] = f"MCP KQL Server v{current} is up to date"
    elif comparison == 1:
        result["message"] = f"MCP KQL Server v{current} (development version, latest stable: {latest})"
    else:  # comparison == -1
        result["update_available"] = True
        result["message"] = f"Update available: v{current} -> v{latest}"

        if auto_update:
            success = install_update()
            result["update_installed"] = success
            if success:
                result["message"] = f"Successfully updated to v{latest}. Please restart the server."
            else:
                result["message"] = f"Update available: v{current} -> v{latest}. Auto-update failed. Run: pip install --upgrade {PYPI_PACKAGE_NAME}"

    return result


def install_update() -> bool:
    """
    Install the latest version from PyPI.

    Returns:
        True if update was successful, False otherwise.
    """
    try:
        logger.info("Installing update from PyPI...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", PYPI_PACKAGE_NAME, "-q"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False  # We handle the return code ourselves
        )

        if result.returncode == 0:
            logger.info("Update installed successfully")
            return True
        else:
            logger.warning("Update failed: %s", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error("Update timed out")
        return False
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        logger.error("Failed to install update: %s", e)
        return False


def get_update_notification() -> Optional[str]:
    """
    Get a user-friendly update notification message.

    Returns:
        Notification string if update is available, None otherwise.
    """
    try:
        current = get_current_version()
        latest = fetch_latest_pypi_version(timeout=3)  # Short timeout for startup

        if latest and compare_versions(current, latest) == -1:
            return (
                f"\n"
                f"+---------------------------------------------------------------+\n"
                f"|  UPDATE AVAILABLE: v{current} -> v{latest:<10}                   |\n"
                f"|  Run: pip install --upgrade {PYPI_PACKAGE_NAME:<20}       |\n"
                f"+---------------------------------------------------------------+\n"
            )
    except Exception:
        pass

    return None


def startup_version_check(auto_update: bool = False, silent: bool = False) -> Tuple[bool, str]:
    """
    Perform version check at startup.

    Args:
        auto_update: If True, automatically install updates.
        silent: If True, don't log update notifications.

    Returns:
        Tuple of (update_available, message)
    """
    try:
        result = check_for_updates(auto_update=auto_update)

        if result["update_available"] and not silent:
            if result["update_installed"]:
                logger.warning("MCP KQL Server updated to v%s. Please restart.", result["latest_version"])
            else:
                logger.info(
                    "Update available: v%s -> v%s. Run: pip install --upgrade %s",
                    result["current_version"],
                    result["latest_version"],
                    PYPI_PACKAGE_NAME
                )

        return result["update_available"], result["message"]
    except Exception as e:
        logger.debug("Version check failed: %s", e)
        return False, f"Version check failed: {e}"
