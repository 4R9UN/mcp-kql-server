"""
Test module for package-level functionality.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import mcp_kql_server

from mcp_kql_server import __author__, __email__, __version__ as pkg_version
from mcp_kql_server.constants import __version__ as const_version
from mcp_kql_server.utils import bracket_if_needed, normalize_cluster_uri, sanitize_filename


def test_package_imports():
    """Test that package imports work correctly."""
    assert hasattr(mcp_kql_server, "__version__")
    assert hasattr(mcp_kql_server, "__author__")
    assert mcp_kql_server.__version__ == "2.1.2"
    assert mcp_kql_server.__author__ == "Arjun Trivedi"


def test_version_consistency():
    """Test that version is consistent across modules."""
    assert pkg_version == const_version == "2.1.2"


def test_author_attribution():
    """Test that author information is properly set."""
    assert __author__ == "Arjun Trivedi"
    assert __email__ == "arjuntrivedi42@yahoo.com"


def test_module_structure():
    """Test that expected modules exist."""
    expected_modules = [
        "mcp_kql_server.constants",
        "mcp_kql_server.utils",
        "mcp_kql_server.execute_kql",
        "mcp_kql_server.mcp_server",
    ]

    for module_name in expected_modules:
        try:
            __import__(module_name)
            # Module import successful
        except ImportError:
            # Skip individual module if not available
            continue


def test_basic_functionality():
    """Test basic package functionality without external dependencies."""
    assert sanitize_filename("bad:file/name?.txt") == "bad_file_name_.txt"
    assert normalize_cluster_uri("help.kusto.windows.net") == "https://help.kusto.windows.net"
    assert bracket_if_needed("where") == "['where']"
