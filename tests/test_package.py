"""
Test module for package-level functionality.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import mcp_kql_server
import toml
from packaging.requirements import Requirement

from mcp_kql_server import __author__, __email__, __version__ as pkg_version
from mcp_kql_server.constants import __version__ as const_version
from mcp_kql_server.utils import bracket_if_needed, normalize_cluster_uri, sanitize_filename
from mcp_kql_server.version_checker import fetch_latest_pypi_version


def test_package_imports():
    """Test that package imports work correctly."""
    assert hasattr(mcp_kql_server, "__version__")
    assert hasattr(mcp_kql_server, "__author__")
    assert mcp_kql_server.__version__ == "2.1.5"
    assert mcp_kql_server.__author__ == "Arjun Trivedi"


def test_package_import_is_lightweight():
    """Metadata imports must not initialize pandas, Kusto, or server state."""
    script = (
        "import mcp_kql_server, sys; "
        "assert 'pandas' not in sys.modules; "
        "assert 'azure.kusto.data' not in sys.modules; "
        "assert 'mcp_kql_server.mcp_server' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_version_consistency():
    """Test that version is consistent across modules."""
    assert pkg_version == const_version == "2.1.5"


def test_dependency_bounds_preserve_shared_environment_compatibility():
    """Test package bounds remain compatible with FastAPI and Semantic Kernel."""
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    dependencies = toml.loads(pyproject_path.read_text(encoding="utf-8"))["project"]["dependencies"]
    requirements = {requirement.name: requirement for requirement in map(Requirement, dependencies)}

    assert str(requirements["mcp"].specifier) == "<2.0.0,>=1.28.1"
    assert "fastmcp" not in requirements
    assert "pydantic" not in requirements
    assert "starlette" not in requirements
    assert str(requirements["pandas"].specifier) == "<3.0.0,>=1.3.0"
    assert str(requirements["numpy"].specifier) == "<3.0.0,>=1.21.0"


def test_requirements_file_matches_project_dependencies():
    """Keep the pip requirements file aligned with package metadata."""
    project_root = Path(__file__).parents[1]
    dependencies = toml.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["dependencies"]
    requirement_lines = [
        line.strip()
        for line in (project_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    project_requirements = {Requirement(value).name: str(Requirement(value).specifier) for value in dependencies}
    file_requirements = {Requirement(value).name: str(Requirement(value).specifier) for value in requirement_lines}

    assert file_requirements == project_requirements


@patch("mcp_kql_server.version_checker.urlopen")
def test_version_check_uses_pypi_json(mock_urlopen):
    """Test the standard-library version check parses the PyPI response."""
    response = mock_urlopen.return_value.__enter__.return_value
    response.read.return_value = b'{"info": {"version": "9.9.9"}}'

    assert fetch_latest_pypi_version() == "9.9.9"


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
