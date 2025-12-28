# Contributing to MCP KQL Server

Thank you for your interest in contributing to the MCP KQL Server! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

---

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow:

- **Be Respectful**: Treat everyone with respect and consideration
- **Be Inclusive**: Welcome contributors of all backgrounds and experience levels
- **Be Constructive**: Provide helpful feedback and accept constructive criticism
- **Be Professional**: Focus on what is best for the community and project

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Azure subscription (for testing with real Kusto clusters)
- VS Code (recommended) with Python extension

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/mcp-kql-server.git
cd mcp-kql-server
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/4R9UN/mcp-kql-server.git
```

---

## Development Setup

### Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate
```

### Install Dependencies

```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio pylint
```

### Verify Installation

```bash
# Run tests to verify setup
python -m pytest tests/ -v

# Check code quality
python -m pylint mcp_kql_server/
```

---

## Making Changes

### Branch Naming Convention

Create a new branch for your changes:

```bash
git checkout -b <type>/<short-description>
```

**Branch Types:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications
- `perf/` - Performance improvements

**Examples:**
- `feature/query-templates`
- `fix/schema-cache-timeout`
- `docs/api-reference`

### Commit Message Format

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Test additions
- `chore`: Maintenance tasks

**Examples:**
```
feat(memory): add connection pooling for Kusto clients

- Implement KustoClientPool class with configurable pool size
- Add automatic connection recycling after 30 minutes
- Include health checks for pooled connections

Closes #42
```

---

## Pull Request Process

### Before Submitting

1. **Update from upstream:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all tests:**
   ```bash
   python -m pytest tests/ -v --cov=mcp_kql_server
   ```

3. **Check code quality:**
   ```bash
   python -m pylint mcp_kql_server/ --disable=C0301,C0114,C0115,C0116
   ```

4. **Update documentation** if needed

5. **Update RELEASE_NOTES.md** with your changes

### Submitting PR

1. Push your branch to your fork:
   ```bash
   git push origin <branch-name>
   ```

2. Create a Pull Request on GitHub

3. Fill out the PR template with:
   - Description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (if UI changes)

### PR Review Process

- All PRs require at least one review
- CI/CD checks must pass
- Address all review comments
- Squash commits before merge (if requested)

---

## Coding Standards

### Python Style Guide

- Follow **PEP 8** style guide
- Use **type hints** for all function signatures
- Maximum line length: **120 characters**
- Use **docstrings** for all public functions and classes

### Example Function

```python
def execute_query(
    query: str,
    cluster_url: str,
    database: str,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Execute a KQL query against Azure Data Explorer.

    Args:
        query: The KQL query string to execute
        cluster_url: The Kusto cluster URL (e.g., https://cluster.kusto.windows.net)
        database: The database name to query
        timeout: Query timeout in seconds (default: 300)

    Returns:
        Dictionary containing:
            - success: bool indicating if query succeeded
            - data: List of result rows (if successful)
            - error: Error message (if failed)

    Raises:
        ValueError: If query is empty or cluster_url is invalid
        ConnectionError: If unable to connect to cluster

    Example:
        >>> result = execute_query("StormEvents | take 10", 
        ...                        "https://help.kusto.windows.net",
        ...                        "Samples")
        >>> print(result["success"])
        True
    """
    # Implementation here
    pass
```

### Import Order

```python
# Standard library imports
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
from azure.kusto.data import KustoClient

# Local imports
from .constants import DEFAULT_TIMEOUT
from .memory import get_memory_manager
```

---

## Testing Guidelines

### Test Structure

```
tests/
├── __init__.py
├── test_constants.py      # Test constants and configuration
├── test_execute_kql.py    # Test query execution
├── test_kql_auth.py       # Test authentication
├── test_mcp_server.py     # Test MCP server functions
├── test_memory.py         # Test memory manager
├── test_utils.py          # Test utility functions
└── conftest.py            # Shared fixtures
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestQueryExecution:
    """Tests for query execution functionality."""

    def test_valid_query_execution(self):
        """Test that valid KQL queries execute successfully."""
        # Arrange
        query = "StormEvents | take 10"
        
        # Act
        result = execute_query(query, cluster, database)
        
        # Assert
        assert result["success"] is True
        assert len(result["data"]) <= 10

    def test_invalid_query_returns_error(self):
        """Test that invalid queries return proper error messages."""
        # Arrange
        query = "INVALID SYNTAX HERE"
        
        # Act
        result = execute_query(query, cluster, database)
        
        # Assert
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_async_query_execution(self):
        """Test async query execution."""
        result = await async_execute_query(query, cluster, database)
        assert result["success"] is True
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=mcp_kql_server --cov-report=html

# Run specific test file
python -m pytest tests/test_memory.py -v

# Run tests matching pattern
python -m pytest tests/ -v -k "test_query"
```

---

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description of function.

    Longer description if needed. Can span multiple lines
    and include more details about the function behavior.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer

    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
```

### Updating Documentation

- Update `README.md` for user-facing changes
- Update `docs/` for detailed documentation
- Update `RELEASE_NOTES.md` for all changes
- Add inline comments for complex logic

---

## Questions?

- Open an issue for questions or discussions
- Join our community discussions on GitHub
- Contact maintainers: arjuntrivedi42@yahoo.com

---

## Recognition

Contributors will be recognized in:
- `RELEASE_NOTES.md` for significant contributions
- GitHub contributors page
- Special thanks section in README (for major contributions)

Thank you for contributing to MCP KQL Server! 🎉
