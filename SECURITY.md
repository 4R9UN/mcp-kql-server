# Security Policy

## Supported Versions

The following versions of MCP KQL Server are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.1.x   | :white_check_mark: |
| 2.0.x   | :white_check_mark: |
| 1.x.x   | :x:                |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security vulnerabilities by:

1. **Email**: Send details to **arjuntrivedi42@yahoo.com** with subject line: `[SECURITY] MCP KQL Server Vulnerability Report`

2. **GitHub Security Advisory**: Use [GitHub's private vulnerability reporting](https://github.com/4R9UN/mcp-kql-server/security/advisories/new)

### What to Include

Please include the following information in your report:

- **Description**: Clear description of the vulnerability
- **Impact**: Potential impact and severity assessment
- **Reproduction Steps**: Detailed steps to reproduce the issue
- **Affected Versions**: Which versions are affected
- **Suggested Fix**: If you have a suggested fix, include it
- **Your Contact**: How we can reach you for follow-up

### Example Report Format

```
Subject: [SECURITY] SQL Injection in query parameter

Description:
The execute_kql_query function does not properly sanitize the 'query' parameter,
allowing potential KQL injection attacks.

Impact:
An attacker could execute arbitrary KQL queries, potentially accessing
unauthorized data or causing denial of service.

Severity: HIGH

Affected Versions: 2.0.0 - 2.1.0

Reproduction Steps:
1. Call execute_kql_query with query parameter containing malicious KQL
2. The query is executed without sanitization
3. Attacker gains access to unintended data

Suggested Fix:
Implement query parameterization or strict input validation.

Contact: security-researcher@example.com
```

---

## Response Timeline

| Action | Timeline |
|--------|----------|
| Initial Response | Within 48 hours |
| Vulnerability Assessment | Within 7 days |
| Fix Development | Within 30 days (critical: 7 days) |
| Public Disclosure | After fix is released |

---

## Security Best Practices

When using MCP KQL Server, follow these security best practices:

### Authentication

```python
# DO: Use Managed Identity in production
from azure.identity import DefaultAzureCredential
credential = DefaultAzureCredential()

# DON'T: Hardcode credentials
# credential = ClientSecretCredential(tenant_id, client_id, "hardcoded_secret")  # BAD!
```

### Configuration

```python
# DO: Use environment variables for sensitive config
import os
cluster_url = os.environ.get("KUSTO_CLUSTER_URL")
database = os.environ.get("KUSTO_DATABASE")

# DON'T: Hardcode sensitive values
# cluster_url = "https://mycompany.kusto.windows.net"  # Avoid in code
```

### Query Handling

```python
# DO: Validate and sanitize user inputs
def safe_query(table_name: str, limit: int) -> str:
    # Validate table name against known tables
    if table_name not in allowed_tables:
        raise ValueError("Invalid table name")
    # Use parameterized limits
    if not isinstance(limit, int) or limit < 1 or limit > 10000:
        raise ValueError("Invalid limit")
    return f"{table_name} | take {limit}"

# DON'T: Directly interpolate user input
# query = f"{user_input} | take {user_limit}"  # Dangerous!
```

### Logging

```python
# DO: Sanitize logs
logger.info("Query executed for database: %s", database_name)

# DON'T: Log sensitive data
# logger.info(f"Query: {query}, Credentials: {credentials}")  # BAD!
```

---

## Security Features

### Built-in Security

MCP KQL Server includes these security features:

| Feature | Description |
|---------|-------------|
| **Azure AD Authentication** | Supports Managed Identity, Service Principal, Device Code |
| **Query Validation** | Basic KQL syntax validation before execution |
| **Connection Encryption** | All connections use TLS 1.2+ |
| **No Credential Storage** | Credentials are never stored on disk |
| **Timeout Protection** | Configurable query timeouts prevent runaway queries |

### Recommended Security Configuration

```json
{
  "mcpServers": {
    "kql-server": {
      "command": "python",
      "args": ["-m", "mcp_kql_server"],
      "env": {
        "KQL_QUERY_TIMEOUT": "300",
        "KQL_MAX_ROWS": "10000",
        "KQL_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

> **Note**: This server uses **Azure CLI authentication** (`az login`). No service principal credentials (AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET) are required. Simply ensure you are logged in via `az login` before starting the server.

---

## Known Security Considerations

### Current Limitations

1. **Query Content**: The server executes KQL queries as provided. Ensure queries are from trusted sources.

2. **Local Cache**: Schema information is cached locally in SQLite. Ensure appropriate file system permissions.

3. **Log Output**: Query errors may include partial query text in logs. Configure log levels appropriately.

### Mitigation Recommendations

| Risk | Mitigation |
|------|------------|
| Unauthorized data access | Use Azure RBAC to limit database permissions |
| Query injection | Validate and sanitize all user inputs |
| Credential exposure | Use Managed Identity, never hardcode secrets |
| Data leakage in logs | Set appropriate log levels in production |
| Cache tampering | Protect local file system permissions |

---

## Security Updates

Security updates are released as:

- **Critical**: Patch release within 7 days
- **High**: Patch release within 14 days
- **Medium**: Included in next minor release
- **Low**: Included in next major release

### Staying Updated

```bash
# Check for updates
pip index versions mcp-kql-server

# Update to latest
pip install --upgrade mcp-kql-server

# Enable auto-update checks (v2.1.0+)
# The server automatically checks for updates at startup
```

---

## Compliance

### Data Handling

- No customer data is transmitted outside of Azure Data Explorer connections
- Schema metadata is cached locally for performance
- Query results are returned to the MCP client only
- No telemetry or analytics data is collected

### Audit Trail

For compliance requirements, enable detailed logging:

```python
import logging
logging.getLogger("mcp_kql_server").setLevel(logging.DEBUG)
```

---

## Contact

- **Security Issues**: arjuntrivedi42@yahoo.com
- **General Questions**: Open a GitHub issue
- **Repository**: https://github.com/4R9UN/mcp-kql-server

---

Thank you for helping keep MCP KQL Server secure!
