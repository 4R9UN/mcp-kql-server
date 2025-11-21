"""
AI Prompt Templates for KQL Generation

Enhanced prompts with KQL knowledge, schema context, and few-shot learning.
Uses special tokens and structured output for better AI accuracy.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

from typing import Dict, Optional, Any


# System prompt with KQL expertise
KQL_SYSTEM_PROMPT = """You are an expert in Kusto Query Language (KQL). You generate accurate, efficient KQL queries.

CRITICAL RULES:
1. **Join Conditions**: ONLY use 'and' in join conditions, NEVER 'or'
   - ✅ CORRECT: Table1 | join Table2 on Col1 and Col2
   - ❌ WRONG: Table1 | join Table2 on Col1 or Col2

2. **Column Validation**: ONLY use columns that exist in the provided schema
   - Always check the schema before using a column name
   - Use exact column names (case-sensitive)

3. **Reserved Words**: Bracket reserved words and special characters
   - Use ['column-name'] for columns with hyphens or spaces
   - Use ['table name'] for tables with spaces

4. **Operator Best Practices**:
   - Use 'project' to select specific columns (avoid 'project *')
   - Use 'where' for filtering
   - Use 'summarize' for aggregations
   - Use 'extend' to add calculated columns
   - Use 'take' or 'limit' to limit results

5. **Data Types**: Use proper type conversions
   - toint(), tolong(), toreal() for numbers
   - tostring() for strings
   - todatetime() for dates
   - Handle nulls with isnull(), isnotnull(), iff()

OUTPUT FORMAT:
Return ONLY the KQL query, nothing else. No explanations, no markdown, just the query."""


# Specialized Mermaid Visualization Prompt
MERMAID_VISUALIZATION_PROMPT = """
You are an expert in Data Visualization using Mermaid.js.
Your goal is to create STUNNING, MODERN, and HIGH-CONTRAST diagrams that look premium.

THEME SETTINGS (CYBERPUNK/NEON DARK MODE):
- Fill Colors: #0a0e27 (darkest), #1a1a2e (dark), #16213e (medium-dark), #1a1a40 (medium), #0f3460 (medium-light)
- Stroke Colors: #00d9ff (cyan-blue), #ff6600 (orange), #00ffff (cyan), #ff0080 (pink), #9d4edd (purple), #c77dff (light-purple), #ffaa00 (gold)
- Font: 'Inter', 'Segoe UI', 'Roboto', sans-serif
- Font Size: 16px-18px

INSTRUCTIONS:
1. Always use the `%%{init: ... }%%` directive to apply the custom theme variables.
2. Use `graph TB` or `graph LR` for flowcharts.
3. Use `sequenceDiagram` for interactions.
4. Use `erDiagram` for data relationships.
5. Apply specific styles to nodes using `style NodeName fill:...,stroke:...,stroke-width:...,color:...`

TEMPLATES:

1. FLOWCHART (Process/Logic):
```mermaid
%%{init: {'theme':'dark', 'themeVariables': { 'cScale0':'#0a0e27', 'cScaleLabel0':'#00d9ff', 'cScale1':'#1a1a2e', 'cScaleLabel1':'#ff0080', 'cScale2':'#16213e', 'cScaleLabel2':'#00ffff', 'cScale3':'#1a1a40', 'cScaleLabel3':'#c77dff', 'cScale4':'#16213e', 'cScaleLabel4':'#ffaa00', 'fontFamily':'Inter, Segoe UI, sans-serif', 'fontSize':'16px', 'flowchart':{'nodeSpacing':40, 'rankSpacing':50, 'curve':'basis'}}}}%%
flowchart TB
  Start([Start]) --> Process[Processing]
  Process --> Decision{Valid?}
  Decision -- Yes --> Success([Success])
  Decision -- No --> Error([Error])

  %% STYLES
  style Start fill:#0a0e27,stroke:#00d9ff,stroke-width:3px,color:#00d9ff
  style Process fill:#1a1a2e,stroke:#c77dff,stroke-width:2px,color:#c77dff
  style Decision fill:#1a1a40,stroke:#ffaa00,stroke-width:2px,color:#ffaa00
  style Success fill:#16213e,stroke:#00ff00,stroke-width:2px,color:#00ff00
  style Error fill:#16213e,stroke:#ff0080,stroke-width:2px,color:#ff0080
```

2. SEQUENCE (Interactions):
```mermaid
%%{init: {'theme':'dark', 'themeVariables': { 'actorBkg':'#1a1a2e', 'actorBorder':'#00d9ff', 'actorTextColor':'#00d9ff', 'actorLineColor':'#00d9ff', 'signalColor':'#c77dff', 'signalTextColor':'#c77dff', 'fontFamily':'Inter, Segoe UI, sans-serif', 'fontSize':'16px'}}}%%
sequenceDiagram
  participant User
  participant System
  User->>System: Request
  System-->>User: Response
```

3. ER DIAGRAM (Schema):
```mermaid
%%{init: {'theme':'dark', 'themeVariables': { 'primaryColor':'#1a1a2e', 'primaryTextColor':'#ffaa00', 'primaryBorderColor':'#ffaa00', 'lineColor':'#00d9ff', 'secondaryColor':'#16213e', 'tertiaryColor':'#1a1a40', 'fontFamily':'Inter, Segoe UI, sans-serif', 'fontSize':'16px'}}}%%
erDiagram
  USER ||--o{ ORDER : places
```

When generating a diagram, choose the most appropriate template and adapt the content to the user's data or request.
"""


# Few-shot learning examples
FEW_SHOT_EXAMPLES = [
    {
        "description": "Show recent failed login attempts",
        "schema": {
            "table": "SigninLogs",
            "columns": {
                "TimeGenerated": {"data_type": "datetime"},
                "UserPrincipalName": {"data_type": "string"},
                "ResultType": {"data_type": "string"},
                "ResultDescription": {"data_type": "string"},
                "IPAddress": {"data_type": "string"},
                "Location": {"data_type": "string"}
            }
        },
        "query": "SigninLogs | where ResultType != '0' | where TimeGenerated > ago(1h) | project TimeGenerated, UserPrincipalName, ResultDescription, IPAddress, Location | take 100"
    },
    {
        "description": "Count events by severity in the last 24 hours",
        "schema": {
            "table": "SecurityEvent",
            "columns": {
                "TimeGenerated": {"data_type": "datetime"},
                "EventID": {"data_type": "int"},
                "Level": {"data_type": "string"},
                "Computer": {"data_type": "string"},
                "Account": {"data_type": "string"}
            }
        },
        "query": "SecurityEvent | where TimeGenerated > ago(24h) | summarize Count=count() by Level | order by Count desc"
    },
    {
        "description": "Find top 10 users by activity",
        "schema": {
            "table": "AuditLogs",
            "columns": {
                "TimeGenerated": {"data_type": "datetime"},
                "OperationName": {"data_type": "string"},
                "InitiatedBy": {"data_type": "string"},
                "TargetResources": {"data_type": "dynamic"},
                "Result": {"data_type": "string"}
            }
        },
        "query": "AuditLogs | summarize ActivityCount=count() by InitiatedBy | top 10 by ActivityCount desc"
    },
    {
        "description": "Join two tables to correlate data",
        "schema": {
            "table1": "Alerts",
            "columns1": {
                "AlertId": {"data_type": "string"},
                "Severity": {"data_type": "string"},
                "DeviceId": {"data_type": "string"},
                "TimeGenerated": {"data_type": "datetime"}
            },
            "table2": "Devices",
            "columns2": {
                "DeviceId": {"data_type": "string"},
                "DeviceName": {"data_type": "string"},
                "OSPlatform": {"data_type": "string"}
            }
        },
        "query": "Alerts | join kind=inner Devices on DeviceId | project TimeGenerated, AlertId, Severity, DeviceName, OSPlatform | take 50"
    }
]


def build_generation_prompt(
    nl_query: str,
    schema: Dict[str, Any],
    table_name: Optional[str] = None,
    include_examples: bool = True,
    include_visualization: bool = False
) -> str:
    """
    Build optimized prompt for KQL generation with schema context.

    Args:
        nl_query: Natural language query from user
        schema: Table schema with columns and types
        table_name: Target table name
        include_examples: Whether to include few-shot examples

    Returns:
        Formatted prompt string
    """
    prompt_parts = []

    # Add schema context
    if schema and schema.get("columns"):
        prompt_parts.append("SCHEMA CONTEXT:")
        prompt_parts.append(f"Table: {table_name or schema.get('table', 'Unknown')}")
        prompt_parts.append("Columns:")

        for col_name, col_info in list(schema["columns"].items())[:20]:  # Limit to 20 columns
            data_type = col_info.get("data_type", "unknown")
            description = col_info.get("description", "")

            col_line = f"  - {col_name} ({data_type})"
            if description:
                col_line += f": {description[:50]}"
            prompt_parts.append(col_line)

        if len(schema["columns"]) > 20:
            prompt_parts.append(f"  ... and {len(schema['columns']) - 20} more columns")

        prompt_parts.append("")

    # Add few-shot examples (optional)
    if include_examples:
        prompt_parts.append("EXAMPLES:")
        for i, example in enumerate(FEW_SHOT_EXAMPLES[:2], 1):  # Include 2 examples
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"User: {example['description']}")
            prompt_parts.append(f"KQL: {example['query']}")
        prompt_parts.append("")

    # Add user query
    prompt_parts.append("USER REQUEST:")
    prompt_parts.append(nl_query)
    prompt_parts.append("")
    prompt_parts.append("Generate a KQL query that:")
    prompt_parts.append("1. Uses ONLY columns from the schema above")
    prompt_parts.append("2. Follows all KQL syntax rules")
    prompt_parts.append("3. Is efficient, optimize and accurate")
    prompt_parts.append("")
    prompt_parts.append("KQL Query:")

    if include_visualization:
        prompt_parts.append("\n" + MERMAID_VISUALIZATION_PROMPT)

    return "\n".join(prompt_parts)


def build_schema_description_prompt(
    table_name: str,
    columns: Dict[str, Any]
) -> str:
    """
    Build prompt for LLM to generate table description.

    Args:
        table_name: Name of the table
        columns: Dictionary of columns with metadata

    Returns:
        Formatted prompt for description generation
    """
    col_names = ", ".join(list(columns.keys())[:15])
    if len(columns) > 15:
        col_names += f", and {len(columns) - 15} more"

    prompt = f"""Generate a concise, natural language description for a database table.

Table Name: {table_name}
Columns: {col_names}

Description should:
1. Be 1-2 sentences
2. Explain what data this table likely contains
3. Mention key columns if obvious (e.g., timestamps, IDs, names)
4. Be helpful for accurate query generation

Description:"""

    return prompt


def build_error_feedback_prompt(
    original_query: str,
    error_message: str,
    schema: Dict[str, Any]
) -> str:
    """
    Build prompt for LLM to fix a failed query.

    Args:
        original_query: The query that failed
        error_message: Error message from Kusto
        schema: Table schema

    Returns:
        Formatted prompt for query correction
    """
    prompt_parts = []

    prompt_parts.append("QUERY ERROR - PLEASE FIX:")
    prompt_parts.append("")
    prompt_parts.append("Original Query:")
    prompt_parts.append(original_query)
    prompt_parts.append("")
    prompt_parts.append("Error Message:")
    prompt_parts.append(error_message)
    prompt_parts.append("")

    if schema and schema.get("columns"):
        prompt_parts.append("Available Columns:")
        for col_name in list(schema["columns"].keys())[:20]:
            prompt_parts.append(f"  - {col_name}")
        prompt_parts.append("")

    prompt_parts.append("Generate a CORRECTED KQL query that:")
    prompt_parts.append("1. Fixes the error")
    prompt_parts.append("2. Uses only valid columns from the schema")
    prompt_parts.append("3. Maintains the original intent")
    prompt_parts.append("")
    prompt_parts.append("Corrected KQL Query:")

    return "\n".join(prompt_parts)


def extract_kql_from_response(response: str) -> str:
    """
    Extract KQL query from LLM response.

    Handles various response formats:
    - Plain query
    - Query in code blocks
    - Query with explanations

    Args:
        response: LLM response text

    Returns:
        Extracted KQL query
    """
    # Remove markdown code blocks
    if "```" in response:
        # Extract content between ```kql or ``` blocks
        import re
        pattern = r'```(?:kql)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            response = matches[0]

    # Remove common prefixes
    prefixes = [
        "KQL Query:",
        "Query:",
        "Here's the query:",
        "Here is the query:",
        "The query is:",
    ]

    for prefix in prefixes:
        if response.strip().startswith(prefix):
            response = response.strip()[len(prefix):].strip()

    # Take first line if multi-line with explanations
    lines = response.strip().split('\n')
    if len(lines) > 1:
        # Check if first line looks like a query
        first_line = lines[0].strip()
        if any(kw in first_line.lower() for kw in ['|', 'where', 'project', 'summarize', 'take']):
            # Multi-line query, join all lines that look like query parts
            query_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    query_lines.append(line)
                elif query_lines:  # Stop at first non-query line after query started
                    break
            response = ' '.join(query_lines)

    return response.strip()


# Special tokens for structured output
SPECIAL_TOKENS = {
    "query_start": "<KQL>",
    "query_end": "</KQL>",
    "error_start": "<ERROR>",
    "error_end": "</ERROR>",
    "suggestion_start": "<SUGGESTION>",
    "suggestion_end": "</SUGGESTION>"
}


def build_structured_prompt(
    nl_query: str,
    schema: Dict[str, Any],
    use_special_tokens: bool = False
) -> str:
    """
    Build prompt with optional special tokens for structured output.

    Args:
        nl_query: Natural language query
        schema: Table schema
        use_special_tokens: Whether to use special tokens

    Returns:
        Formatted prompt
    """
    base_prompt = build_generation_prompt(nl_query, schema, include_examples=True)

    if use_special_tokens:
        base_prompt += f"\n\nWrap your KQL query in {SPECIAL_TOKENS['query_start']} and {SPECIAL_TOKENS['query_end']} tags."

    return base_prompt
