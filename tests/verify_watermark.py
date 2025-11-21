"""
Test to verify watermark appears in generated reports.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_kql_server.mcp_server import _format_report_markdown

def test_watermark_in_report():
    """Test that watermark appears in formatted reports."""
    # Create a simple test report
    test_report = {
        'summary': '## Executive Summary\nTest summary',
        'analysis': '## Data Analysis\nTest analysis',
        'visualizations': ['```mermaid\ngraph TD\nA-->B\n```'],
        'recommendations': ['Test recommendation 1', 'Test recommendation 2']
    }
    
    # Format the report
    markdown = _format_report_markdown(test_report)
    
    # Check for watermark
    assert 'MCP-KQL-Server' in markdown, "Watermark missing from report!"
    assert 'https://github.com/4R9UN/mcp-kql-server' in markdown, "GitHub link missing from report!"
    assert 'Give stars' in markdown, "Call to action missing from report!"
    
    print("✅ Watermark correctly appears in report")
    print("\nReport preview (last 300 chars):")
    print(markdown[-300:])

if __name__ == "__main__":
    test_watermark_in_report()
