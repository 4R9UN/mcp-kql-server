import sys
import os

# Add parent dir to path AND insert at 0 to prioritize local code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_kql_server.ai_prompts import build_generation_prompt

def test_prompt_enhancements():
    print("Testing prompt enhancements...")
    
    schema = {
        "table": "TestTable",
        "columns": {"Col1": {"data_type": "string"}}
    }
    
    # Test WITHOUT visualization
    prompt_no_viz = build_generation_prompt("Show me data", schema, include_visualization=False)
    if "MERMAID_VISUALIZATION_PROMPT" in prompt_no_viz or "graph TB" in prompt_no_viz:
        print("❌ Visualization prompt included when include_visualization=False")
    else:
        print("✅ Visualization prompt correctly excluded")
        
    # Test WITH visualization
    prompt_with_viz = build_generation_prompt("Show me data", schema, include_visualization=True)
    if "THEME SETTINGS (CYBERPUNK/NEON DARK MODE)" in prompt_with_viz:
        print("✅ Visualization prompt correctly included with Cyberpunk theme")
    else:
        print("❌ Visualization prompt missing or incorrect theme")

if __name__ == "__main__":
    test_prompt_enhancements()
