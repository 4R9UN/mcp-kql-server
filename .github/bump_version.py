#!/usr/bin/env python3
"""
Version Bump Script for MCP KQL Server
Automatically updates version across all project files
"""

import re
import sys
from pathlib import Path


def update_file(file_path: Path, old_version: str, new_version: str) -> bool:
    """Update version in a single file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        updated_content = content.replace(old_version, new_version)
        
        if content != updated_content:
            file_path.write_text(updated_content, encoding='utf-8')
            print(f"✅ Updated {file_path}")
            return True
        else:
            print(f"⚠️  No changes needed in {file_path}")
            return False
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False


def main():
    if len(sys.argv) != 3:
        print("Usage: python bump_version.py <old_version> <new_version>")
        print("Example: python bump_version.py 2.0.9 2.1.0")
        sys.exit(1)
    
    old_version = sys.argv[1]
    new_version = sys.argv[2]
    
    # Validate version format
    version_pattern = r'^\d+\.\d+\.\d+$'
    if not re.match(version_pattern, old_version) or not re.match(version_pattern, new_version):
        print("❌ Invalid version format. Use semantic versioning (e.g., 2.0.8)")
        sys.exit(1)
    
    print(f"\n🔄 Bumping version from {old_version} to {new_version}\n")
    
    # Define files to update
    project_root = Path(__file__).parent.parent
    files_to_update = [
        project_root / "pyproject.toml",
        project_root / "mcp_kql_server" / "__init__.py",
        project_root / "mcp_kql_server" / "constants.py",
        project_root / "server.json",
    ]
    
    # Update each file
    updated_count = 0
    for file_path in files_to_update:
        if file_path.exists():
            if update_file(file_path, old_version, new_version):
                updated_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print(f"\n✅ Updated {updated_count} file(s)")
    print("\n📝 Next steps:")
    print(f"   1. Update RELEASE_NOTES.md with v{new_version} changes")
    print("   2. Run: git add .")
    print(f"   3. Run: git commit -m 'Bump version to {new_version}'")
    print("   4. Run: git push origin main")
    print("   5. Run: python -m build")
    print("   6. Run: twine upload dist/*")
    print(f"   7. Run: git tag v{new_version}")
    print(f"   8. Run: git push origin v{new_version}")
    print("\n   GitHub Actions will automatically publish to MCP Registry!")


if __name__ == "__main__":
    main()
