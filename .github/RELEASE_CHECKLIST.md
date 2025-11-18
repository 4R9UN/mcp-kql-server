# Release Checklist for MCP KQL Server

Use this checklist when preparing a new release.

## Pre-Release Steps

### 1. Update Version Numbers
Update version in all files to the new version (e.g., `2.0.9`):

- [ ] `pyproject.toml` - line 3: `version = "2.0.9"`
- [ ] `mcp_kql_server/__init__.py` - line 21: `__version__ = "2.0.9"`
- [ ] `mcp_kql_server/constants.py` - line 19: `__version__ = "2.0.9"`
- [ ] `mcp_kql_server/constants.py` - line 88: `FASTAPI_VERSION = "2.0.9"`
- [ ] `server.json` - line 5: `"version": "2.0.9"`
- [ ] `server.json` - line 10: `"version": "2.0.9"` (in packages array)

### 2. Update Release Notes
- [ ] Add new section in `RELEASE_NOTES.md` with version, date, and changes
- [ ] Document all new features, bug fixes, and improvements
- [ ] Include breaking changes (if any)

### 3. Run Quality Checks
```bash
# Run linting
ruff check .

# Run tests (if available)
pytest

# Verify version consistency
python -c "import tomli; print(f'pyproject.toml: {tomli.load(open(\"pyproject.toml\", \"rb\"))[\"project\"][\"version\"]}')"
python -c "from mcp_kql_server import __version__; print(f'__init__.py: {__version__}')"
```

### 4. Build and Test Package Locally
```bash
# Build package
python -m build

# Test installation in virtual environment
python -m venv test_env
test_env\Scripts\activate
pip install dist/mcp_kql_server-2.0.9-py3-none-any.whl
# Test the package
deactivate
rm -r test_env
```

## Release Steps

### 5. Commit Changes
```bash
git add .
git commit -m "Bump version to 2.0.9"
git push origin main
```

### 6. Publish to PyPI
```bash
# Build fresh distribution
python -m build

# Upload to PyPI
twine upload dist/*
```

### 7. Create GitHub Release
```bash
# Create and push tag
git tag v2.0.9
git push origin v2.0.9
```

### 8. Verify Automated Publishing
The GitHub Actions workflow will automatically:
- [ ] Download latest mcp-publisher
- [ ] Authenticate via GitHub OIDC
- [ ] Publish to MCP Registry

Monitor at: https://github.com/4R9UN/mcp-kql-server/actions

### 9. Verify Publication
```bash
# Check PyPI
curl https://pypi.org/pypi/mcp-kql-server/json

# Check MCP Registry
curl "https://registry.modelcontextprotocol.io/v0/servers?search=io.github.4R9UN/mcp-kql-server"
```

## Post-Release Steps

### 10. Update Documentation
- [ ] Update README.md if needed
- [ ] Update any version-specific documentation
- [ ] Check that badges are displaying correctly

### 11. Announce Release
- [ ] Create GitHub Release with release notes
- [ ] Update project website (if applicable)
- [ ] Post to relevant communities (if applicable)

## Troubleshooting

### Common Issues

**Version already exists on PyPI:**
- Cannot republish the same version to PyPI
- Bump to next version (e.g., 2.0.10)

**MCP Registry duplicate version error:**
- Version already published to MCP Registry
- Bump to next version

**GitHub Actions workflow fails:**
- Check workflow logs at: https://github.com/4R9UN/mcp-kql-server/actions
- Verify OIDC permissions are set correctly
- Ensure server.json is valid

**Version mismatch:**
- Use grep to find all version references: `grep -r "2\.0\.[0-9]" --include="*.py" --include="*.toml" --include="*.json"`
- Update all occurrences to match

## Quick Version Bump Script

```bash
# Save this as bump_version.sh
OLD_VERSION="2.0.8"
NEW_VERSION="2.0.9"

# Update all version files
sed -i "s/$OLD_VERSION/$NEW_VERSION/g" pyproject.toml
sed -i "s/$OLD_VERSION/$NEW_VERSION/g" mcp_kql_server/__init__.py
sed -i "s/$OLD_VERSION/$NEW_VERSION/g" mcp_kql_server/constants.py
sed -i "s/$OLD_VERSION/$NEW_VERSION/g" server.json

echo "Version bumped from $OLD_VERSION to $NEW_VERSION"
echo "Remember to update RELEASE_NOTES.md manually!"
```

## Version History Template

```markdown
## 📦 **v2.0.9 - [Release Title]**

> **[Brief Description]** 🚀

**Release Date**: [Date]
**Author**: Arjun Trivedi
**Email**: arjuntrivedi42@yahoo.com
**Repository**: https://github.com/4R9UN/mcp-kql-server

### 🚀 **What's New in v2.0.9**

#### **1. [Feature Category]**
- **Feature 1**: Description
- **Feature 2**: Description

#### **2. [Bug Fixes]**
- **Fix 1**: Description
- **Fix 2**: Description

#### **3. [Improvements]**
- **Improvement 1**: Description
- **Improvement 2**: Description

### 🔧 **Technical Changes**
- Change 1
- Change 2

### 📝 **Documentation**
- Update 1
- Update 2

### ⚠️ **Breaking Changes** (if any)
- Breaking change 1
- Breaking change 2
```
