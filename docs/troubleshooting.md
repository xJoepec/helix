# Troubleshooting Guide

This guide helps resolve common issues when using Helix.

## PyTorch Errors

### Error: "PyTorch is required for the demo/helixenv/analyze command"

**Problem:** PyTorch is not installed in your environment.

**Solutions:**

1. **Quick fix - Install PyTorch only:**
   ```bash
   pip install torch
   ```

2. **Recommended - Install with Helix extras:**
   ```bash
   # Install with PyTorch and visualization tools
   pip install -e '.[viz]'

   # Or install everything
   pip install -e .[full]
   ```

3. **Using uv package manager:**
   ```bash
   uv pip install torch
   ```

**Note:** Most Helix CLI commands require PyTorch. Only the Python API for partition extraction works without it.

### Error: "No module named 'torch'"

**Problem:** Your Python interpreter can't find PyTorch.

**Possible causes:**
- PyTorch is not installed
- Virtual environment is not activated
- PyTorch is installed in a different Python environment

**Solution:**
1. Make sure your virtual environment is activated:
   ```bash
   source .venv/bin/activate
   ```

2. Install PyTorch:
   ```bash
   pip install torch
   ```

## Installation Issues

### Virtual Environment Problems

**Problem:** Commands fail with permission errors or package conflicts.

**Solution:** Always use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[full]
```

### Using uv Instead of pip

This project supports both pip and uv package managers. If you see a `uv.lock` file:

```bash
# Install uv if needed
pip install uv

# Install dependencies with uv
uv pip install -e .[full]
```

## Optional Dependencies

### Missing Visualization Tools

**Error:** Plots don't display or save

**Solution:** Install visualization extras:
```bash
pip install -e .[viz]
# Or manually:
pip install matplotlib
```

### Missing Symbolic Math

**Error:** K-theory computations fail

**Solution:** Install sympy:
```bash
pip install sympy
```

### Missing Persistent Homology

**Error:** Topology features unavailable

**Solution:** Install ripser:
```bash
pip install ripser
```

### Missing Terminal UI

**Error:** "Textual is not installed"

**Solution:** Install TUI extras:
```bash
pip install -e .[tui]
```

## Common Runtime Issues

### Memory Errors with Large Models

**Problem:** Out of memory when analyzing large models

**Solution:**
- Reduce batch size
- Use sparse representations
- Limit the number of layers analyzed

### Slow Performance

**Problem:** Analysis takes too long

**Solutions:**
- Ensure PyTorch is using GPU if available
- Reduce `ulam_bins` parameter
- Reduce `ulam_samples_per_cell` parameter
- Use `--no-train` flag to skip training

### File Not Found Errors

**Problem:** Can't load model or data files

**Solution:** Use absolute paths or ensure files are in the current directory:
```bash
# Use absolute path
./helix analyze --model-path /full/path/to/model.pt

# Or change to the directory first
cd /path/to/your/files
./helix analyze --model-path model.pt
```

## Getting Help

If you encounter issues not covered here:

1. Check the main documentation:
   - [README.md](../README.md) - Overview and quick start
   - [docs/getting-started.md](getting-started.md) - Detailed setup
   - [docs/api.md](api.md) - Python API reference

2. Report issues:
   - GitHub Issues: https://github.com/xJoepec/helix

3. Check your installation:
   ```bash
   # Show installed packages
   pip list | grep -E "torch|helix|numpy"

   # Show Python version
   python --version

   # Show Helix version
   ./helix --help | head -1
   ```
