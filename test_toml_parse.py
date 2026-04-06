import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        print("tomli not installed, installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tomli"])
        import tomli as tomllib

try:
    with open('pyproject.toml', 'rb') as f:
        config = tomllib.load(f)
    print("✓ Parsed successfully")
    print("Black config:", config.get('tool', {}).get('black', {}))
except Exception as e:
    print(f"✗ Parsing failed: {type(e).__name__}: {e}")
