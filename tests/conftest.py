"""Make `reap` importable from the src-layout without an editable install.

When the package has been installed via `uv pip install --editable .` this is a
no-op; when running tests directly (e.g. `python -m pytest tests/`) it inserts
`src` onto sys.path so `import reap` resolves.
"""
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))