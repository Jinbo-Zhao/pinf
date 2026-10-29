"""Console entry point ``pinf-gui``: starts the bundled Streamlit app."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "app.py"

MISSING_STREAMLIT_MESSAGE = (
    "The pinf GUI needs Streamlit, which is not installed. "
    'Install the GUI extra with: pip install "pinf[gui]"'
)


def streamlit_command(extra_args: Sequence[str]) -> list[str]:
    """Command that runs the bundled app; ``extra_args`` go to ``streamlit run``."""
    return [sys.executable, "-m", "streamlit", "run", str(APP_PATH), *extra_args]


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the GUI and return Streamlit's exit code.

    Extra arguments are passed through, e.g. ``pinf-gui --server.port 8600``.
    """
    if importlib.util.find_spec("streamlit") is None:
        print(MISSING_STREAMLIT_MESSAGE, file=sys.stderr)
        return 1
    args = sys.argv[1:] if argv is None else argv
    return subprocess.run(streamlit_command(args), check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
