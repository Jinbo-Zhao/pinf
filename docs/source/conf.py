"""Sphinx configuration for the plausible-inference documentation.

Build locally from the repository root:

    pip install -e ".[docs]"
    sphinx-build -b html docs/source docs/build/html
"""

import shutil
from pathlib import Path

DOCS_SOURCE = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_SOURCE.parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
TUTORIALS_DIR = DOCS_SOURCE / "tutorials"

# Sphinx only reads files under the source directory, so mirror the demo
# notebooks into it on every build (the copies are git-ignored).
TUTORIALS_DIR.mkdir(exist_ok=True)
for notebook in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
    shutil.copy2(notebook, TUTORIALS_DIR / notebook.name)

# -- Project information -----------------------------------------------------

project = "Plausible Inference"
author = "Jinbo Zhao and David J. Eckman"
copyright = f"2026, {author}"  # noqa: A001
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",  # Markdown (MyST) pages and Jupyter notebooks
    "autoapi.extension",  # API reference parsed from source, no import needed
    "sphinx.ext.napoleon",  # Google/NumPy-style docstrings
    "sphinx.ext.mathjax",
]

# ".rst" stays first: sphinx-autoapi writes its reST pages with the first suffix.
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", ".DS_Store", "**.ipynb_checkpoints"]

# README sections are included into pages that already have an H1 title, so
# their headings start below H1; that is intentional.
suppress_warnings = ["myst.header"]

myst_enable_extensions = ["dollarmath", "amsmath"]

# The demos need Gurobi/SCIP, which Read the Docs does not have, so render the
# outputs already stored in the notebooks instead of executing them.
nb_execution_mode = "off"

# -- AutoAPI -----------------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = [str(PROJECT_ROOT / "src" / "plausible_inference")]
autoapi_root = "autoapi"
autoapi_add_toctree_entry = False
autoapi_keep_files = False
autoapi_member_order = "bysource"
autoapi_python_class_content = "both"
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_context = {
    "display_github": True,
    "github_user": "Jinbo-Zhao",
    "github_repo": "pinf",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
