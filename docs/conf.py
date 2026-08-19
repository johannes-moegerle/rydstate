# ruff: noqa: INP001

from __future__ import annotations

from typing import TYPE_CHECKING

import rydstate

if TYPE_CHECKING:
    from sphinx.application import Sphinx

# -- Project information -----------------------------------------------------

project = "RydState"
copyright = "2025, RydState Developers"  # noqa: A001
author = "RydState Developers"

version = rydstate.__version__  # The short X.Y version, use via |version|
release = version  # The full version, including alpha/beta/rc tags, use via |release|

language = "en"


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "_doctrees", "Thumbs.db", ".DS_Store"]  # Ignore these source files and folders
source_suffix = ".rst"
master_doc = "index"
pygments_style = "sphinx"  # syntax highlighting
todo_include_todos = False


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_static_path = ["_static"]


# -- Options for jupyter notebooks -------------------------------------------------
nbsphinx_prolog = """
{% set docname = (env.doc2path(env.docname, base=None) | string).split("/")[-1] %}

.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
    </style>

    <div class="admonition note">
      This page was generated from the Jupyter notebook
      <a class="reference external" href="{{ docname|e }}">{{ docname|e }}</a>.
    </div>
"""


# -- Options forautosummary -------------------------------------------
autosummary_ignore_module_all = False


# -- Options for autodoc -------------------------------------------
autodoc_class_signature = "mixed"  # combine class and __init__ doc
autodoc_typehints = "both"


# -- Render MQDT.model_classes as a clickable list of FModel classes ----------
# By default autodoc prints the raw repr of the ``model_classes`` value, i.e.
# ``[<class '...Yb171_S05_HighN'>, ...]``, which shows full dotted paths and is
# not clickable. The base class docstring hides that raw value (``:meta
# hide-value:``); here we replace the docstring with a bullet list of ``:class:``
# cross-references that render as short, clickable class names.
def document_model_classes(
    _app: Sphinx,
    what: str,
    name: str,
    obj: object,
    _options: object,
    lines: list[str],
) -> None:
    if what not in {"attribute", "data"} or not name.endswith(".model_classes"):
        return
    if not (isinstance(obj, (list, tuple)) and len(obj) > 0):
        return

    if len(lines) > 0 and len(lines[-1].strip()) > 0:
        lines.append("")
    lines.extend([f"* :class:`~{cls.__module__}.{cls.__qualname__}`" for cls in obj])
    lines.append("")


def setup(app: Sphinx) -> dict[str, bool]:
    app.connect("autodoc-process-docstring", document_model_classes)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
