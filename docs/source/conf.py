# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

from click import prompt

# sys.path.insert(0, os.path.abspath("../.."))
# sys.path.insert(0, os.path.abspath("../../src"))

# Always resolve paths relative to this conf.py file
conf_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(conf_dir, "../.."))
src_dir = os.path.abspath(os.path.join(conf_dir, "../../DomainMind"))
data_dir = os.path.abspath(os.path.join(conf_dir, "../../data"))
prompts_dir = os.path.abspath(os.path.join(conf_dir, "../../prompts"))

sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, data_dir)
sys.path.insert(0, prompts_dir)


project = "DomainMind"
copyright = "2025, Salman Fariz Navas"
author = "Salman Fariz Navas"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",  # Optional: Adds links to source code
    "sphinx.ext.napoleon",  # Optional: Supports Google/NumPy style docstrings
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
