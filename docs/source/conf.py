# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
#import os
#import sys
#sys.path.insert(0, os.path.abspath('../../%s/' % 'pkg'))


from datetime import date
today = str(date.today().year)

# -- Project information -----------------------------------------------------

project = 'python-spare-code'
copyright = '2021-%s, Bernard Hernandez' % today
author = 'Bernard Hernandez'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    #'sphinx.ext.autodoc',
    #'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',        # docstrings
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',     # gh-pages needs a .nojekyll file
    'sphinx_gallery.gen_gallery'  # example galleries
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
    '../examples/pandas/plot_format05_damien_sepsis.py',     # Does not work
    '../examples/pandas/plot_format06_stacked_oucru_v1.py',  # Does not work
    '../examples/pandas/plot_format06_stacked_oucru_v2.py',  # Does not work
    '../examples/plotly/plot_main06_treemap_v3.py',          # Does not work
    '../examples/utils/plot_drug_resistance_index.py'        # Does not work
]

# ------------------
# Napoleon extension
# ------------------
# Configuration parameters for napoleon extension
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

# ------------------
# Plotly outcomes
# ------------------
# Requires kaleido library
import plotly.io as pio
from plotly.io._sg_scraper import plotly_sg_scraper
#pio.renderers.default = 'sphinx_gallery'    # It does not generate thumbmails
pio.renderers.default = 'sphinx_gallery_png' # It distorts htmls a bit.
image_scrapers = ('matplotlib', plotly_sg_scraper)


# ------------------
# Sphinx gallery
# ------------------
# Information about the sphinx gallery configuration
# https://sphinx-gallery.github.io/stable/configuration.html

# Import library
from sphinx_gallery.sorting import FileNameSortKey
from plotly.io._sg_scraper import plotly_sg_scraper

# Configuration for sphinx_gallery
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': [
        '../../examples/',
        #'../../examples/plotly',
        #'../../examples/matplotlib',
        #'../../examples/shap',
        #'../../examples/tableone',
        #'../../examples/plotly'
    ],
    # path to where to save gallery generated output
    'gallery_dirs': [
        #'../source/_examples/plotly',
        #'../source/_examples/matplotlib',
        #'../source/_examples/shap',
        #'../source/_examples/tableone',
        #'../source/_examples/plotly'
        '../source/_examples',
    ],
    # Other
    'line_numbers': True,
    'download_all_examples': False,
    'within_subsection_order': FileNameSortKey,
    'image_scrapers': image_scrapers,
    'matplotlib_animations': True
}

# ------------------
# Todo extension
# ------------------
todo_include_todos = True



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. If using sphinx-rtd-theme, more configuration
# options are available at:
# https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
# https://sphinx-rtd-trial.readthedocs.io/en/1.1.3/theming.html (not working)
html_theme = 'sphinx_rtd_theme'

# Configuration of sphin_rtd_theme
#html_logo = './_static/images/logo-ls2d-v1.png'
html_favicon = './_static/images/python_icon_191765.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add custom css file.
html_css_files = ['css/custom.css']

# Substitute project name into .rst files when |project_name| is used
rst_epilog = '.. |project_name| replace:: %s' % project