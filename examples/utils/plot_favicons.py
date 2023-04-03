"""
Plot favicons
=============

This example lists all the libraries installed
in the environment using pip, gets their site
url, downloads the icon and displays it in the
example.

.. note:: The command ``cut`` is not available in windows! It might
          available on Cygwin o Git for windows.

"""

# Libraries
import favicon
import subprocess as sp


class IconHTML:
    """Class to display html in sphinx-gallery."""
    TMP0 = '<img src={url} style="{s}", width={w} height={h}>'
    TMP1 = '<div>'+TMP0+' <span>{name}</span></div>'

    """Class to display icons on sphinx-gallery."""
    def __init__(self, d, width=25, height=25, verbose=0):
        self.d = d
        self.width = width
        self.height = height
        self.style = "display: inline; vertical-align:middle;"
        self.verbose = verbose

    def _repr_html_short_(self):
        return ' '.join([self.TMP0.format(url=v,
            w=self.width, h=self.height, s=self.style)
                for k,v in self.d.items()])

    def _repr_html_long_(self):
        return ' '.join([self.TMP1.format(url=v,
            w=self.width, h=self.height, s=self.style, name=k.lower())
                for k, v in self.d.items()])

    def _repr_html_(self):
        if self.verbose == 0:
            return self._repr_html_short_()
        return self._repr_html_long_()


# List of libraries for which the icon (if found)
# should be included in the output.
INCLUDE = [
    'pandas',
    'Flask',
    'imblearn',
    'numba',
    'numpy',
    'plotly',
    'PyYAML',
    'scipy',
    'seaborn',
    'statsmodels',
    'alabaster',
    'attrs',
    'Babel',
    'bokeh',
    'joblib',
    'nltk',
    'notebook',
    'torch',
    'matplotlib',
    'pillow',
    'pygments',
    'pytest',
    'tqdm',
    'urllib3',
    'future'
]

# Define command to list packages and urls
COMMAND = "pip list --format=freeze | cut -d= -f1 | xargs pip show | "
COMMAND+= "awk '/^Name/{printf $2} /^Home-page/{print \": \"$2}'"

# List of package name and url.
output = sp.getoutput(COMMAND)

# Show
print(output)

# Create dictionary
d = {}
for line in output.split("\n")[2:]:
    # Find name and url
    name, url = line.split(': ')
    if name not in INCLUDE:
        continue
    # Store name and url
    icons = favicon.get(url)
    if len(icons):
        d[name] = icons[0].url


# %%
#
aux = IconHTML(d)
aux

# %%
#
aux = IconHTML(d, verbose=1)
aux

# %%
#

# Create dictionary
d = {}
for line in output.split("\n")[2:]:
    # Find name and url
    name, url = line.split(': ')
    if not url.startswith('https:'):
        continue
    # Store name and url
    icons = favicon.get(url)
    for i, ico in enumerate(icons):
        d['%s-%s' % (name, i)] = ico.url

# %%
#
aux = IconHTML(d, verbose=1)
aux