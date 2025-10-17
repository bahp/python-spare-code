"""
02. Visualizing Python Library Favicons
=======================================

This script identifies installed Python packages, fetches their homepage
URLs, and downloads their favicons. It then uses a custom HTML
representation to display these icons in a grid.

.. note:: The command ``cut`` is not available in windows! Thus, the
          code will not run in te standard Windows Command Prompt. However,
          it might be available on Cygwin o Git for windows.

.. note:: The visual output of this script is generated via a special
          _repr_html_() method. This method is automatically detected
          and rendered in rich display environments (Jupyter Notebook/Lab &
          IPython). If you run this as a regular .py file, no images will
          appear in your terminal. To see the output, you must manually get
           the HTML content and save it to a file:
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

def list_packages_command():
    """List packages using linux <cut>."""
    # Define command to list packages and urls
    COMMAND = "pip list --format=freeze | cut -d= -f1 | xargs pip show | "
    COMMAND+= "awk '/^Name/{printf $2} /^Home-page/{print \": \"$2}'"
    # List of package name and url.
    output = sp.getoutput(COMMAND)
    # Return
    return output.split("\n")[2:]


def list_packages_importlib():
    """List packages using importlib."""
    # --- ADD THIS NEW BLOCK ---
    import importlib.metadata

    lines = []
    # Iterate through all installed packages
    for dist in importlib.metadata.distributions():
        # Get package metadata
        meta = dist.metadata
        name = meta['Name']
        url = meta.get('Home-page', '')  # Use .get() for safety

        # Check if the package is in our include list and has a valid URL
        if name in INCLUDE and url.startswith('https'):
            lines.append(f"{name}: {url}")

    return lines



import platform

system = platform.system()

if system == 'Windows':
    print("This is a Windows system.")
elif system == 'Darwin':
    print("This is a macOS system.")
elif system == 'Linux':
    print("This is a Linux system.")
else:
    print(f"This is a different system: {system}")

# Get packages
packages = list_packages_importlib()

# Show
print("\nCommand output:")
print(packages)

# Create dictionary
d = {}
for line in packages:
    # Find name and url
    name, url = line.split(': ')
    if not url.startswith('https:'):
        continue
    if name not in INCLUDE:
        continue
    # Store name and url
    icons = favicon.get(url)
    for i, ico in enumerate(icons):
        d['%s-%s' % (name, i)] = ico.url


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
for line in packages:
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

# %%
# Write the string to a file

#
from pathlib import Path
output_dir = Path('./objects/main02')
output_dir.mkdir(parents=True, exist_ok=True)

with open("%s/icons.html" % output_dir, "w") as f:
    f.write(aux._repr_html_())
