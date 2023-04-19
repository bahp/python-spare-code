"""
Screenshot web
==============

.. note:: For ome reason, in windows it does not save the images. But
          it does work as expected (more or less) when using macOs.
"""

# Geeric
import yaml
import requests

# Specific
from pathlib import Path
from html2image import Html2Image
from bs4 import BeautifulSoup

# Configuration
OUTPUT = './outputs'
YAML_FILE = './portfolio.yaml'

HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

# Load configuration from file
with open(Path(YAML_FILE)) as file:
    CONFIG = yaml.full_load(file)

# Create folder if it does not exist
p = Path(OUTPUT).mkdir(parents=True, exist_ok=True)

# Create object
hti = Html2Image(output_path=OUTPUT)

# Loop
for info in CONFIG['projects']:


    """
    # Get response
    response = requests.get(info['url'], headers=HEADERS, verify=False)

    # Parse response
    soup = BeautifulSoup(response.text, 'html.parser')

    # Locate elements
    #aux = list(soup.find_all("div", class_="widget-items"))
    aux = soup.select("div.widget-items")

    for i,e in enumerate(aux):

        hti.screenshot(html_str=e,
                       save_as='%s-%s.png' % (info['name'], i),
                       size=eval(info['size']))
    """

    # Get and save image
    hti.screenshot(url=info['url'],
                   save_as='%s.png'%info['name'],
                   size=eval(info['size']))