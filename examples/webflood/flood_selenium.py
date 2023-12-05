"""

.. note:: Calling your file <selenium.py> will produce import issues.



Install selenium:

  $ pip -m pip install selenium

Download the corresponding browser driver. For this, checkout the first
paragraph within the following link:

https://www.browserstack.com/guide/python-selenium-to-run-web-automation-test

Step 1 — Find chromedriver binary path
To find chromedriver binary path, run the following command in the terminal:

which chromedriver
The output should be similar to:

terminal output
/usr/local/bin/chromedriver

Step 2 — Lift the quarantine for the chromedriver binary
Now you need to tell Mac OS to trust this binary by lifting the quarantine. Do this by the following terminal command:

xattr -d com.apple.quarantine /usr/local/bin/chromedriver
Now, rerun your test or script, and it should be able to run chromedriver without the error.

https://www.zenrows.com/blog/selenium-avoid-bot-detection#follow-the-page-flow

Ideas
-----

1. Detect changes in the corresponding websites and if there are no changes
   within the last N days then the script is probably not working since
   counts are not being increased.

   https://www.geeksforgeeks.org/python-script-to-monitor-website-changes/

2. Rotate IP address and/or use different VPNs.

   https://www.pluralsight.com/guides/advanced-web-scraping-tactics-python-playbook

3. See for deprecation warning (path in WebDriver)

   https://stackoverflow.com/questions/64717302/deprecationwarning-executable-path-has-been-deprecated-selenium-python

4. Create a fake user agent:

   https://stackoverflow.com/questions/70199088/selenium-with-proxy-not-working-wrong-options

"""
# Libraries
import time
import warnings

# Selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def bmcmidm2017():
    """"""
    return int(browser.find_element(By.CLASS_NAME, 'c-article-metrics-bar__count').text.split(" ")[0])

#path = "C:/ProgramData/chocolatey/lib/chromedriver/tools/chromedriver.exe"
path = './drivers/chromedriver_mac64/chromedriver'

url = "https://www.elpais.com"
url = "https://bahp.github.io/portfolio-academic/"
url = "https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-017-0550-1" # 3730
#url = "https://www.frontiersin.org/articles/10.3389/fdgth.2023.1057467/full"
url = "https://www.mdpi.com/2079-6382/10/10/1267"
url = "https://www.frontiersin.org/articles/10.3389/fdgth.2023.1057467/full"
no_views = 50
duration = float(5)

"""
'bmcidm2017': {
    'name': 'One',
    'url': 'https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-017-0550-1',
},
'portfolio': {
    'name': '',
    'url': 'https://bahp.github.io/portfolio-academic/supervision'
}
'demo-micro': {
    'url': 'https://www.youtube.com/watch?v=32pTOcXszyg'
}
'as.com': {
    'url': 'https://as.com/'
},
"""

sites = {
    'whatismyipaddress.com': {
        'url': 'https://whatismyipaddress.com/'
    }
}


# .. note: Since selenium by default starts up a browser with a clean,
#          brand-new profile, you are actually already browsing privately.
#          However, it is also possible to strictly enforce/turn on
#          incognito/private mode anyway.

PROXY = "177.242.151.143:8080"
PROXY = "11.456.448.110:8080"
PROXY = "177.202.59.58:8080"

# Define options
options = webdriver.ChromeOptions()
#options.add_argument('--proxy-server=%s' % PROXY)
options.add_argument("--incognito")
options.add_argument('--no-sandbox')
options.add_argument('--disable-gpu')
#options.add_argument(f'user-agent={userAgent}')
#options.add_argument("--headless")

"""
# Adding argument to disable the AutomationControlled flag 
options.add_argument("--disable-blink-features=AutomationControlled")
# Exclude the collection of enable-automation switches 
options.add_experimental_option("excludeSwitches", ["enable-automation"])
# Turn-off userAutomationExtension 
options.add_experimental_option("useAutomationExtension", False)
# Setting the driver path and requesting a page 
#driver = webdriver.Chrome(options=options)

# Changing the property of the navigator value for webdriver to undefined 
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
"""

# Create service.
service = Service(ChromeDriverManager().install())


# Loop through
for i in range(0, no_views):
    for j in sites.keys():

        # Open browser, get url and scroll down.
        browser = webdriver.Chrome(service=service, options=options)
        #browser = webdriver.Chrome(path, chrome_options=options)
        browser.get(sites[j]['url'])
        #browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Sleep
        time.sleep(duration)

        # Save screenshot
        fpath = "./output/screenshots/%s_%s.png" % (j, i+1)
        browser.save_screenshot(fpath)

        # Quit
        browser.quit()

        # Logging
        print("%s visited %s times" % (j, i+1))
