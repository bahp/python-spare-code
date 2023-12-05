"""

List of free proxies:


"""

import requests

# Exceptions
from requests.exceptions import ProxyError
from requests.exceptions import ReadTimeout
from requests.exceptions import ConnectTimeout

# ----------------------------------------------------
#                    CONSTANTS
# ----------------------------------------------------
# .. note: This URL is used to identify the IP address
#          that has been used to make the get. Th
URL = 'https://ip.oxylabs.io/ip'
URL = "https://bahp.github.io/portfolio-academic/teaching/"

# Dictionary with proxies. There is a list of free proxies
# which also allows automatic extraction. For more info
# see: https://geonode.com/free-proxy-list
scheme_proxy_map = {
    "cz": "https://79.110.40.129:8080",
    #"mx": "https://177.242.151.143:8080"
}

# Timeout
TIMEOUT_IN_SECONDS = 10


# ----------------------------------------------------
#                    Main
# ----------------------------------------------------
# See local IP address.
response = requests.get(URL)
print("IP: %s" % response.text)



try:
    # Use one of the proxies.
    response = requests.get(URL,
        proxies=scheme_proxy_map,
        timeout=TIMEOUT_IN_SECONDS)
except (ProxyError, ReadTimeout, ConnectTimeout) as error:
        print('Unable to connect to the proxy: ', error)
else:
    print("IP: %s" % response.text)