
import requests

"""

    #print(element.prettify())
    #print(element.contents)
    #print(len(element.contents))
    #print(element.contents[0].contents[0].contents[0].contents[0].contents[0])
    #print(element.contents[1].contents[0].contents[1].contents[0])
    #print(element.contents[2].contents[0].contents[1].contents[0].contents[0].contents[0])

"""

# Configuration
chrome_path = 'C:\Program Files (x86)\Google\Chrome\Application'

# ---------------------------------------------------
# Helper methods
# ---------------------------------------------------
def get_html(url):
    """"""
    from selenium import webdriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome()
    driver.get(url)
    html_content = driver.page_source
    driver.quit()
    return html_content

url = "https://olympics.com/en/olympic-games/tokyo-2020/results/athletics/men-s-10000m"
#html_content = get_html(url)
#print(html_content)


# Path to the HTML file
file_path = "./sample.html"

# Save the HTML content to the file
#with open(file_path, "w", encoding="utf-8") as file:
#    file.write(html_content)

# Load HTML content from file with UTF-8 encoding
with open(file_path, "r", encoding="utf-8") as file:
    loaded_html_content = file.read()

print("Loaded HTML content:")
print(loaded_html_content)

from bs4 import BeautifulSoup

# Parse the HTML content
soup = BeautifulSoup(loaded_html_content, 'html.parser')

# Find all elements with data-cy attribute equal to single-athlete-result-row
elements = soup.find_all(attrs={"data-cy": "single-athlete-result-row"})

# Libraries
import pandas as pd

# Print each element
result = []
for element in elements:
    # Extract information
    rank = element.contents[0].find('span').get_text(strip=True)
    country = element.contents[1].find('span').get_text(strip=True)
    athlete = element.contents[2].find('h3').get_text(strip=True)
    # Append
    result.append({'athlete': athlete, 'country': country, 'rank': rank})

# Show results
print(pd.DataFrame(result))

import sys
sys.exit()
print("\n\nElements:")
print(len(elements))


# Get rank
elements = soup.find_all(attrs={"data-cy": "medal-main"})
ranks = [e.find('span').get_text() for e in elements]
print(ranks)

# Get country
elements = soup.find_all(attrs={"data-cy": "picture-wrapper"})

countries = [e.find('span').get_text() for e in elements]
print(countries)

"""
# Find all <span> elements with the specified class
spans = soup.find_all('span', class_='sc-bdnyFh JizeU text--sm-body')

print(spans)

# Extract and print the text from each <span> element
for span in spans:
    print(span.get_text())

# Find all elements containing the class 'JiZeu'
elements = soup.find_all(class_='JiZeu')

# Print each element's text
for element in elements:
    print(element)
"""


import sys
sys.exit()



def get_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.RequestException as e:
        print(f"Error occurred: {e}")
        return None

# Example usage
#url = "https://olympics.com/en/olympic-games/tokyo-2020/results/athletics/men-s-10000m"
#html_content = get_html(url)

#print(html_content)





def get_html(url):
    # Path to your WebDriver (e.g., ChromeDriver)
    driver_path = chrome_path
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Chrome() #executable_path=driver_path, options=options)
    driver.get(url)
    html_content = driver.page_source
    driver.quit()
    return html_content

# Example usage
#url = "https://olympics.com/en/olympic-games/tokyo-2020/results/athletics/men-s-10000m"
#html_content = get_html(url)
print(html_content)

import urllib.request

import sys
sys.exit()
def get_html(url):
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode()
    except urllib.error.URLError as e:
        print(f"Error occurred: {e}")
        return None

# Example usage
url = "https://olympics.com/en/olympic-games/tokyo-2020/results/athletics/men-s-10000m"
#html_content = get_html(url)
#print(html_content)

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample HTML</title>
</head>
<body>
    <h1>Hello, World!</h1>
    <p>This is a sample HTML file.</p>
</body>
</html>
"""

# Path to the HTML file
file_path = "./sample.html"

# Save the HTML content to the file
with open(file_path, "w") as file:
    file.write(html_content)