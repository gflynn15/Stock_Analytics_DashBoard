# %% [markdown]
# - Webscrapping the S&P500 for stock symbols

# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Send a request to fetch the page content
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find the first table on the page (where the S&P 500 companies are listed)
table = soup.find('table', {'id': 'constituents'})

# Extract the table headers
headers = [th.text.strip() for th in table.find_all('th')]

# Extract table rows
rows = []
for tr in table.find_all('tr')[1:]:  # Skip the header row
    cells = [td.text.strip() for td in tr.find_all(['td', 'th'])]
    rows.append(cells)

# Create DataFrame
symbols = []

for x in range(len(rows)):
    symbols.append(rows[x][0])
    
symbols = pd.Series(data=symbols, name='Tick_Symbols')

symbols.to_csv(path_or_buf=r"C:\Users\Gabriel Flynn\OneDrive\OneDrive - University of Texas at El Paso\Documents\Python Projects\Yfinance_Stock_Data_Analysis\yfin_dash_app_api_backend\stock_symbols_list.csv",
               index=False)


