from bs4 import BeautifulSoup
import requests
import pandas as pd
from io import StringIO 

def fetch_sp500_data():
    #Fetches S&P 500 data and returns the DataFrame.
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table", {"id": "constituents"})

    # Fix for Pandas FutureWarning using StringIO
    sp500_df = pd.read_html(StringIO(str(table)))[0]
    
    return sp500_df

if __name__ == "__main__":
    df = fetch_sp500_data()
    df.to_csv(path_or_buf=r"C:\Users\Gabriel Flynn\OneDrive\OneDrive - University of Texas at El Paso\Documents\Python Projects\Yfinance_Stock_Data_Analysis\yfin_dash_app_api_backend\stock_symbols_list.csv")