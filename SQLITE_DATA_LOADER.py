# %% [markdown]
# - The purpose of this file will be to run daily on a schedule that allows the sqlite database to be updated with recent stock data
# 

# %%
import pandas as pd
import sqlite3
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import os, sys
from sqlalchemy import create_engine
from sqlalchemy import text

##Loading in tic symbols from the stock symbols list excel file
symbols_file_path = r"C:\Users\Gabriel Flynn\OneDrive\OneDrive - University of Texas at El Paso\Documents\Python Projects\Yfinance_Stock_Data_Analysis\yfin_dash_app_api_backend\Sp500Symbols.xlsx"
tickers = pd.read_excel(io=symbols_file_path, engine="openpyxl")
##Navigating 1 directory up to get the fred API key in the .env file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Add that parent directory to the system path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
##Importing the Fred API key
load_dotenv()

fred_key = os.getenv("fred_api")

engine = create_engine("sqlite:///Closing_Price_db.db")
conn=sqlite3.connect('Closing_Price_db.db')

##Creating a analytics 
    ##1. convert stock symbols column into a list to pass into yfinance
stock_symbols = tickers["Symbol"].tolist()

pricing_df_list = []
bad_symbols = []
try:
    for stock in stock_symbols:
        stock_df = yf.download(tickers=stock, period="10y",interval="1d")
        stock_df.columns = stock_df.columns.get_level_values(level=0)
        stock_df["Company"] = tickers.loc[tickers["Symbol"] == stock, "Symbol"].iloc[0] + '-' + tickers.loc[tickers["Symbol"] == stock, "Security"].iloc[0]
        stock_df[["Close","High","Low","Open"]] = stock_df[["Close","High","Low","Open"]].round(2)
        stock_df.columns = stock_df.columns.str.upper()
        stock_df.index.name = "DATE"
        pricing_df_list.append(stock_df)
        continue
except Exception as e:
    bad_symbols.append(e)

yfin_db_df = pd.concat(pricing_df_list, axis=0)

yfin_db_df.to_sql(name="HISTORICAL_STOCK_PRICES",
             con=engine,
             index=True,
             if_exists='replace')

# Open an explicit connection door to the database
with engine.connect() as conn:
    # Use the connection (conn) instead of the engine
    yfin_table = pd.read_sql(text('SELECT * FROM HISTORICAL_STOCK_PRICES'), con=conn)


# %% [markdown]
# - Creating Data pull for the market leading indicators from the FRED  API

# %%
                                                                ###Created a cariable to execute the fred call###
fred_trigger = Fred(api_key=fred_key)

##Created a leading_indicators dictionary so that I can utilize the key value pairings in the callback function to display the name of the indicator versus it's id
leading_indicators_dict = {"GDP":["GDP","q"],
                      "INDUSTRIAL PRODUCTION":["INDPRO","m"],
                      "RETAIL SALES":["RSAFS","m"],
                      "BUSINESS INVENTORIES":["BUSINV","m"],
                      "UNEMPLOYEMENT":["UNRATE","m"],
                      "TOTAL NONFARM PAYROLLS":["PAYEMS","m"],
                      "INITIAL UNEMPLOYMENT CLAIMS":["ICSA","w"],
                      "CPI":["CPIAUCSL","m"],
                      "PCE":["PCEPI","m"],
                      "FEDERAL FUNDS EFFECTIVE RATE":["FEDFUNDS","m"],
                      "10-YEAR TREASURY YIELD":["DGS10","m"],
                      "2-YEAR TREASURY YIELD":["DGS2","m"],
                      "30-YEAR FIXED MORTAGAGE RATE":["MORTGAGE30US","w"],
                      "M2 MONEY SUPPLY":["M2SL","m"],
                      "RECESSION INDICATOR":["USREC","m"]
                    }


        ###Created function to execute the fred function utilizing the dicitonary to make it easier access the parameters needed for each indicator###
def fred_function(metrics_list):
        try:
            if metrics_list in ["RECESSION INDICATOR","FEDERAL FUNDS EFFECTIVE RATE"]:
                metric = leading_indicators_dict[metrics_list]
                macro_indi = fred_trigger.get_series(series_id=metric[0], frequency=metric[1], units="lin")
            else:
                metric = leading_indicators_dict[metrics_list]
                macro_indi = fred_trigger.get_series(metric[0], frequency=metric[1], units="lin")
                macro_indi.name = metrics_list
        except Exception as e:
            print(f"An error occurred to process {metrics_list}: {e}")
        return macro_indi

fred_indicator_list = list(leading_indicators_dict.keys())

                                ###Creating the lagging indicator table to upload into the sqllite datbase file###
fred_df_list = []
for x in fred_indicator_list:
    fred_series = fred_function(x)
    fred_df = fred_series.to_frame(name="VALUE")
    fred_df["LEADING_INDICATOR"] = x
    fred_df.index.name = "DATE"
    fred_df_list.append(fred_df)
    
fred_df_summary = pd.concat(objs=fred_df_list, axis=0)

                                                ###Uploading the table to the SQLite File###
fred_df_summary.to_sql(name="MACRO_INDICATOR_VALUES",
             con=engine,
             index=True,
             if_exists='replace')

# Open an explicit connection door to the database
with engine.connect() as conn:
    # Use the connection (conn) instead of the engine
    macro_table = pd.read_sql(text('SELECT * FROM MACRO_INDICATOR_VALUES'), con=conn)

macro_table


# %%



