# %% [markdown]
# - Importing Libraries

# %%
import pandas as pd
import sqlite3
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import os, sys
from sqlalchemy import create_engine
from sqlalchemy import text
import psycopg2

##Loading in tic symbols from the stock symbols list excel file
symbols_file_path = r"C:\Users\Gabriel Flynn\OneDrive\OneDrive - University of Texas at El Paso\Documents\Python Projects\Yfinance_Stock_Data_Analysis\yfin_dash_app_api_backend\Sp500Symbols.xlsx"
tickers = pd.read_excel(io=symbols_file_path, engine="openpyxl")
##Navigating 1 directory up to get the fred API key in the .env file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Add that parent directory to the system path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
##Importing enviroment variables
load_dotenv()
fred_key = os.getenv("fred_api")
render_url = os.getenv("render_db_url")

# %% [markdown]
# - The purpose of this file will be to run daily on a schedule that allows the sqlite database to be updated with recent stock data
# 

# %%
engine = create_engine("sqlite:///STOCK_DATA_WAREHOUSE.db")
conn=sqlite3.connect('STOCK_DATA_WAREHOUSE.db')

##Creating a analytics 
    ##1. convert stock symbols column into a list to pass into yfinance
stock_symbols = tickers["Symbol"].tolist()
extended_symbols_list = ["^GSPC-SP500","^DJI-DowJones","^IXIC-NasDaq","GC=F-Gold","SI=F-Silver","PL=F-Platinum","PA=F-Palladium","CL=F-CrudeOil","BZ=F-BrentCrudeOil","NG=F-NaturalGas","HO=F-HeatingOil","RB=F-RBOBGasoline","ZC=F-Corn","ZW=F-Wheat","ZS=SoyBeans","KC=F-Coffee","LE=F-LiveCattle","HE=F-LeanHogs","SB=F-Sugar"]
stock_symbols.extend(extended_symbols_list)

pricing_df_list = []
bad_symbols = []
    
for stock in stock_symbols:
    try:        
        # 1. Route the Logic: Excel lookup vs. Manual string
        if "-" in stock:
            # For extended items like "^GSPC-SP500"
            clean_ticker = stock.split("-")[0]
            company_name = stock # Just use the exact string you typed as the name!
        else:
            # For standard items like "AMD" from the Excel file
            clean_ticker = stock
            # Look it up in the dataframe safely
            security_name = tickers.loc[tickers["Symbol"] == clean_ticker, "Security"].iloc[0]
            company_name = f"{clean_ticker}-{security_name}"
            
        # 2. Download the data (Using clean_ticker)
        stock_df = yf.download(tickers=clean_ticker, period="10y", interval="1d", auto_adjust=True)
        
        if stock_df.empty:
            raise ValueError(f"No data returned for {clean_ticker}")
        
        # 3. Apply the formatting
        stock_df.columns = stock_df.columns.get_level_values(level=0)
        stock_df["Company"] = company_name # Use the dynamically generated name
        stock_df[["Close","High","Low","Open"]] = stock_df[["Close","High","Low","Open"]].round(2)
        stock_df.columns = stock_df.columns.str.upper()
        stock_df.index.name = "DATE"
        
        pricing_df_list.append(stock_df)
        print(f"[+] Successfully warehoused: {company_name}")
        
    except Exception as e:
        # Save the stock name AND the specific error message so you can read it later
        bad_symbols.append((stock, str(e)))
        print(f"[-] Failed: {stock} | Error: {e}")

# ... (Your SQL load logic stays exactly the same)

yfin_db_df = pd.concat(pricing_df_list, axis=0)

yfin_db_df.to_sql(name="HISTORICAL_STOCK_PRICES",
             con=engine,
             index=True,
             if_exists='replace')

# Open an explicit connection door to the database
with engine.connect() as conn:
    # Use the connection (conn) instead of the engine
    yfin_table = pd.read_sql('SELECT * FROM HISTORICAL_STOCK_PRICES', con=conn)


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
    macro_table = pd.read_sql('SELECT * FROM MACRO_INDICATOR_VALUES', con=conn)

# %% [markdown]
# - News Upload for page 2 of the stock application
#     - Extract news aticles for all company stock indicators in stocks list

# %%
import warnings
warnings.filterwarnings("ignore")
engine = create_engine("sqlite:///STOCK_DATA_WAREHOUSE.db")
conn=sqlite3.connect('STOCK_DATA_WAREHOUSE.db')

##utilizing thread locking to prevent multiple calls happening at once
def news_extraction_function(asset): 
    thread_lock = threading.Lock()
    with thread_lock:
        stock = yf.Ticker(asset)
    news = stock.get_news(count=10, tab="news", proxy=None)
    articles = []
    for arts in range(len(news)):
        keys = ["title","summary","pubDate","clickThroughUrl"]
        news_df = pd.DataFrame(news[arts]["content"])
        articles.append(news_df.loc["url",:])
    articles_df = pd.DataFrame(articles, columns=keys)
    articles_df.columns = articles_df.columns.str.upper()
    articles_df = articles_df[["PUBDATE","TITLE","SUMMARY","CLICKTHROUGHURL"]]
    articles_df["PUBDATE"] = pd.to_datetime(articles_df["PUBDATE"]).dt.date
    articles_df["CLICKTHROUGHURL"] = articles_df["CLICKTHROUGHURL"].apply(
        lambda x: f"[VIEW ARTICLE]({x})" if x else "Sorry we could not find your link. Visit Yahoo finance to find the article"
    )
    articles_df["COMPANY"] = asset
    articles_df.reset_index(drop=True, inplace=True)
    return articles_df

equity_list = []
for equity in stock_symbols:
    if "-" in equity:
        equity_clean = equity.split("-")[0]
    else:
        equity_clean = equity
    try:
        equity_df = news_extraction_function(equity_clean)
        equity_list.append(equity_df)
    except Exception as e:
        print(f"Failed to Upload {e}")

equities_df = pd.concat(equity_list, axis=0, ignore_index=True)

equities_df.to_sql(
                name="STOCK_NEWS_TABLE",
                con=engine,
                index=False,
                if_exists='replace')

with engine.connect() as conn:
    news_table = pd.read_sql("SELECT * FROM STOCK_NEWS_TABLE", con=conn)

# %%
engine = create_engine("sqlite:///STOCK_DATA_WAREHOUSE.db")
conn=sqlite3.connect('STOCK_DATA_WAREHOUSE.db')

companies = tickers["Symbol"].tolist()
    
#ownership_info = pd.DataFrame(comp_info["companyOfficers"]).columns.str.upper()
company_info_columns = ["industry", "sector", "fullTimeEmployees","auditRisk","boardRisk","compensationRisk","shareHolderRightsRisk",
                        "overallRisk","payoutRatio","beta","trailingPE","forwardPE","marketCap","nonDilutedMarketCap","priceToSalesTrailing12Months",
                        "trailingAnnualDividendRate","trailingAnnualDividendYield","enterpriseValue","profitMargins","floatShares","sharesOutstanding",
                        "sharesShort","sharesShortPriorMonth","sharesShortPreviousMonthDate","shortRatio","shortPercentOfFloat","heldPercentInsiders",
                        "heldPercentInsiders","bookValue","priceToBook","earningsQuarterlyGrowth",
                        "trailingEps","forwardEps","enterpriseToEbitda","totalCash","totalCashPerShare","ebitda",
                        "totalDebt","quickRatio","currentRatio","totalRevenue","debtToEquity","returnOnAssets","returnOnEquity",
                        "grossProfits","freeCashflow","operatingCashflow","earningsGrowth","revenueGrowth",
                        "grossMargins","ebitdaMargins","operatingMargins","trailingPegRatio","longBusinessSummary"]#, "earningsTimestamp"]

info_list_dfs = []
fails = []
for comp in companies:
    ticker = yf.Ticker(ticker=comp)
    ticker_info = ticker.get_info()
    values = {}
    for x in company_info_columns:
        try:
            info_value = ticker_info[x]
            values[x] = info_value
        except Exception as e:
            fails.append(e)
    info_df = pd.Series(data=values, index=values.keys(), name=comp)
    info_list_dfs.append(info_df)

fundamental_df = pd.concat(info_list_dfs, axis=1, ignore_index=False)
fundamental_df_flip = fundamental_df.transpose()
#fundamental_df_flip["EARNINGS_REPORT_DATE"] = pd.to_datetime(fundamental_df_flip[""], unit="s").date()
fundamental_df_flip.to_sql(
    name="FUNDAMENTAL_DATA",
    con=conn,
    index=True,
    if_exists="replace"
)
#ceo_df = pd.concat(ceo_list_dfs, axis=0, ignore_index=True)

###=====================================Grouping of the above list===========================================================================###
#1. Identity & Corporate Governance
    ###Long business summary, sector/industry, full time employees, audit risk, board risk, compensation, share holders right risk, overall risk
#2. Size and Valuation
    ###Market Cap, Non Diluted Market Cap, Enterprise Value, Trailing PE, Forward PE, Trailing Peg Ratio, Price to Sales Trailing 12 Months, Price to Book, Enterprise to Book, Enterprise to Ebitda
#3. Financial Health & Dividends
    ###Total Cash, Total Cash Per Share, Total Debt, Debt to Equity, Current Ratio, Quick Ratio, Payout Ratio, Trailing Annual Dividend Ratio & Yield
#4. The Profit Engine
    ###Total Revenue, Gross Profits, Ebitda, Free Cash Flow, Operating Cash Flow, Trailing EPS, Forward, EPS, Profit Margins, Gross MArgins, Operating Margins, Return on Assets,
    ###Return on Equity, Revenue Growth, Earnings Growth, Earnings Quarterly Growth
#5. Market Mechanics and Sentiment
    ###Beta, Shares Outstanding, Float Shares, Held Percent Insiders & institutions, Shares Short, Shares Short Prior Month, Shres Short Previous Month Date
    ###Short Percent Float, Short Ratio,      

# %% [markdown]
# - Cloud Migration script

# %%
# 1. The Cloud Destination (Paste your Render External URL here)
render_url = fr"{render_url}"

cloud_engine = create_engine(render_url)

# 2. The Local Source (Your current SQLite database)
local_conn = sqlite3.connect('STOCK_DATA_WAREHOUSE.db')

# 3. Extract the data from local
print("Extracting local data...")
local_df = pd.read_sql("SELECT * FROM FUNDAMENTAL_DATA", con=local_conn)

# 4. Load the data to Render Postgres
print("Pushing data to the cloud...")
local_df.to_sql(
    name='FUNDAMENTAL_DATA', 
    con=cloud_engine, 
    if_exists='replace', 
    index=False,
    # You can re-paste your schema_mapping dictionary here from earlier 
    # if you want to strictly enforce the dtypes again.
)

local_conn.close()
print("Migration Complete: Historical data is now live on Render.")


# %%



