# %% [markdown]
# - Creating the second page for the dash application
# - Purpose:
#     - To review business sector trends utilizing the list of companies from the SP500
#     - Create a correlation heat map to see how other companies interact with each other
#     - Incorporate macro economic data to see how they impact company stock price

# %% [markdown]
# - Importing all necessary libraries

# %%
from app_init import cache
import numpy as np
import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import sys
import os
import statsmodels
import statsmodels.api as sm
from flask import Flask
from flask_caching import Cache
# This finds the directory one level up from where this notebook is located
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Add that parent directory to the system path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from app_functions import make_plot
from app_functions import price_card_info
from app_functions import make_card
from sqlalchemy import create_engine
from sqlalchemy import text
import sqlite3
# This finds the directory one level up from where this notebook is located
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Add that parent directory to the system path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from app_functions import make_plot
from app_functions import price_card_info
from app_functions import make_card
from app_functions import data_query
# %% [markdown]
# - Creating the SQLite DB Connection strings
from dotenv import load_dotenv
                                        ###Importing the databse url###
load_dotenv()
render_url = os.getenv("render_db_url")

engine = create_engine(render_url)

# %% [markdown]
# %%    
# %% [markdown]
# - Developing the second page pf the financial stock analytics dash board
# - Steps to follow below
#     1. Build the layout of the dashboard
#         - 5 panel dashboard
#         - Distribution of the SP500 by the GISC Sector
#             - Can be utilized to understand what market sectors are driving change within the SP500
#                 - Pie chart
#         - Average price change by sector to measure the performance of the entire sector
#     2. Build the callbacks for the application
#         - First callback will be to build the pie chart of the GISC sector code to understand the distribution of the SP 500

# %%
        ###Drop downs allowing the data to be filtered by period and interval settings in the drop down menus###
period = ["W","M","3M","1Y", "2Y","3Y","5Y","YTD","MAX"]
interval = ["D", "W", "M", "Q", "Y"]

with engine.connect() as conn:
    stock_symbols = pd.read_sql(text('SELECT DISTINCT "COMPANY" FROM "HISTORICAL_STOCK_PRICES"'), con=conn)
    stock_symbols_list = stock_symbols["COMPANY"].tolist()

# %% [markdown]
# - Creating equity dictionaries to add into graphics

# %%
energy = ["CL=F-CrudeOil","BZ=F-BrentCrudeOil","NG=F-NaturalGas","HO=F-HeatingOil","RB=F-RBOBGasoline"]
metals = ["SI=F-Silver","PL=F-Platinum","PA=F-Palladium","GC=F-Gold"]
ag = ["ZC=F-Corn","ZW=F-Wheat","KC=F-Coffee","LE=F-LiveCattle","HE=F-LeanHogs","SB=F-Sugar"]

# %% [markdown]
# Category,Commodity,yfinance Ticker
# 
# Energy,Crude Oil (WTI),CL=F
# ,Brent Crude Oil,BZ=F
# ,Natural Gas,NG=F
# ,Heating Oil,HO=F
# ,RBOB Gasoline,RB=F
# 
# Metals,Gold,GC=F
# ,Silver,SI=F
# ,Copper,HG=F
# ,Platinum,PL=F
# ,Palladium,PA=F
# 
# Agriculture,Corn,ZC=F
# ,Wheat,ZW=F
# ,Soybeans,ZS=F
# ,Sugar,SB=F
# ,Coffee,KC=F
# ,Live Cattle,LE=F
# ,Lean Hogs,HE=F

# %% [markdown]
# - Gathering the market news for the market table

# %%
with engine.connect() as conn:
    articles_df = pd.read_sql(text("""SELECT * FROM "STOCK_NEWS_TABLE" WHERE "COMPANY" IN ('^GSPC','^DJI','^IXIC')"""), con=conn)
    #articles_df.drop(columns=["index"], inplace=True)
    articles_df["PUBDATE"] = pd.to_datetime(articles_df["PUBDATE"]).dt.date
    articles_df.sort_values(by="PUBDATE",ascending=False, inplace=True)
    articles_df = articles_df.iloc[:15,:]

##Creating the dash table
market_news = dash_table.DataTable(
        columns=[
        {"name": "PUBDATE", "id": "PUBDATE"},
        {"name": "TITLE", "id": "TITLE"},
        {"name": "SUMMARY", "id": "SUMMARY"},
        {"name": "LINK", "id": "CLICKTHROUGHURL", "presentation": "markdown"}
        ],
        data=articles_df.to_dict("records"),
        cell_selectable=True,
        page_size=5,
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "border": "1px solid #ccc"
        },
        style_data={
            "textAlign": "left",
            "whiteSpace": "normal",
            "color":"white",
            "backgroundColor": "#585859",
            "fontSize":25,
            "fontFamily":"Inter"
        },
        style_header={
            "backgroundColor": "eb4904d4",
            "color": "black",
            "fontWeight": "bold",
            "textAlign": "center",
            "fontSize":30
        },
    )

# %% [markdown]
# - Building the application layout for page 2

# %%
##Establishign the application variable
dash.register_page(__name__, name="MARKET PULSE", path="/MARKET_PULSE_1_1", order=3, external_stylesheets=[dbc.themes.CYBORG, "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"])
load_figure_template('cyborg')

## 1. Initialize Flask Server FIRST
#server = Flask(__name__)

##Creating a cache file to store the data
#cache = Cache(server, config={
#    'CACHE_TYPE': 'FileSystemCache',
#    'CACHE_DIR': 'Market_Review_cache_file',
#    'CACHE_DEFAULT_TIMEOUT': 300,
#    'CACHE_THRESHOLD': 500
#

## Designing the application layout below
layout = dbc.Container([
    ## Title of Dashboard
    dbc.Row([
        dbc.Col(html.H1("MARKET PULSE", style={
                'textAlign': 'center', 
                'fontFamily': 'Inter', 
                'fontWeight': '900',
                'background': '-webkit-linear-gradient(45deg, #FF416C, #FF4B2B)',
                '-webkit-background-clip': 'text',
                '-webkit-text-fill-color': 'transparent',
                'fontSize': '5rem',
                'paddingBottom': '20px',
                'paddingTop': '20px'
            }),
            width=12,
            className="d-flex justify-content-center align-items-center")
    ], className="animate__animated animate__fadeInDown macro-health-header"),

    ## Global Filters
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Period:"), style={"fontSize":35}),
            dcc.Dropdown(period, "5Y", id="index_period", clearable=False, style={"fontSize":25})
        ], xs=6, md=3, lg=2),
        dbc.Col([
            html.Label(html.B("Select Interval"), style={"fontSize":35}),
            dcc.Dropdown(interval, "D", id="index_interval", clearable=False, style={"fontSize":25})
        ], xs=6, md=3, lg=2),
    ], className="mb-4 g-3"), # g-3 adds "gutter" spacing between columns

    ## Scorecards Row (Stacks 1x1 on mobile, 3x1 on laptop)
    dbc.Row([
        dbc.Col(html.Div(id='sp500_change'), xs=12, md=4),
        dbc.Col(html.Div(id='dow_change'), xs=12, md=4),
        dbc.Col(html.Div(id='nas_change'), xs=12, md=4),
    ], className="mb-3 g-3"),
    
    ## Indices Graphs Row
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="sp500", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="dowjones", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="nasdaq", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4)
    ], className="mb-5 g-3"),
    
    ###=========================================Market news table===================================================###
    dbc.Row([
        dbc.Col([html.H1(
                    "MARKET DIGEST", 
                    style={"textAlign":"center","fontFamily":
                            "Inter","fontWeight":"900",
                            "color":"white"}
                        )
                ],width=12)
            ],
            className="market-news-header animate__animated animate__fadeInLeft"),
    
    dbc.Row([
        dbc.Col(html.Div(market_news), width=12)
    ], className="mb-5"),
    
    ###=============================================METALS SECTION==================================================###
    #metals = ["SI=F-Silver","PL=F-Platinum","PA=F-Palladium","GC=F-Gold"]
    dbc.Row([
        dbc.Col(
            [
            html.H1("METAL COMMODITIES TRENDS", 
            style={"textAlign":"center","fontFamily":"Inter","fontWeight":"900","color":"white"})
            ],width=12)
            ],style={"textAlign":"center","fontWeight":"bold"}, className="commodities-header animate__animated animate__fadeInRight"
            ),
    
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Period")),
            dcc.Dropdown(period, "5Y", id="commodity_period_m", clearable=False, style={"fontSize":25})
        ], xs=6, md=2),
        dbc.Col([
            html.Label(html.B("Select Interval")),
            dcc.Dropdown(interval, "D", id="commodity_interval_m", clearable=False, style={"fontSize":25})
        ], xs=6, md=2),
    ], className="mb-3 g-3"),
    
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Metal")),
            dcc.Dropdown(metals, "SI=F-Silver", id="metal_1", clearable=False, style={"fontSize":25}),
            html.Div(id="metal_1_price")
        ], xs=12, md=4),
        dbc.Col([
            html.Label(html.B("Select Metal")),
            dcc.Dropdown(metals, "GC=F-Gold", id="metal_2", clearable=False, style={"fontSize":25}),
            html.Div(id="metal_2_price")
        ], xs=12, md=4),
        dbc.Col([
            html.Label(html.B("Select Metal")),
            dcc.Dropdown(metals, "PL=F-Platinum", id="metal_3", clearable=False, style={"fontSize":25}),
            html.Div(id="metal_3_price")
        ], xs=12, md=4)
    ], className="g-3"),
    
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="metal_1_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="metal_2_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="metal_3_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4)
    ], className="mb-5 g-3"),

    ###=============================================ENERGY SECTION======================================================###
    #energy = ["CL=F=CrudeOil","BZ=F-BrentCrudeOil","NG=F-NaturalGas","Ho=F-HeatingOil","RB=F-RBOBGasoline"]
    dbc.Row([
        dbc.Col([
            html.H1("ENERGY COMMODITIES TRENDS", 
            style={"textAlign":"center","fontFamily":"Inter","fontWeight":"900","color":"white"})
            ],width=12)
            ], 
        className="energy-header animate__animated animate__fadeInLeft", 
        style={"textAlign":"center","fontWeight":"bold"}
            ),
    
    ###=========================================Energy Filters============================================###
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Period")),
            dcc.Dropdown(period, "5Y", id="commodity_period_e", clearable=False, style={"fontSize":25})
        ], xs=6, md=3, lg=2),
        dbc.Col([
            html.Label(html.B("Select Interval")),
            dcc.Dropdown(interval, "D", id="commodity_interval_e", clearable=False, style={"fontSize":25})
        ], xs=6, md=3, lg=2),
    ], className="mb-3 g-3"),
    
    
    ###=====================================Energy Selectors and Price Cards========================================###
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Energy")),
            dcc.Dropdown(energy, "CL=F-CrudeOil", id="energy_1", clearable=False, style={"fontSize":25}),
            dcc.Loading(html.Div(id="energy_1_price"), type="circle")
        ], xs=12, md=4),
        dbc.Col([
            html.Label(html.B("Select Energy")),
            dcc.Dropdown(energy, "NG=F-NaturalGas", id="energy_2", clearable=False, style={"fontSize":25}),
            dcc.Loading(html.Div(id="energy_2_price"), type="circle")
        ], xs=12, md=4),
        dbc.Col([
            html.Label(html.B("Select Energy")),
            dcc.Dropdown(energy, "HO=F-HeatingOil", id="energy_3", clearable=False, style={"fontSize":25}),
            dcc.Loading(html.Div(id="energy_3_price"), type="circle")
        ], xs=12, md=4)
    ], className="g-3"),
    
    ## Energy Trend Charts
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="energy_1_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="energy_2_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="energy_3_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4)
    ], className="mb-5 g-3"),

    ###=============================================AGRICULTURE SECTION=========================================###
    #ag = ["ZC=F-Corn","ZW=F-Wheat","KC=F-Coffee","LE=F-LiveCattle","HE=F-LeanHogs","SB=F-Sugar"]
    dbc.Row([
        dbc.Col(
            [
            html.H1("AGRICULTURE COMMODITIES TRENDS", 
            style={"textAlign":"center","fontFamily":"Inter","fontWeight":"900","color":"white"})
            ], 
            width=12), 
            ], 
            className="agg-header animate__animated animate__fadeInRight", 
            style={"textAlign":"center","fontWeight":"bold"}
            ),
    
    ## Ag Filters
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Period")),
            dcc.Dropdown(period, "5Y", id="commodity_period_a", clearable=False, style={"fontSize":25})
        ], xs=6, md=3, lg=2),
        dbc.Col([
            html.Label(html.B("Select Interval")),
            dcc.Dropdown(interval, "D", id="commodity_interval_a", clearable=False, style={"fontSize":25})
        ], xs=6, md=3, lg=2),
    ], className="mb-3 g-3"),
    
    ## Ag Selectors and Price Cards
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Agriculture Commodity")),
            dcc.Dropdown(ag, "ZC=F-Corn", id="ag_1", clearable=False, style={"fontSize":25}),
            dcc.Loading(html.Div(id="ag_1_price"), type="circle")
        ], xs=12, md=4),
        dbc.Col([
            html.Label(html.B("Select Agriculture Commodity")),
            dcc.Dropdown(ag, "ZW=F-Wheat", id="ag_2", clearable=False, style={"fontSize":25}),
            dcc.Loading(html.Div(id="ag_2_price"), type="circle")
        ], xs=12, md=4),
        dbc.Col([
            html.Label(html.B("Select Agriculture Commodity")),
            dcc.Dropdown(ag, "LE=F-LiveCattle", id="ag_3", clearable=False, style={"fontSize":25}),
            dcc.Loading(html.Div(id="ag_3_price"), type="circle")
        ], xs=12, md=4)
    ], className="g-3"),
    
    ## Ag Trend Charts
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="ag_1_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="ag_2_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="ag_3_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4)
    ], className="mb-5 g-3")
    
], fluid=True)

# %% [markdown]
# - Building out the callbacks for the application
#     1. indeces graph callbacks
#     2. metal commodity graphs callbacks
#     3. energy commodity graphs callbacks

# %% [markdown]
# 1. Indeces graphs callbacks

# %%
##Indeces Graphs callback
@callback(
    Output("sp500_change", "children"),
    Output("dow_change", "children"),
    Output("nas_change","children"),
    Output("sp500", "figure"),
    Output("dowjones", "figure"),
    Output("nasdaq", "figure"),
    Input("index_period", "value"),
    Input("index_interval", "value")
)
def index_trend_chart(period, interval):
    indices = [ '^GSPC-SP500','^DJI-DowJones','^IXIC-NasDaq']
    closing_prices = data_query(indices, period, interval).dropna(axis=0)
    ##Creating Price Change Variables From Previous Data Point
    ##Price Change Function
    sp500_change = price_card_info(closing_prices, "^GSPC-SP500")
    dow_change = price_card_info(closing_prices, "^DJI-DowJones")
    nas_change = price_card_info(closing_prices, "^IXIC-NasDaq")
    ##Creating Cards
    sp500_card = make_card("SP500 PRICE CHANGE",sp500_change[0], "SP500 PREVIOUS PRICE", sp500_change[1])
    dow_card = make_card("DOW PRICE CHANGE", dow_change[0], "DOW PREVIOUS PRICE", dow_change[1])
    nas_card = make_card("NASDAQ PRICE CHANGE", nas_change[0], "NAS PREVIOUS PRICE", nas_change[1])
        
    # Calculate MAs
    for col in closing_prices.columns:
        closing_prices[f"{col}_30_MA"] = closing_prices[col].rolling(window=30).mean()
        closing_prices[f"{col}_50_MA"] = closing_prices[col].rolling(window=50).mean()
        closing_prices[f"{col}_200_MA"] = closing_prices[col].rolling(window=200).mean()

    closing_prices_nona = closing_prices.dropna(axis=0)
    # Create the figures using the actual string keys
    sp500_fig = make_plot(closing_prices_nona,"^GSPC-SP500", "S&P 500 Price Trend")
    dowjones_fig = make_plot(closing_prices_nona,"^DJI-DowJones", "Dow Jones Industrial Average Price Trend")
    nasdaq_fig = make_plot(closing_prices_nona,"^IXIC-NasDaq", "Nasdaq Composite Price Trend")
    
    return sp500_card,dow_card,nas_card,sp500_fig, dowjones_fig, nasdaq_fig

# %% [markdown]
# 2. Metal commodity graphs call backs

# %%
@callback(
    Output("metal_1_price", "children"),
    Output("metal_2_price", "children"),
    Output("metal_3_price", "children"),
    Output("metal_1_graph", "figure"),
    Output("metal_2_graph", "figure"),
    Output("metal_3_graph", "figure"),
    Input("commodity_period_m", "value"),
    Input("commodity_interval_m", "value"),
    Input("metal_1", "value"),
    Input("metal_2", "value"),
    Input("metal_3", "value")
)
def metals_trend_charts(period, interval, metal_1, metal_2, metal_3):
    metals_list = [metal_1, metal_2, metal_3]
    metals_closing = data_query(metals_list, period, interval).dropna(axis=0)
    ##Getting Price Changes
    metals_pricing_list = {}
    for x in metals_closing.columns:
        metals_prices = price_card_info(metals_closing,x)
        metals_pricing_list[x] = metals_prices
    metal_cards = {}
    for key, val in metals_pricing_list.items():
        metal_cards[key] = make_card(f"{key} PRICE CHANGE",val[0], f"{key} PREVIOUS PRICE", val[1])
    
    for col in metals_closing.columns:
        metals_closing[f"{col}_30_MA"] = metals_closing[col].rolling(window=30).mean()
        metals_closing[f"{col}_50_MA"] = metals_closing[col].rolling(window=50).mean()
        metals_closing[f"{col}_200_MA"] = metals_closing[col].rolling(window=200).mean()
    
    metals_closing_nona = metals_closing[pd.notna(metals_closing)]
    metal_1_fig = make_plot(metals_closing_nona,metal_1, f"{metal_1.split('-')[0]} Price Trend")
    metal_2_fig = make_plot(metals_closing_nona,metal_2, f"{metal_2.split('-')[0]} Price Trend")
    metal_3_fig = make_plot(metals_closing_nona,metal_3, f"{metal_3.split('-')[0]} Price Trend")
    
    return metal_cards[metal_1], metal_cards[metal_2], metal_cards[metal_3],metal_1_fig, metal_2_fig, metal_3_fig

# %% [markdown]
# 3. Energy Commoditiy Graphs

# %%
@callback(
    Output("energy_1_price", "children"),
    Output("energy_2_price", "children"),
    Output("energy_3_price", "children"),
    Output("energy_1_graph", "figure"),
    Output("energy_2_graph", "figure"),
    Output("energy_3_graph", "figure"),
    Input("commodity_period_e", "value"),
    Input("commodity_interval_e", "value"),
    Input("energy_1", "value"),
    Input("energy_2", "value"),
    Input("energy_3", "value")
)
def energy_trend_charts(period, interval, energy_1, energy_2, energy_3):
    energy_list = [energy_1, energy_2, energy_3]
    energy_closing = data_query(energy_list, period, interval).dropna(axis=0)
    ##Getting Price Changes
    energy_pricing_list = {}
    for x in energy_closing.columns:
        energy_prices = price_card_info(energy_closing,x)
        energy_pricing_list[x] = energy_prices
    energy_cards = {}
    for key, val in energy_pricing_list.items():
        energy_cards[key] = make_card(f"{key} PRICE CHANGE",val[0], f"{key} PREVIOUS PRICE", val[1])
    
    for col in energy_closing.columns:
        energy_closing[f"{col}_30_MA"] = energy_closing[col].rolling(window=30).mean()
        energy_closing[f"{col}_50_MA"] = energy_closing[col].rolling(window=50).mean()
        energy_closing[f"{col}_200_MA"] = energy_closing[col].rolling(window=200).mean()
    energy_closing_nona = energy_closing[pd.notna(energy_closing)]
    energy_1_fig = make_plot(energy_closing_nona,energy_1, f"{energy_1.split('-')[0]} Price Trend")
    energy_2_fig = make_plot(energy_closing_nona,energy_2, f"{energy_2.split('-')[0]} Price Trend")
    energy_3_fig = make_plot(energy_closing_nona,energy_3, f"{energy_3.split('-')[0]} Price Trend")
    
    return energy_cards[energy_1], energy_cards[energy_2], energy_cards[energy_3],energy_1_fig, energy_2_fig, energy_3_fig

# %% [markdown]
# 4. Agriculture Commodities callbacks

# %%
@callback(
    Output("ag_1_price", "children"),
    Output("ag_2_price", "children"),
    Output("ag_3_price", "children"),
    Output("ag_1_graph", "figure"),
    Output("ag_2_graph", "figure"),
    Output("ag_3_graph", "figure"),
    Input("commodity_period_a", "value"),
    Input("commodity_interval_a", "value"),
    Input("ag_1", "value"),
    Input("ag_2", "value"),
    Input("ag_3", "value")
)
def ag_trend_charts(period, interval, ag_1, ag_2, ag_3):
    ag_list = [ag_1, ag_2, ag_3]
    ag_closing = data_query(ag_list, period, interval).dropna(axis=0)
    ##Getting Price Changes
    ag_pricing_list = {}
    for x in ag_closing.columns:
        ag_prices = price_card_info(ag_closing,x)
        ag_pricing_list[x] = ag_prices
    ag_cards = {}
    for key, val in ag_pricing_list.items():
        ag_cards[key] = make_card(f"{key} PRICE CHANGE",val[0], f"{key} PREVIOUS PRICE", val[1])
    
    for col in ag_closing.columns:
        ag_closing[f"{col}_30_MA"] = ag_closing[col].rolling(window=30).mean()
        ag_closing[f"{col}_50_MA"] = ag_closing[col].rolling(window=50).mean()
        ag_closing[f"{col}_200_MA"] = ag_closing[col].rolling(window=200).mean()
    ag_closing_nona = ag_closing[pd.notna(ag_closing)]
    ag_1_fig = make_plot(ag_closing_nona,ag_1, f"{ag_1.split('-')[0]} Price Trend")
    ag_2_fig = make_plot(ag_closing_nona,ag_2, f"{ag_2.split('-')[0]} Price Trend")
    ag_3_fig = make_plot(ag_closing_nona,ag_3, f"{ag_3.split('-')[0]} Price Trend")
    
    return ag_cards[ag_1], ag_cards[ag_2], ag_cards[ag_3],ag_1_fig, ag_2_fig, ag_3_fig


