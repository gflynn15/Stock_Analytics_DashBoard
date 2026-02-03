# %% [markdown]
# - Creating the second page for the dash application
# - Purpose:
#     - To review business sector trends utilizing the list of companies from the SP500
#     - Create a correlation heat map to see how other companies interact with each other
#     - Incorporate macro economic data to see how they impact company stock price

# %% [markdown]
# - Importing all necessary libraries

# %%
import numpy as np
import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table
from flask_caching import Cache
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import openpyxl
import threading
import sys
import os
# This finds the directory one level up from where this notebook is located
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Add that parent directory to the system path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from app_functions import make_plot
from app_functions import price_card_info
from app_functions import make_card

# %% [markdown]
# - Developing the second page pf the financial stock analytics dash board
# - Steps to follow below
#     1. Build the layout of the dashboard
#         - 5 panel dashboard
#         - Distribution of the SP500 by the GISC Sector
#             - Can be utilized to understand what market sectors are driving change within the S500
#                 - Pie chart
#         - Average price change by sector to measure the performance of the entire sector
#     2. Build the callbacks for the application
#         - First callback will be to build the pie chart of the GISC sector code to understand the distribution of the SP 500

# %%
##Creating the period and interval list for the drop down menus
interval = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
period = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
##Creating list of major market indeces
market_indeces = ["^GSPC","^DJI","^IXIC"]
##Creating list for commodities
###METALS
metals = ["GC=F","SI=F","PL=F","PA=F"]
commodity_names_m = {
        "GC=F": "Gold",
        "SI=F": "Silver",
        "PL=F": "Platinum",
        "PA=F": "Palladium"
    }
###ENERGY
energy = ["CL=F","BZ=F","NG=F","HO=F","RB=F"]
commodity_names_e = {
    "CL=F":"CRUDE OIL",
    "BZ=F":"BRENT CRUDE",
    "NG=F":"NATURAL GAS",
    "HO=F":"HEATING OIL",
    "RB=F":"RBOB GASOLINE"
}
###AGRICULTURE
ag = ["ZC=F","ZW=F","ZS=F","KC=F","LE=F","HE=F","SB=F"]
commodity_names_ag = {
    "ZC=F":"CORN",
    "ZW=F":"WHEAT",
    "ZS=F":"SOY BEANS",
    "KC=F":"COFFEE",
    "HE=F":"LEAN HOGS",
    "SB=F":"SUGAR",
    "LE=F":"LIVE CATTLE"
}
# %%
##utilizing thread locking to prevent multiple calls happening at once
thread_lock = threading.Lock()
with thread_lock:
    stock = yf.Ticker("^GSPC")
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
            "color":"black"
        },
        style_header={
            "backgroundColor": "black",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "left",
        },
    )

# %% [markdown]
# - Building the application layout for page 2

# %%
##Establishign the application variable
dash.register_page(__name__, name="Market_Review", path="/", order=1)
load_figure_template('simplex')
## Designing the application layout below
layout = dbc.Container([
    ## Title of Dashboard
    dbc.Row([
        dbc.Col(html.H1("MARKET REVIEW", style={"textAlign":"center", "fontWeight":"bold"}), width=12)
    ], className="mb-4 mt-2"),

    ## Global Filters
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Period:")),
            dcc.Dropdown(period, "1y", id="index_period", clearable=False)
        ], xs=6, md=3, lg=2),
        dbc.Col([
            html.Label(html.B("Select Interval")),
            dcc.Dropdown(interval, "1d", id="index_interval", clearable=False)
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
    
    ## Market news table
    dbc.Row([
        dbc.Col(html.H1("RECENT NEWS", style={"textAlign":"center","fontWeight":"bold"}), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(market_news), width=12)
    ], className="mb-5"),
    
    ## METALS SECTION
    dbc.Row([
        dbc.Col(html.H1("METAL COMMODITIES TRENDS", style={"textAlign":"center","fontWeight":"bold"}), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Period")),
            dcc.Dropdown(period, "1y", id="commodity_period_m", clearable=False)
        ], xs=6, md=2),
        dbc.Col([
            html.Label(html.B("Select Interval")),
            dcc.Dropdown(interval, "1d", id="commodity_interval_m", clearable=False)
        ], xs=6, md=2),
    ], className="mb-3 g-3"),
    
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Metal")),
            dcc.Dropdown(metals, "SI=F", id="metal_1", clearable=False),
            html.Div(id="metal_1_price")
        ], xs=12, md=4),
        dbc.Col([
            html.Label(html.B("Select Metal")),
            dcc.Dropdown(metals, "GC=F", id="metal_2", clearable=False),
            html.Div(id="metal_2_price")
        ], xs=12, md=4),
        dbc.Col([
            html.Label(html.B("Select Metal")),
            dcc.Dropdown(metals, "PL=F", id="metal_3", clearable=False),
            html.Div(id="metal_3_price")
        ], xs=12, md=4)
    ], className="g-3"),
    
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="metal_1_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="metal_2_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="metal_3_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4)
    ], className="mb-5 g-3"),

    ### ENERGY SECTION
    dbc.Row([
        dbc.Col(html.H1("ENERGY COMMODITIES TRENDS", style={"textAlign":"center","fontWeight":"bold"}), width=12)
    ], className="mt-5 mb-3"),
    
    ## Energy Filters
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Period")),
            dcc.Dropdown(period, "1y", id="commodity_period_e", clearable=False)
        ], xs=6, md=3, lg=2),
        dbc.Col([
            html.Label(html.B("Select Interval")),
            dcc.Dropdown(interval, "1d", id="commodity_interval_e", clearable=False)
        ], xs=6, md=3, lg=2),
    ], className="mb-3 g-3"),
    
    ## Energy Selectors and Price Cards
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Energy")),
            dcc.Dropdown(energy, "CL=F", id="energy_1", clearable=False),
            dcc.Loading(html.Div(id="energy_1_price"), type="circle")
        ], xs=12, md=4),
        dbc.Col([
            html.Label(html.B("Select Energy")),
            dcc.Dropdown(energy, "NG=F", id="energy_2", clearable=False),
            dcc.Loading(html.Div(id="energy_2_price"), type="circle")
        ], xs=12, md=4),
        dbc.Col([
            html.Label(html.B("Select Energy")),
            dcc.Dropdown(energy, "HO=F", id="energy_3", clearable=False),
            dcc.Loading(html.Div(id="energy_3_price"), type="circle")
        ], xs=12, md=4)
    ], className="g-3"),
    
    ## Energy Trend Charts
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="energy_1_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="energy_2_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="energy_3_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4)
    ], className="mb-5 g-3"),

    ### AGRICULTURE SECTION
    dbc.Row([
        dbc.Col(html.H1("AGRICULTURE COMMODITIES TRENDS", style={"textAlign":"center","fontWeight":"bold"}), width=12)
    ], className="mt-5 mb-3"),
    
    ## Ag Filters
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Period")),
            dcc.Dropdown(period, "1y", id="commodity_period_a", clearable=False)
        ], xs=6, md=3, lg=2),
        dbc.Col([
            html.Label(html.B("Select Interval")),
            dcc.Dropdown(interval, "1d", id="commodity_interval_a", clearable=False)
        ], xs=6, md=3, lg=2),
    ], className="mb-3 g-3"),
    
    ## Ag Selectors and Price Cards
    dbc.Row([
        dbc.Col([
            html.Label(html.B("Select Agriculture Commodity")),
            dcc.Dropdown(ag, "ZC=F", id="ag_1", clearable=False),
            dcc.Loading(html.Div(id="ag_1_price"), type="circle")
        ], xs=12, md=4),
        dbc.Col([
            html.Label(html.B("Select Agriculture Commodity")),
            dcc.Dropdown(ag, "ZW=F", id="ag_2", clearable=False),
            dcc.Loading(html.Div(id="ag_2_price"), type="circle")
        ], xs=12, md=4),
        dbc.Col([
            html.Label(html.B("Select Agriculture Commodity")),
            dcc.Dropdown(ag, "LE=F", id="ag_3", clearable=False),
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
    with thread_lock:
        # market_indeces = ["^GSPC", "^DJI", "^IXIC"]
        index_data = yf.download(tickers=market_indeces, period=period, interval=interval, threads=True, auto_adjust=True)
    
    # Use .copy() to prevent SettingWithCopy errors
    Closing_prices = index_data["Close"].copy()
    ##Creating Price Change Variables From Previous Data Point
    ##Price Change Function
    sp500_change = price_card_info(Closing_prices, "^GSPC")
    dow_change = price_card_info(Closing_prices, "^DJI")
    nas_change = price_card_info(Closing_prices, "^IXIC")
    ##Creating Cards
    sp500_card = make_card("SP500 PRICE CHANGE",sp500_change[0], "SP500 PREVIOUS PRICE", sp500_change[1])
    dow_card = make_card("DOW PRICE CHANGE", dow_change[0], "DOW PREVIOUS PRICE", dow_change[1])
    nas_card = make_card("NASDAQ PRICE CHANGE", nas_change[0], "NAS PREVIOUS PRICE", nas_change[1])
        
    # Calculate MAs
    for col in market_indeces:
        Closing_prices[f"{col}_30_MA"] = Closing_prices[col].rolling(window=30).mean()
        Closing_prices[f"{col}_50_MA"] = Closing_prices[col].rolling(window=50).mean()
        Closing_prices[f"{col}_200_MA"] = Closing_prices[col].rolling(window=200).mean()

    closing_prices_nona = Closing_prices[pd.notna(Closing_prices)]
    # Create the figures using the actual string keys
    sp500_fig = make_plot(closing_prices_nona,"^GSPC", "S&P 500 Price Trend")
    dowjones_fig = make_plot(closing_prices_nona,"^DJI", "Dow Jones Industrial Average Price Trend")
    nasdaq_fig = make_plot(closing_prices_nona,"^IXIC", "Nasdaq Composite Price Trend")
    
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
    with thread_lock:
        metals_data = yf.download(tickers=[metal_1, metal_2, metal_3], period=period, interval=interval, threads=True, auto_adjust=True)
    
    metals_closing = metals_data["Close"].copy()
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
    metal_1_fig = make_plot(metals_closing_nona,metal_1, f"{commodity_names_m.get(metal_1, metal_1)} Price Trend")
    metal_2_fig = make_plot(metals_closing_nona,metal_2, f"{commodity_names_m.get(metal_2, metal_2)} Price Trend")
    metal_3_fig = make_plot(metals_closing_nona,metal_3, f"{commodity_names_m.get(metal_3, metal_3)} Price Trend")
    
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
    with thread_lock:
        energy_data = yf.download(tickers=[energy_1, energy_2, energy_3], period=period, interval=interval, threads=True, auto_adjust=True)
    
    energy_closing = energy_data["Close"].copy()
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
    energy_1_fig = make_plot(energy_closing_nona,energy_1, f"{commodity_names_e.get(energy_1, energy_1)} Price Trend")
    energy_2_fig = make_plot(energy_closing_nona,energy_2, f"{commodity_names_e.get(energy_2, energy_2)} Price Trend")
    energy_3_fig = make_plot(energy_closing_nona,energy_3, f"{commodity_names_e.get(energy_3, energy_3)} Price Trend")
    
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
    with thread_lock:
        ag_data = yf.download(tickers=[ag_1, ag_2, ag_3], period=period, interval=interval, threads=True, auto_adjust=True)
    
    ag_closing = ag_data["Close"].copy()
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
    ag_1_fig = make_plot(ag_closing_nona,ag_1, f"{commodity_names_ag.get(ag_1, ag_1)} Price Trend")
    ag_2_fig = make_plot(ag_closing_nona,ag_2, f"{commodity_names_ag.get(ag_2, ag_2)} Price Trend")
    ag_3_fig = make_plot(ag_closing_nona,ag_3, f"{commodity_names_ag.get(ag_3, ag_3)} Price Trend")
    
    return ag_cards[ag_1], ag_cards[ag_2], ag_cards[ag_3],ag_1_fig, ag_2_fig, ag_3_fig


