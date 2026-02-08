import numpy as np
import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import os
import sys

# Import Cache from your main app
from Yfinance_Dash_2_5 import cache

# Add parent directory to path to find app_functions
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from app_functions import make_plot
from app_functions import price_card_info
from app_functions import make_card

# --- SETUP PAGE ---
dash.register_page(__name__, name="Market_Review", path="/", order=1)
load_figure_template('simplex')

# --- CONFIG DATA ---
interval = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
period = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
market_indeces = ["^GSPC","^DJI","^IXIC"]

# Commodity mappings
metals = ["GC=F","SI=F","PL=F","PA=F"]
commodity_names_m = {"GC=F": "Gold", "SI=F": "Silver", "PL=F": "Platinum", "PA=F": "Palladium"}

energy = ["CL=F","BZ=F","NG=F","HO=F","RB=F"]
commodity_names_e = {"CL=F":"CRUDE OIL", "BZ=F":"BRENT CRUDE", "NG=F":"NATURAL GAS", "HO=F":"HEATING OIL", "RB=F":"RBOB GASOLINE"}

ag = ["ZC=F","ZW=F","ZS=F","KC=F","LE=F","HE=F","SB=F"]
commodity_names_ag = {"ZC=F":"CORN", "ZW=F":"WHEAT", "ZS=F":"SOY BEANS", "KC=F":"COFFEE", "HE=F":"LEAN HOGS", "SB=F":"SUGAR", "LE=F":"LIVE CATTLE"}


# ==========================================
# CACHED DATA FUNCTIONS
# ==========================================

@cache.memoize(timeout=600) # 10 Minutes
def get_market_data(tickers, period, interval):
    # threads=True is faster for multiple tickers
    df = yf.download(tickers=tickers, period=period, interval=interval, threads=True, auto_adjust=True)
    return df.to_json(date_format='iso')

@cache.memoize(timeout=1800) # 30 Minutes
def get_market_news():
    try:
        stock = yf.Ticker("^GSPC")
        return stock.get_news(count=10, tab="news")
    except Exception:
        return []

# ==========================================
# LAYOUT
# ==========================================
layout = dbc.Container([
    ## Title
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
    ], className="mb-4 g-3"),

    ## Scorecards
    dbc.Row([
        dbc.Col(html.Div(id='sp500_change'), xs=12, md=4),
        dbc.Col(html.Div(id='dow_change'), xs=12, md=4),
        dbc.Col(html.Div(id='nas_change'), xs=12, md=4),
    ], className="mb-3 g-3"),
    
    ## Indices Graphs
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="sp500", style={'minHeight': '400px', 'height': '60vh'}, persistence=True)), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="dowjones", style={'minHeight': '400px', 'height': '60vh'}, persistence=True)), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="nasdaq", style={'minHeight': '400px', 'height': '60vh'}, persistence=True)), xs=12, lg=4)
    ], className="mb-5 g-3"),
    
    ## Market news table (Now loaded via callback to prevent crash)
    dbc.Row([
        dbc.Col(html.H1("RECENT NEWS", style={"textAlign":"center","fontWeight":"bold"}), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Loading(html.Div(id='market_news_table')), width=12)
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
        dbc.Col(dcc.Loading(dcc.Graph(id="metal_1_graph", style={'minHeight': '400px', 'height': '60vh'}, persistence=True)), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="metal_2_graph", style={'minHeight': '400px', 'height': '60vh'}, persistence=True)), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="metal_3_graph", style={'minHeight': '400px', 'height': '60vh'}, persistence=True)), xs=12, lg=4)
    ], className="mb-5 g-3"),

    ### ENERGY SECTION
    dbc.Row([
        dbc.Col(html.H1("ENERGY COMMODITIES TRENDS", style={"textAlign":"center","fontWeight":"bold"}), width=12)
    ], className="mt-5 mb-3"),
    
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
    
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="energy_1_graph", style={'minHeight': '400px', 'height': '60vh'}, persistence=True)), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="energy_2_graph", style={'minHeight': '400px', 'height': '60vh'}, persistence=True)), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="energy_3_graph", style={'minHeight': '400px', 'height': '60vh'}, persistence=True)), xs=12, lg=4)
    ], className="mb-5 g-3"),

    ### AGRICULTURE SECTION
    dbc.Row([
        dbc.Col(html.H1("AGRICULTURE COMMODITIES TRENDS", style={"textAlign":"center","fontWeight":"bold"}), width=12)
    ], className="mt-5 mb-3"),
    
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
    
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="ag_1_graph", style={'minHeight': '400px', 'height': '60vh'}, persistence=True)), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="ag_2_graph", style={'minHeight': '400px', 'height': '60vh'}, persistence=True)), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="ag_3_graph", style={'minHeight': '400px', 'height': '60vh'}, persistence=True)), xs=12, lg=4)
    ], className="mb-5 g-3")
], fluid=True)


# ==========================================
# CALLBACKS
# ==========================================

# 1. NEWS CALLBACK (Safe for deployment)
@callback(
    Output("market_news_table", "children"),
    Input("index_period", "value") # Dummy input to trigger on load
)
def update_news(dummy):
    news = get_market_news()
    
    articles = []
    if news:
        for arts in range(len(news)):
            try:
                content = news[arts]["content"]
                articles.append({
                    "PUBDATE": content.get("pubDate"),
                    "TITLE": content.get("title"),
                    "SUMMARY": content.get("summary"),
                    "CLICKTHROUGHURL": content.get("clickThroughUrl")
                })
            except:
                continue
    
    articles_df = pd.DataFrame(articles)
    if not articles_df.empty:
        articles_df["PUBDATE"] = pd.to_datetime(articles_df["PUBDATE"]).dt.date
        articles_df["CLICKTHROUGHURL"] = articles_df["CLICKTHROUGHURL"].apply(
            lambda x: f"[VIEW ARTICLE]({x})" if x else "Link Unavailable"
        )
    else:
        articles_df = pd.DataFrame(columns=["PUBDATE", "TITLE", "SUMMARY", "CLICKTHROUGHURL"])

    return dash_table.DataTable(
        columns=[
            {"name": "PUBDATE", "id": "PUBDATE"},
            {"name": "TITLE", "id": "TITLE"},
            {"name": "SUMMARY", "id": "SUMMARY"},
            {"name": "LINK", "id": "CLICKTHROUGHURL", "presentation": "markdown"}
        ],
        data=articles_df.to_dict("records"),
        cell_selectable=True,
        page_size=5,
        style_table={"overflowX": "auto", "overflowY": "auto", "border": "1px solid #ccc"},
        style_data={"textAlign": "left", "whiteSpace": "normal", "color":"black"},
        style_header={"backgroundColor": "black", "color": "white", "fontWeight": "bold", "textAlign": "left"},
    )


# 2. INDICES CALLBACK
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
    # Fetch Cached JSON
    json_data = get_market_data(market_indeces, period, interval)
    index_data = pd.read_json(json_data, convert_dates=True)
    
    if index_data.empty:
        return [html.Div("No Data")]*3 + [go.Figure()]*3

    # Handle MultiIndex if present
    if isinstance(index_data.columns, pd.MultiIndex):
        # If 'Close' is level 0, we select it. 
        # Note: Depending on yf version, it might come as [Price, Ticker] or [Ticker, Price]
        try:
            Closing_prices = index_data['Close'].copy()
        except KeyError:
            # Fallback if 'Close' isn't top level
            Closing_prices = index_data.copy() 
    else:
        # If accessing index_data['Close'] failed previously or structure is flat
        if 'Close' in index_data:
            Closing_prices = index_data['Close'].copy()
        else:
             Closing_prices = index_data.copy()

    # Calculate Changes
    sp500_change = price_card_info(Closing_prices, "^GSPC")
    dow_change = price_card_info(Closing_prices, "^DJI")
    nas_change = price_card_info(Closing_prices, "^IXIC")
    
    sp500_card = make_card("SP500 PRICE CHANGE",sp500_change[0], "SP500 PREVIOUS PRICE", sp500_change[1])
    dow_card = make_card("DOW PRICE CHANGE", dow_change[0], "DOW PREVIOUS PRICE", dow_change[1])
    nas_card = make_card("NASDAQ PRICE CHANGE", nas_change[0], "NAS PREVIOUS PRICE", nas_change[1])
        
    # Moving Averages
    for col in market_indeces:
        if col in Closing_prices.columns:
            Closing_prices[f"{col}_30_MA"] = Closing_prices[col].rolling(window=30).mean()
            Closing_prices[f"{col}_50_MA"] = Closing_prices[col].rolling(window=50).mean()
            Closing_prices[f"{col}_200_MA"] = Closing_prices[col].rolling(window=200).mean()

    closing_prices_nona = Closing_prices[pd.notna(Closing_prices)]
    
    sp500_fig = make_plot(closing_prices_nona,"^GSPC", "S&P 500 Price Trend")
    dowjones_fig = make_plot(closing_prices_nona,"^DJI", "Dow Jones Industrial Average Price Trend")
    nasdaq_fig = make_plot(closing_prices_nona,"^IXIC", "Nasdaq Composite Price Trend")
    
    return sp500_card, dow_card, nas_card, sp500_fig, dowjones_fig, nasdaq_fig


# 3. METALS CALLBACK
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
    tickers = [metal_1, metal_2, metal_3]
    json_data = get_market_data(tickers, period, interval)
    metals_data = pd.read_json(json_data, convert_dates=True)

    if metals_data.empty:
        return [html.Div("No Data")]*3 + [go.Figure()]*3

    # Extract Close
    if 'Close' in metals_data.columns or (isinstance(metals_data.columns, pd.MultiIndex) and 'Close' in metals_data.columns.get_level_values(0)):
        try:
             metals_closing = metals_data["Close"].copy()
        except:
             metals_closing = metals_data.copy()
    else:
        metals_closing = metals_data.copy()

    # Price Cards
    metal_cards = {}
    for x in tickers:
        if x in metals_closing.columns:
            prices = price_card_info(metals_closing, x)
            metal_cards[x] = make_card(f"{x} PRICE CHANGE", prices[0], f"{x} PREVIOUS PRICE", prices[1])
        else:
            metal_cards[x] = html.Div("N/A")

    # Moving Averages
    for col in tickers:
        if col in metals_closing.columns:
            metals_closing[f"{col}_30_MA"] = metals_closing[col].rolling(window=30).mean()
            metals_closing[f"{col}_50_MA"] = metals_closing[col].rolling(window=50).mean()
            metals_closing[f"{col}_200_MA"] = metals_closing[col].rolling(window=200).mean()
    
    metals_closing_nona = metals_closing.dropna()
    
    fig1 = make_plot(metals_closing_nona, metal_1, f"{commodity_names_m.get(metal_1, metal_1)} Price Trend")
    fig2 = make_plot(metals_closing_nona, metal_2, f"{commodity_names_m.get(metal_2, metal_2)} Price Trend")
    fig3 = make_plot(metals_closing_nona, metal_3, f"{commodity_names_m.get(metal_3, metal_3)} Price Trend")
    
    return metal_cards[metal_1], metal_cards[metal_2], metal_cards[metal_3], fig1, fig2, fig3


# 4. ENERGY CALLBACK
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
    tickers = [energy_1, energy_2, energy_3]
    json_data = get_market_data(tickers, period, interval)
    energy_data = pd.read_json(json_data, convert_dates=True)
    
    if energy_data.empty:
         return [html.Div("No Data")]*3 + [go.Figure()]*3

    if 'Close' in energy_data:
        energy_closing = energy_data["Close"].copy()
    else:
        energy_closing = energy_data.copy()

    energy_cards = {}
    for x in tickers:
        if x in energy_closing.columns:
            prices = price_card_info(energy_closing, x)
            energy_cards[x] = make_card(f"{x} PRICE CHANGE", prices[0], f"{x} PREVIOUS PRICE", prices[1])
        else:
            energy_cards[x] = html.Div("N/A")
    
    for col in tickers:
        if col in energy_closing.columns:
            energy_closing[f"{col}_30_MA"] = energy_closing[col].rolling(window=30).mean()
            energy_closing[f"{col}_50_MA"] = energy_closing[col].rolling(window=50).mean()
            energy_closing[f"{col}_200_MA"] = energy_closing[col].rolling(window=200).mean()
            
    energy_closing_nona = energy_closing.dropna()
    
    fig1 = make_plot(energy_closing_nona, energy_1, f"{commodity_names_e.get(energy_1, energy_1)} Price Trend")
    fig2 = make_plot(energy_closing_nona, energy_2, f"{commodity_names_e.get(energy_2, energy_2)} Price Trend")
    fig3 = make_plot(energy_closing_nona, energy_3, f"{commodity_names_e.get(energy_3, energy_3)} Price Trend")
    
    return energy_cards[energy_1], energy_cards[energy_2], energy_cards[energy_3], fig1, fig2, fig3


# 5. AGRICULTURE CALLBACK
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
    tickers = [ag_1, ag_2, ag_3]
    json_data = get_market_data(tickers, period, interval)
    ag_data = pd.read_json(json_data, convert_dates=True)
    
    if ag_data.empty:
         return [html.Div("No Data")]*3 + [go.Figure()]*3

    if 'Close' in ag_data:
        ag_closing = ag_data["Close"].copy()
    else:
        ag_closing = ag_data.copy()

    ag_cards = {}
    for x in tickers:
        if x in ag_closing.columns:
            prices = price_card_info(ag_closing, x)
            ag_cards[x] = make_card(f"{x} PRICE CHANGE", prices[0], f"{x} PREVIOUS PRICE", prices[1])
        else:
             ag_cards[x] = html.Div("N/A")
    
    for col in tickers:
        if col in ag_closing.columns:
            ag_closing[f"{col}_30_MA"] = ag_closing[col].rolling(window=30).mean()
            ag_closing[f"{col}_50_MA"] = ag_closing[col].rolling(window=50).mean()
            ag_closing[f"{col}_200_MA"] = ag_closing[col].rolling(window=200).mean()
            
    ag_closing_nona = ag_closing.dropna()
    
    fig1 = make_plot(ag_closing_nona, ag_1, f"{commodity_names_ag.get(ag_1, ag_1)} Price Trend")
    fig2 = make_plot(ag_closing_nona, ag_2, f"{commodity_names_ag.get(ag_2, ag_2)} Price Trend")
    fig3 = make_plot(ag_closing_nona, ag_3, f"{commodity_names_ag.get(ag_3, ag_3)} Price Trend")
    
    return ag_cards[ag_1], ag_cards[ag_2], ag_cards[ag_3], fig1, fig2, fig3

