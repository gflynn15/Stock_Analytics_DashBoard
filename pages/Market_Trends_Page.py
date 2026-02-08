import numpy as np
import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import threading
import sys
import os

# Find parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from app_functions import make_plot, price_card_info, make_card
from Yfinance_Dash_2_5 import cache

# --- CONFIG ---
dash.register_page(__name__, name="MARKET OVERVIEW", path="/", order=1)
load_figure_template('simplex')

interval = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
period = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
market_indeces = ["^GSPC","^DJI","^IXIC"]

metals = ["GC=F","SI=F","PL=F","PA=F"]
commodity_names_m = {"GC=F": "Gold", "SI=F": "Silver", "PL=F": "Platinum", "PA=F": "Palladium"}

energy = ["CL=F","BZ=F","NG=F","HO=F","RB=F"]
commodity_names_e = {"CL=F":"CRUDE OIL", "BZ=F":"BRENT CRUDE", "NG=F":"NATURAL GAS", "HO=F":"HEATING OIL", "RB=F":"RBOB GASOLINE"}

ag = ["ZC=F","ZW=F","ZS=F","KC=F","LE=F","HE=F","SB=F"]
commodity_names_ag = {"ZC=F":"CORN", "ZW=F":"WHEAT", "ZS=F":"SOY BEANS", "KC=F":"COFFEE", "HE=F":"LEAN HOGS", "SB=F":"SUGAR", "LE=F":"LIVE CATTLE"}

# --- HELPER: ROBUST DOWNLOADER ---
def get_combined_data(tickers, period, interval):
    """
    Downloads tickers ONE BY ONE to prevent batch timeouts.
    Returns a combined DataFrame of 'Close' prices.
    """
    combined_df = pd.DataFrame()
    
    for t in tickers:
        try:
            # Download single ticker - highly reliable
            df = yf.download(t, period=period, interval=interval, threads=False, progress=False, auto_adjust=True)
            
            # Extract Close price
            if not df.empty:
                if 'Close' in df.columns:
                    combined_df[t] = df['Close']
                elif 'Adj Close' in df.columns:
                    combined_df[t] = df['Adj Close']
        except Exception:
            # If one fails, we just skip it so the others still show up
            continue
            
    return combined_df

# --- HELPER: SAFE PLOT ---
def create_graph(df, ticker, name):
    """
    Checks if data exists for the ticker. If yes, plots it.
    If no (download failed), returns empty graph to prevent crash.
    """
    if df is None or ticker not in df.columns:
        return go.Figure(layout=dict(title=f"Data Unavailable for {name}"))
    
    # Filter for valid columns (Ticker + its MAs)
    cols_to_plot = [ticker]
    for ma in ["30_MA", "50_MA", "200_MA"]:
        if f"{ticker}_{ma}" in df.columns:
            cols_to_plot.append(f"{ticker}_{ma}")
            
    # Create subset and drop NaNs only for this ticker
    # This prevents the 'ValueError: All arguments should have same length'
    df_clean = df[cols_to_plot].dropna(subset=[ticker])
    
    if df_clean.empty:
        return go.Figure(layout=dict(title=f"No Data for {name}"))

    return make_plot(df_clean, ticker, name)


# --- LAYOUT ---
layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("MARKET REVIEW", style={"textAlign":"center", "fontWeight":"bold"}), width=12)], className="mb-4 mt-2"),

    # Filters
    dbc.Row([
        dbc.Col([html.Label(html.B("Select Period:")), dcc.Dropdown(period, "1y", id="index_period", clearable=False)], xs=6, md=3, lg=2),
        dbc.Col([html.Label(html.B("Select Interval")), dcc.Dropdown(interval, "1d", id="index_interval", clearable=False)], xs=6, md=3, lg=2),
    ], className="mb-4 g-3"),

    # Indices
    dbc.Row([
        dbc.Col(html.Div(id='sp500_change'), xs=12, md=4),
        dbc.Col(html.Div(id='dow_change'), xs=12, md=4),
        dbc.Col(html.Div(id='nas_change'), xs=12, md=4),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="sp500", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="dowjones", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="nasdaq", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4)
    ], className="mb-5 g-3"),
    
    # News
    dbc.Row([dbc.Col(html.H1("RECENT NEWS", style={"textAlign":"center","fontWeight":"bold"}), width=12)]),
    dbc.Row([dbc.Col(dcc.Loading(html.Div(id='market_news_table')), width=12)], className="mb-5"),
    
    # Metals
    dbc.Row([dbc.Col(html.H1("METAL COMMODITIES TRENDS", style={"textAlign":"center","fontWeight":"bold"}), width=12)]),
    dbc.Row([
        dbc.Col([html.Label(html.B("Select Period")), dcc.Dropdown(period, "1y", id="commodity_period_m", clearable=False)], xs=6, md=2),
        dbc.Col([html.Label(html.B("Select Interval")), dcc.Dropdown(interval, "1d", id="commodity_interval_m", clearable=False)], xs=6, md=2),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col([html.Label(html.B("Select Metal")), dcc.Dropdown(metals, "SI=F", id="metal_1", clearable=False), html.Div(id="metal_1_price")], xs=12, md=4),
        dbc.Col([html.Label(html.B("Select Metal")), dcc.Dropdown(metals, "GC=F", id="metal_2", clearable=False), html.Div(id="metal_2_price")], xs=12, md=4),
        dbc.Col([html.Label(html.B("Select Metal")), dcc.Dropdown(metals, "PL=F", id="metal_3", clearable=False), html.Div(id="metal_3_price")], xs=12, md=4)
    ], className="g-3"),
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="metal_1_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="metal_2_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="metal_3_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4)
    ], className="mb-5 g-3"),

    # Energy
    dbc.Row([dbc.Col(html.H1("ENERGY COMMODITIES TRENDS", style={"textAlign":"center","fontWeight":"bold"}), width=12)], className="mt-5 mb-3"),
    dbc.Row([
        dbc.Col([html.Label(html.B("Select Period")), dcc.Dropdown(period, "1y", id="commodity_period_e", clearable=False)], xs=6, md=3, lg=2),
        dbc.Col([html.Label(html.B("Select Interval")), dcc.Dropdown(interval, "1d", id="commodity_interval_e", clearable=False)], xs=6, md=3, lg=2),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col([html.Label(html.B("Select Energy")), dcc.Dropdown(energy, "CL=F", id="energy_1", clearable=False), dcc.Loading(html.Div(id="energy_1_price"), type="circle")], xs=12, md=4),
        dbc.Col([html.Label(html.B("Select Energy")), dcc.Dropdown(energy, "NG=F", id="energy_2", clearable=False), dcc.Loading(html.Div(id="energy_2_price"), type="circle")], xs=12, md=4),
        dbc.Col([html.Label(html.B("Select Energy")), dcc.Dropdown(energy, "HO=F", id="energy_3", clearable=False), dcc.Loading(html.Div(id="energy_3_price"), type="circle")], xs=12, md=4)
    ], className="g-3"),
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="energy_1_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="energy_2_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="energy_3_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4)
    ], className="mb-5 g-3"),

    # Agriculture
    dbc.Row([dbc.Col(html.H1("AGRICULTURE COMMODITIES TRENDS", style={"textAlign":"center","fontWeight":"bold"}), width=12)], className="mt-5 mb-3"),
    dbc.Row([
        dbc.Col([html.Label(html.B("Select Period")), dcc.Dropdown(period, "1y", id="commodity_period_a", clearable=False)], xs=6, md=3, lg=2),
        dbc.Col([html.Label(html.B("Select Interval")), dcc.Dropdown(interval, "1d", id="commodity_interval_a", clearable=False)], xs=6, md=3, lg=2),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col([html.Label(html.B("Select Agriculture Commodity")), dcc.Dropdown(ag, "ZC=F", id="ag_1", clearable=False), dcc.Loading(html.Div(id="ag_1_price"), type="circle")], xs=12, md=4),
        dbc.Col([html.Label(html.B("Select Agriculture Commodity")), dcc.Dropdown(ag, "SB=F", id="ag_2", clearable=False), dcc.Loading(html.Div(id="ag_2_price"), type="circle")], xs=12, md=4),
        dbc.Col([html.Label(html.B("Select Agriculture Commodity")), dcc.Dropdown(ag, "LE=F", id="ag_3", clearable=False), dcc.Loading(html.Div(id="ag_3_price"), type="circle")], xs=12, md=4)
    ], className="g-3"),
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="ag_1_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="ag_2_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4),
        dbc.Col(dcc.Loading(dcc.Graph(id="ag_3_graph", style={'minHeight': '400px', 'height': '60vh'})), xs=12, lg=4)
    ], className="mb-5 g-3")
], fluid=True)


# --- CALLBACKS ---

# 1. NEWS CALLBACK
@callback(
    Output("market_news_table", "children"),
    Input("index_period", "value")
)
@cache.memoize(timeout=3600)
def update_news(dummy):
    try:
        stock = yf.Ticker("^GSPC")
        news = stock.get_news(count=10, tab="news")
        
        articles = []
        for item in news:
            if "content" in item:
                news_df = pd.DataFrame(item["content"])
                if "url" in news_df.index:
                    articles.append(news_df.loc["url", :])
        
        if not articles:
            return dash_table.DataTable(data=[])

        keys = ["title", "summary", "pubDate", "clickThroughUrl"]
        articles_df = pd.DataFrame(articles, columns=keys)
        articles_df.columns = articles_df.columns.str.upper()
        articles_df = articles_df[["PUBDATE", "TITLE", "SUMMARY", "CLICKTHROUGHURL"]]
        articles_df["PUBDATE"] = pd.to_datetime(articles_df["PUBDATE"]).dt.date
        articles_df["CLICKTHROUGHURL"] = articles_df["CLICKTHROUGHURL"].apply(
            lambda x: f"[VIEW ARTICLE]({x})" if x else "Link Unavailable"
        )

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
    except Exception:
        return dash_table.DataTable(data=[])


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
@cache.memoize(timeout=600)
def index_trend_chart(period, interval):
    # Use Robust Downloader
    Closing_prices = get_combined_data(market_indeces, period, interval)
    
    if Closing_prices.empty:
        return [html.Div("No Data")]*3 + [go.Figure()]*3

    # Calculate Changes (Handle Missing Columns)
    sp500_change = price_card_info(Closing_prices, "^GSPC") if "^GSPC" in Closing_prices.columns else [0,0]
    dow_change = price_card_info(Closing_prices, "^DJI") if "^DJI" in Closing_prices.columns else [0,0]
    nas_change = price_card_info(Closing_prices, "^IXIC") if "^IXIC" in Closing_prices.columns else [0,0]
    
    sp500_card = make_card("SP500 PRICE CHANGE",sp500_change[0], "SP500 PREVIOUS PRICE", sp500_change[1])
    dow_card = make_card("DOW PRICE CHANGE", dow_change[0], "DOW PREVIOUS PRICE", dow_change[1])
    nas_card = make_card("NASDAQ PRICE CHANGE", nas_change[0], "NAS PREVIOUS PRICE", nas_change[1])
        
    for col in market_indeces:
        if col in Closing_prices.columns:
            if np.count_nonzero(Closing_prices.index) > 200:
                Closing_prices[f"{col}_30_MA"] = Closing_prices[col].rolling(window=30).mean()
                Closing_prices[f"{col}_50_MA"] = Closing_prices[col].rolling(window=50).mean()
                Closing_prices[f"{col}_200_MA"] = Closing_prices[col].rolling(window=200).mean()
            elif 50 <= np.count_nonzero(Closing_prices.index) < 200:
                Closing_prices[f"{col}_30_MA"] = Closing_prices[col].rolling(window=30).mean()
                Closing_prices[f"{col}_50_MA"] = Closing_prices[col].rolling(window=50).mean()
            elif 30 <= np.count_nonzero(Closing_prices.index) < 50:
                Closing_prices[f"{col}_30_MA"] = Closing_prices[col].rolling(window=30).mean()

    # Use Safe Plotter
    sp500_fig = create_graph(Closing_prices, "^GSPC", "S&P 500 Price Trend")
    dowjones_fig = create_graph(Closing_prices, "^DJI", "Dow Jones Industrial Average Price Trend")
    nasdaq_fig = create_graph(Closing_prices, "^IXIC", "Nasdaq Composite Price Trend")
    
    return sp500_card,dow_card,nas_card,sp500_fig, dowjones_fig, nasdaq_fig


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
@cache.memoize(timeout=600)
def metals_trend_charts(period, interval, metal_1, metal_2, metal_3):
    tickers = [metal_1, metal_2, metal_3]
    metals_closing = get_combined_data(tickers, period, interval)

    if metals_closing.empty:
        return [html.Div("No Data")]*3 + [go.Figure()]*3

    metal_cards = {}
    for x in tickers:
        if x in metals_closing.columns:
            prices = price_card_info(metals_closing, x)
            metal_cards[x] = make_card(f"{x} PRICE CHANGE", prices[0], f"{x} PREVIOUS PRICE", prices[1])
        else:
            metal_cards[x] = make_card(f"{x} UNAVAILABLE", 0, "N/A", 0)

    for col in tickers:
        if col in metals_closing.columns:
            if np.count_nonzero(metals_closing.index) > 200:
                metals_closing[f"{col}_30_MA"] = metals_closing[col].rolling(window=30).mean()
                metals_closing[f"{col}_50_MA"] = metals_closing[col].rolling(window=50).mean()
                metals_closing[f"{col}_200_MA"] = metals_closing[col].rolling(window=200).mean()
            elif 50 <= np.count_nonzero(metals_closing.index) < 200:
                metals_closing[f"{col}_30_MA"] = metals_closing[col].rolling(window=30).mean()
                metals_closing[f"{col}_50_MA"] = metals_closing[col].rolling(window=50).mean()
            elif 30 <= np.count_nonzero(metals_closing.index) < 50:
                metals_closing[f"{col}_30_MA"] = metals_closing[col].rolling(window=30).mean()
    
    metal_1_fig = create_graph(metals_closing, metal_1, f"{commodity_names_m.get(metal_1, metal_1)} Price Trend")
    metal_2_fig = create_graph(metals_closing, metal_2, f"{commodity_names_m.get(metal_2, metal_2)} Price Trend")
    metal_3_fig = create_graph(metals_closing, metal_3, f"{commodity_names_m.get(metal_3, metal_3)} Price Trend")
    
    return metal_cards[metal_1], metal_cards[metal_2], metal_cards[metal_3], metal_1_fig, metal_2_fig, metal_3_fig


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
@cache.memoize(timeout=600)
def energy_trend_charts(period, interval, energy_1, energy_2, energy_3):
    tickers = [energy_1, energy_2, energy_3]
    energy_closing = get_combined_data(tickers, period, interval)
    
    if energy_closing.empty:
         return [html.Div("No Data")]*3 + [go.Figure()]*3

    energy_cards = {}
    for x in tickers:
        if x in energy_closing.columns:
            prices = price_card_info(energy_closing, x)
            energy_cards[x] = make_card(f"{x} PRICE CHANGE", prices[0], f"{x} PREVIOUS PRICE", prices[1])
        else:
            energy_cards[x] = make_card(f"{x} UNAVAILABLE", 0, "N/A", 0)
    
    for col in tickers:
        if col in energy_closing.columns:
            if np.count_nonzero(energy_closing.index) > 200:
                energy_closing[f"{col}_30_MA"] = energy_closing[col].rolling(window=30).mean()
                energy_closing[f"{col}_50_MA"] = energy_closing[col].rolling(window=50).mean()
                energy_closing[f"{col}_200_MA"] = energy_closing[col].rolling(window=200).mean()
            elif 50 <= np.count_nonzero(energy_closing.index) < 200:
                energy_closing[f"{col}_30_MA"] = energy_closing[col].rolling(window=30).mean()
                energy_closing[f"{col}_50_MA"] = energy_closing[col].rolling(window=50).mean()
            elif 30 <= np.count_nonzero(energy_closing.index) < 50:
                energy_closing[f"{col}_30_MA"] = energy_closing[col].rolling(window=30).mean()

    energy_1_fig = create_graph(energy_closing, energy_1, f"{commodity_names_e.get(energy_1, energy_1)} Price Trend")
    energy_2_fig = create_graph(energy_closing, energy_2, f"{commodity_names_e.get(energy_2, energy_2)} Price Trend")
    energy_3_fig = create_graph(energy_closing, energy_3, f"{commodity_names_e.get(energy_3, energy_3)} Price Trend")
    
    return energy_cards[energy_1], energy_cards[energy_2], energy_cards[energy_3], energy_1_fig, energy_2_fig, energy_3_fig


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
@cache.memoize(timeout=600)
def ag_trend_charts(period, interval, ag_1, ag_2, ag_3):
    tickers = [ag_1, ag_2, ag_3]
    ag_closing = get_combined_data(tickers, period, interval)
    
    if ag_closing.empty:
         return [html.Div("No Data")]*3 + [go.Figure()]*3

    ag_cards = {}
    for x in tickers:
        if x in ag_closing.columns:
            prices = price_card_info(ag_closing, x)
            ag_cards[x] = make_card(f"{x} PRICE CHANGE", prices[0], f"{x} PREVIOUS PRICE", prices[1])
        else:
            ag_cards[x] = make_card(f"{x} UNAVAILABLE", 0, "N/A", 0)
    
    for col in tickers:
        if col in ag_closing.columns:
            if np.count_nonzero(ag_closing.index) > 200:
                ag_closing[f"{col}_30_MA"] = ag_closing[col].rolling(window=30).mean()
                ag_closing[f"{col}_50_MA"] = ag_closing[col].rolling(window=50).mean()
                ag_closing[f"{col}_200_MA"] = ag_closing[col].rolling(window=200).mean()
            elif 50 <= np.count_nonzero(ag_closing.index) < 200:
                ag_closing[f"{col}_30_MA"] = ag_closing[col].rolling(window=30).mean()
                ag_closing[f"{col}_50_MA"] = ag_closing[col].rolling(window=50).mean()
            elif 30 <= np.count_nonzero(ag_closing.index) < 50:
                ag_closing[f"{col}_30_MA"] = ag_closing[col].rolling(window=30).mean()

    ag_1_fig = create_graph(ag_closing, ag_1, f"{commodity_names_ag.get(ag_1, ag_1)} Price Trend")
    ag_2_fig = create_graph(ag_closing, ag_2, f"{commodity_names_ag.get(ag_2, ag_2)} Price Trend")
    ag_3_fig = create_graph(ag_closing, ag_3, f"{commodity_names_ag.get(ag_3, ag_3)} Price Trend")
    
    return ag_cards[ag_1], ag_cards[ag_2], ag_cards[ag_3], ag_1_fig, ag_2_fig, ag_3_fig