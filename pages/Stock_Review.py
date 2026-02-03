# %% [markdown]
# - Creating a new Dash App using the yfinance API as the data pipeline

# %% [markdown]
# - Importing the necessary libraries to execute the application

# %%
import numpy as np
import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import openpyxl
import warnings
warnings.filterwarnings("ignore")
from bs4 import BeautifulSoup
import requests
import threading
# %% [markdown]
# - Utilizing polygon io to provide a comprehensive list of stock symbols

# %% [markdown]
# - Creating the list for date filters to be used in the line charts

# %% [markdown]
# - Creating the dash board layout
#     - Created a 3 section layer layout with the top section being dedicated to closing price trend analysis
#     - 2nd section being dedicated to a news article table to review company sentiment analysis
#     - 3rd section of the chart reviewing the company historical financials to track performance there

# %%
dash.register_page(__name__, name="Stock_Review", path="/Stock_Review", order=2)
load_figure_template('simplex')
##Importing Stok Symbols
sp500_df = pd.read_csv(r"C:\Users\Gabriel Flynn\OneDrive\OneDrive - University of Texas at El Paso\Documents\Python Projects\Yfinance_Stock_Data_Analysis\yfin_dash_app_api_backend\stock_symbols_list.csv")
symbols = (sp500_df['Symbol'] + "-" + sp500_df['Security']).tolist()
symbols = sorted(symbols)

period = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

layout = dbc.Container([
    # --- Title Row ---
    dbc.Row([
        html.H1('COMPANY OVERVIEW', style={'textAlign': 'center'})
    ]),
    
    # --- Drop Down Menu Row (Reverted to your structure) ---
    dbc.Row([
        dbc.Col([
            html.Label('Select Stock'),
            dcc.Dropdown(symbols, symbols[0], id='stock_symbols', clearable=False),
        ], xs=12, md=4, lg=2),
        dbc.Col([
            html.Label('Select Period'),
            dcc.Dropdown(period, '1y', id='period', clearable=False)
        ], xs=12, md=4, lg=2),
        dbc.Col([
            html.Label('Select Interval'),
            dcc.Dropdown(intervals, '1d', id='intervals', clearable=False)
        ], xs=12, md=4, lg=2)      
    ], className='mb-4'),

    # --- Price Trend Charts (Added Height Fix Only) ---
    dbc.Row([
        dbc.Row([
            html.H3("📈 Pricing Trend", style={'textAlign': 'left'})  
        ]),
        # FIX: Explicit height prevents the graph from disappearing
        dbc.Row([
            dcc.Graph(
                id='trend_line', 
                config={'responsive': True},
                style={'height': '60vh', 'minHeight': '500px'}
            )
        ]),
        
        # --- RSI and MACD (Added Height Fix Only) ---
        dbc.Row([
            dbc.Col([
                dbc.Row([html.H3("📈 Relative Strength Index (RSI)", style={'textAlign': 'left'})]),
                dbc.Row([
                    dcc.Graph(
                        id='RSI', 
                        config={'responsive': True},
                        style={'height': '40vh', 'minHeight': '350px'}
                    )
                ])
            ], xs=12, md=6, lg=6),
            dbc.Col([
                dbc.Row([html.H3("📈 MACD Signals", style={'textAlign': 'left'})]),
                dbc.Row([
                    dcc.Graph(
                        id='MACD', 
                        config={'responsive': True},
                        style={'height': '40vh', 'minHeight': '350px'}
                    )
                ])
            ], xs=12, md=6, lg=6)
        ], className='mb-4')
    ]),

    # --- News Section (Reverted EXACTLY to your structure) ---
    dbc.Row([
        dbc.Row(html.H3("📰 Latest News", style={'textAlign': 'left'})),
        dbc.Row(html.Div(id='news_table'))
    ], className='mb-4'),

    # --- Distribution and Forecast Table ---
    dbc.Row([
        # Left Side
        dbc.Col([
            dbc.Row([
                dbc.Row(html.H3("📈 Price & Volume Distribution")),
                dbc.Row(html.Div(id='price_dis_table'))
            ], className='mb-4'),
            
            dbc.Row([
                dbc.Row(html.H3("🔮 Earnings Forecast")),
                dbc.Row(html.Div(id='forecast'))
            ])
        ], xs=12, md=6, lg=6),
        
        # Right Side
        dbc.Col([
            dbc.Row([
                dbc.Row(html.H3("📊 Historical Financials")),
                dbc.Row(html.Div(id='historical_financials'))
            ], className='mb-4'),
            dbc.Row([
                dbc.Row(html.H3(children='📊 Quarterly Earnings Per Share')),
                dbc.Row([
                    dcc.Graph(
                        id='EPS', 
                        config={'responsive': True},
                        style={'height': '40vh', 'minHeight': '350px'}
                    )
                ])
            ])
        ], xs=12, md=6, lg=6)    
    ])
], fluid=True)
##End of outter layout container

# %% [markdown]
# - Creating the graph call backs

# %%
def get_price_data(ticker: str, period: str, interval: str) -> str:
    """Fetch & cache OHLCV price data as JSON."""
    ticker = ticker.split("-")[0]
    df = yf.download(tickers=ticker, period=period, interval=interval)
    # Use ISO date format so you can read it back easily
    return df.to_json(date_format="iso")
thread_lock = threading.Lock()
# -------- Trend line, RSI, forecast, EPS, MACD --------
@callback(
    Output("news_table", "children"),
    Output("historical_financials", "children"),
    Output("trend_line", "figure"),
    Output("RSI", "figure"),
    Output("forecast", "children"),
    Output("EPS", "figure"),
    Output("MACD", "figure"),
    Output("price_dis_table", "children"),
    Input("stock_symbols", "value"),
    Input("period", "value"),
    Input("intervals", "value"),
)

##Creating the price trend chart analysis
def trend_chart(ticker: str, period: str, intervals: str):
    ticker = ticker.split("-")[0]
    with thread_lock:
        company = yf.Ticker(ticker)
    # News (robust parsing)
    with thread_lock:
        news = company.get_news(count=10, tab="news", proxy=None)
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
    articles = dash_table.DataTable(
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

    # Historical financials
    financials = company.financials

    financials = financials.T

    financials.index = pd.to_datetime(financials.index).year

    key_financials = financials[['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income','Normalized Income']]

    key_financials.reset_index(inplace=True)

    key_financials.rename(columns={'index':'Year'}, inplace=True)
    key_financials.dropna(inplace=True)
    for x in key_financials.columns[1:]:
        key_financials[x] = key_financials[x].apply(lambda x: f"${x:,.0f}")
    
    
    financials_table = dash_table.DataTable(
        columns=[{'name':col, 'id':col} for col in key_financials.columns],
        data=key_financials.to_dict('records'),
        cell_selectable=True,
        style_table={
        'overflowX': 'auto',         # Scroll horizontally if needed
        'overflowY': 'auto',         # Scroll vertically inside the box
        'border': '2px solid #ccc'   # Optional: makes the box look nice
    },
    
    style_cell={
        'textAlign': 'left',
        'whiteSpace': 'normal',
        "color":"black"
    },

    style_header={
        'backgroundColor': 'black',
        'color': 'white',
        'fontWeight': 'bold',
        'textAlign': 'center',
    }
)
    ##Extracting price data from the yfinance API to create the trend line and technical indicator lines
    pricing_data = get_price_data(ticker, period, intervals)
    df = pd.read_json(pricing_data, convert_dates=True)
    
    stock_cols = []
    for x in df.columns:
        stock_vals = x.split(",")[0].replace("(", "").replace("'","")
        stock_cols.append(stock_vals)
    
    df.columns = stock_cols
    #df = yf.download(tickers=ticker, period=period, interval=intervals)

    if df.empty:
        return go.Figure(), go.Figure(), html.Div("No data."), go.Figure(), go.Figure()

    # Flatten multi-index columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Moving averages (guard short series)
    df["50_DAY_MA"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["200_DAY_MA"] = df["Close"].rolling(window=200, min_periods=1).mean()

    # RSI (EMA version)
    rs = df[["Close"]].copy()
    rs["Δ"] = rs["Close"].diff()
    gains = rs["Δ"].clip(lower=0)
    losses = (-rs["Δ"].clip(upper=0))
    rs["avg_gain"] = gains.ewm(com=14 - 1, adjust=False).mean()
    rs["avg_loss"] = losses.ewm(com=14 - 1, adjust=False).mean()
    rs["RSI"] = 100 - (100 / (1 + (rs["avg_gain"] / rs["avg_loss"].replace(0, np.nan))))
    rs["RSI"] = rs["RSI"].round(2)

    # EPS (guard empty)
    earns = getattr(company, "earnings_history", None)
    eps_fig = go.Figure()
    if isinstance(earns, pd.DataFrame) and not earns.empty:
        try:
            earns = earns.copy()
            earns.index = pd.to_datetime(earns.index).date
            eps_fig.add_trace(go.Bar(x=earns.index, y=earns.get("epsActual"), name="EPS Actual"))
            eps_fig.add_trace(go.Bar(x=earns.index, y=earns.get("epsEstimate"), name="EPS Estimate"))
            eps_fig.update_layout(
                title={"text":"<b>Actual vs Estimate Earnings per Share<b>",
                      "x":.5,
                      "y":.95},
                xaxis_title="<b>Date<b>",
                yaxis_title="<b>EPS<b>",
                barmode="group",
                legend=dict(title="Legend", orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
            )
        except Exception:
            pass

    # RSI figure
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=rs.index, y=rs["RSI"], name="RSI"))
    rsi_fig.update_layout(
        title={"text":"<b>Relative Strength Index (RSI)<b>",
              "x":.5,
              "y":.95},
        xaxis_title="<b>DATE<b>",
        yaxis_title="<b>RSI<b>",
        yaxis=dict(range=[0, 100]),
        shapes=[
            dict(type="line", yref="y", y0=70, y1=70, xref="x", x0=df.index[0], x1=df.index[-1]),
            dict(type="line", yref="y", y0=30, y1=30, xref="x", x0=df.index[0], x1=df.index[-1]),
        ],
    )

    # MACD
    macd_df = df[["Close"]].copy()
    macd_df["EMA_12"] = macd_df["Close"].ewm(span=12, adjust=False).mean()
    macd_df["EMA_26"] = macd_df["Close"].ewm(span=26, adjust=False).mean()
    macd_df["MACD_LINE"] = macd_df["EMA_12"] - macd_df["EMA_26"]
    macd_df["SIGNAL_LINE"] = macd_df["MACD_LINE"].ewm(span=9, adjust=False).mean()
    macd_df["HISTOGRAM"] = macd_df["MACD_LINE"] - macd_df["SIGNAL_LINE"]

    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["MACD_LINE"], name="MACD"))
    macd_fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["SIGNAL_LINE"], name="Signal"))
    macd_fig.add_trace(go.Bar(x=macd_df.index, y=macd_df["HISTOGRAM"], name="Histogram"))
    macd_fig.update_layout(title={"text":"<b>MACD<b>",
                                 "x":.5,
                                 "y":.95},
                          yaxis_title="<b>MACD<b>",
                          xaxis_title="<b>DATE<b>")

    # Trend line
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=f"{ticker} Close"))
    trend_fig.add_trace(go.Scatter(x=df.index, y=df["50_DAY_MA"], name="50-Day MA"))
    trend_fig.add_trace(go.Scatter(x=df.index, y=df["200_DAY_MA"], name="200-Day MA"))
    trend_fig.update_layout(title={"text":"<b>Closing Price Trend<b>"}, xaxis_title="<b>DATE<b>", yaxis_title="<b>PRICE<b>")

    # Forecast Table 
    stock = yf.Ticker(ticker)
    forecast_ticker = pd.DataFrame(stock.calendar).T
    forecast_ticker.rename(columns={0:'Projected Forecast'}, inplace=True)
    forecast_ticker.reset_index(inplace=True)
    forecast_ticker.rename(columns={'index':'Forecast Type'}, inplace=True)
    forecast_table = dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in forecast_ticker.columns],
        data=forecast_ticker.to_dict('records'),
        style_table={'overflowY': 'auto','minHeight':'100%'},
        style_cell={
            'textAlign': 'left',
            'whiteSpace': 'normal',
            'height': 'auto',
            'width': 'auto',
            'padding': '2px',
            "color":"black"
        },
        style_header={
            'backgroundColor': 'black',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        }
    )
    
    ##Creating the price/volume distribution table that also includes the last closing date price/volume
    
    df_close_dist = df['Close'].describe()

    df_close_dist = df_close_dist.to_frame(name='Closing Price').round(2)
    
    df_close_dist.reset_index(inplace=True)
    
    df_close_dist.rename(columns={'index':'Statistics'}, inplace=True)
    
    df_volume_dist = df['Volume'].describe()
    
    df_volume_dist = df_volume_dist.to_frame(name='Volume').round(2)
    
    df_volume_dist.reset_index(inplace=True)
    
    df_volume_dist.rename(columns={'index':'Statistics'}, inplace=True)
    
    df_close_dist = pd.concat(objs=[df_close_dist,df_volume_dist[['Volume']]], axis=1, join='inner',ignore_index=False)
    
    previous_data = df.iloc[-1,[0,4]].to_frame().transpose()

    previous_data.rename(index={previous_data.index[0]:'previous data'}, inplace=True)
    
    previous_data.reset_index(inplace=True)
    
    previous_data.rename(columns={'index':'Statistics',
                                  'Close':'Closing Price'},
                                  inplace=True)
    
    previous_data['Closing Price'] = previous_data['Closing Price'].round(2)
    
    df_close_dist = pd.concat(objs=[df_close_dist, previous_data], axis=0,ignore_index=False)
    
    df_close_dist['Volume'] = df_close_dist['Volume'].apply(lambda x: f"{x:.2e}")
    
    dist_table = dash_table.DataTable(
        columns=[{'name':col, 'id':col} for col in df_close_dist.iloc[1:,:].columns],
        data=df_close_dist.iloc[1:,:].to_dict('records'),
        style_table={'overflowY': 'auto',
                     'overflowX':'auto'},
        style_cell={'textAlign': 'left',
                    'whitespace':'normal',
                    "color":"black"},
        style_header={'backgroundColor': 'black',
                      'color': 'white',
                      'fontWeight': 'bold',
                      'textAlign': 'center'}
    )

    return articles,financials_table,trend_fig, rsi_fig, forecast_table, eps_fig, macd_fig, dist_table