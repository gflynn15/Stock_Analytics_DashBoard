import numpy as np
import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Make sure this import matches your main filename
from Yfinance_Dash_2_5 import cache 

dash.register_page(__name__, name="Stock_Review", path="/Stock_Review", order=2)
load_figure_template('simplex')

##Reading stock symbols folder
# Ensure this CSV exists in your root folder
try:
    sp500_df = pd.read_csv(r'stock_symbols_list.csv')
    symbols = (sp500_df['Symbol'] + "-" + sp500_df['Security']).tolist()
    symbols = sorted(symbols)
except:
    symbols = ['AAPL-Apple Inc', 'MSFT-Microsoft']

period = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

layout = dbc.Container([
    # --- Title Row ---
    dbc.Row([
        html.H1('COMPANY OVERVIEW', style={'textAlign': 'center'})
    ]),
    
    # --- Drop Down Menu Row ---
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

    # --- Price Trend Charts ---
    dbc.Row([
        dbc.Row([
            html.H3("📈 Pricing Trend", style={'textAlign': 'left'})  
        ]),
        dbc.Row([dcc.Loading(
                id="loading-trend",
                type="default",
                children=dcc.Graph(
                    id='trend_line', 
                    config={'responsive': True},
                    style={'height': '60vh', 'minHeight': '500px'},
                    persistence=True,
                    persistence_type='memory'
            ))
        ]),
        
        # --- RSI and MACD ---
        dbc.Row([
            dbc.Col([
                dbc.Row([html.H3("📈 Relative Strength Index (RSI)", style={'textAlign': 'left'})]),
                dbc.Row([
                    dcc.Loading(
                    id="loading-rsi",
                    type="default",
                    children=dcc.Graph(
                        id='RSI', 
                        config={'responsive': True},
                        style={'height': '40vh', 'minHeight': '350px'},
                        persistence=True
                    ))
                ])
            ], xs=12, md=6, lg=6),
            dbc.Col([
                dbc.Row([html.H3("📈 MACD Signals", style={'textAlign': 'left'})]),
                dbc.Row([
                    dcc.Loading(
                    id="loading-macd",
                    type="default",
                    children=dcc.Graph(
                        id='MACD', 
                        config={'responsive': True},
                        style={'height': '40vh', 'minHeight': '350px'},
                        persistence=True
                    ))
                ])
            ], xs=12, md=6, lg=6)
        ], className='mb-4')
    ]),

    # --- News Section ---
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
                        style={'height': '40vh', 'minHeight': '350px'},
                        persistence=True
                    )
                ])
            ])
        ], xs=12, md=6, lg=6)    
    ])
], fluid=True)


# ==========================================
# CACHED DATA FUNCTIONS (The Heavy Lifters)
# ==========================================

@cache.memoize(timeout=600)  # 10 minutes for Price
def get_price_data(ticker: str, period: str, interval: str) -> str:
    ticker = ticker.split("-")[0]
    # Fetch data
    df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    
    # FIX: Flatten MultiIndex immediately to prevent JSON errors
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df.to_json(date_format="iso")

@cache.memoize(timeout=1800) # 30 mins for News
def get_news_data(ticker: str):
    ticker = ticker.split("-")[0]
    return yf.Ticker(ticker).get_news(count=10, tab="news")

@cache.memoize(timeout=3600) # 1 hour for Financials
def get_financials_data(ticker: str):
    ticker = ticker.split("-")[0]
    return yf.Ticker(ticker).financials

@cache.memoize(timeout=3600) # 1 hour for Calendar/Earnings
def get_calendar_data(ticker: str):
    ticker = ticker.split("-")[0]
    company = yf.Ticker(ticker)
    return company.calendar, getattr(company, "earnings_history", None)


# ==========================================
# MAIN CALLBACK
# ==========================================

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
def trend_chart(ticker: str, period: str, intervals: str):
    
    # 1. FETCH NEWS (Cached)
    news = get_news_data(ticker)
    
    articles = []
    if news:
        for arts in range(len(news)):
            try:
                content = news[arts]["content"]
                # Use .get() to prevent crashes if keys are missing
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
            lambda x: f"[VIEW ARTICLE]({x})" if x else "Link unavailable"
        )
    else:
        # Fallback if no news
        articles_df = pd.DataFrame(columns=["PUBDATE", "TITLE", "SUMMARY", "CLICKTHROUGHURL"])

    news_table = dash_table.DataTable(
        columns=[
            {"name": "PUBDATE", "id": "PUBDATE"},
            {"name": "TITLE", "id": "TITLE"},
            {"name": "SUMMARY", "id": "SUMMARY"},
            {"name": "LINK", "id": "CLICKTHROUGHURL", "presentation": "markdown"}
        ],
        data=articles_df.to_dict("records"),
        cell_selectable=True,
        page_size=5,
        style_table={"overflowX": "auto", "border": "1px solid #ccc"},
        style_data={"whiteSpace": "normal", "textAlign": "left", "color": "black"},
        style_header={"backgroundColor": "black", "color": "white", "fontWeight": "bold", "textAlign": "left"}
    )

    # 2. FETCH FINANCIALS (Cached)
    financials = get_financials_data(ticker)
    
    if financials is not None and not financials.empty:
        financials = financials.T
        financials.index = pd.to_datetime(financials.index).year
        # Handle cases where columns might be missing
        available_cols = [c for c in ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income','Normalized Income'] if c in financials.columns]
        key_financials = financials[available_cols].copy()
        
        key_financials.reset_index(inplace=True)
        key_financials.rename(columns={'index':'Year'}, inplace=True)
        key_financials.dropna(inplace=True)
        
        for x in key_financials.columns[1:]:
             # Check if numeric before formatting
            if pd.api.types.is_numeric_dtype(key_financials[x]):
                key_financials[x] = key_financials[x].apply(lambda x: f"${x:,.0f}")
        
        fin_data = key_financials.to_dict('records')
        fin_cols = [{'name':col, 'id':col} for col in key_financials.columns]
    else:
        fin_data = []
        fin_cols = []

    financials_table = dash_table.DataTable(
        columns=fin_cols,
        data=fin_data,
        cell_selectable=True,
        style_table={'overflowX': 'auto', 'border': '2px solid #ccc'},
        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', "color":"black"},
        style_header={'backgroundColor': 'black', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'}
    )

    # 3. FETCH PRICING (Cached)
    pricing_data = get_price_data(ticker, period, intervals)
    df = pd.read_json(pricing_data, convert_dates=True)

    if df.empty:
        return news_table, financials_table, go.Figure(), go.Figure(), html.Div("No Data"), go.Figure(), go.Figure(), html.Div("No Data")

    # Clean Logic Removed: We handled cleaning inside get_price_data already!

    # Moving averages
    df["50_DAY_MA"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["200_DAY_MA"] = df["Close"].rolling(window=200, min_periods=1).mean()

    # RSI
    rs = df[["Close"]].copy()
    rs["Δ"] = rs["Close"].diff()
    gains = rs["Δ"].clip(lower=0)
    losses = (-rs["Δ"].clip(upper=0))
    rs["avg_gain"] = gains.ewm(com=14 - 1, adjust=False).mean()
    rs["avg_loss"] = losses.ewm(com=14 - 1, adjust=False).mean()
    rs["RSI"] = 100 - (100 / (1 + (rs["avg_gain"] / rs["avg_loss"].replace(0, np.nan))))
    rs["RSI"] = rs["RSI"].round(2)

    # 4. CALENDAR & EARNINGS (Cached)
    cal, earns = get_calendar_data(ticker)

    # EPS Chart
    eps_fig = go.Figure()
    if isinstance(earns, pd.DataFrame) and not earns.empty:
        try:
            earns = earns.copy()
            earns.index = pd.to_datetime(earns.index).date
            eps_fig.add_trace(go.Bar(x=earns.index, y=earns.get("epsActual"), name="EPS Actual"))
            eps_fig.add_trace(go.Bar(x=earns.index, y=earns.get("epsEstimate"), name="EPS Estimate"))
            eps_fig.update_layout(
                title={"text":"<b>Actual vs Estimate EPS<b>", "x":.5, "y":.95},
                xaxis_title="<b>Date<b>", yaxis_title="<b>EPS<b>", barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)
            )
        except Exception:
            pass

    # Forecast Table
    if cal:
        forecast_ticker = pd.DataFrame(cal).T
        if not forecast_ticker.empty:
            forecast_ticker.rename(columns={0:'Projected Forecast'}, inplace=True)
            forecast_ticker.reset_index(inplace=True)
            forecast_ticker.rename(columns={'index':'Forecast Type'}, inplace=True)
            forecast_data = forecast_ticker.to_dict('records')
            forecast_cols = [{'name': col, 'id': col} for col in forecast_ticker.columns]
        else:
            forecast_data, forecast_cols = [], []
    else:
        forecast_data, forecast_cols = [], []

    forecast_table = dash_table.DataTable(
        columns=forecast_cols,
        data=forecast_data,
        style_table={'overflowY': 'auto','minHeight':'100%'},
        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', "color":"black"},
        style_header={'backgroundColor': 'black', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'}
    )

    # RSI Figure
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=rs.index, y=rs["RSI"], name="RSI"))
    rsi_fig.update_layout(
        title={"text":"<b>RSI<b>", "x":.5, "y":.95},
        xaxis_title="<b>DATE<b>", yaxis_title="<b>RSI<b>",
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
    macd_fig.update_layout(title={"text":"<b>MACD<b>", "x":.5, "y":.95}, yaxis_title="<b>MACD<b>", xaxis_title="<b>DATE<b>",
                           legend=dict(orientation="h",yanchor="bottom",y=-.03,xanchor="center",x=0.5))

    # Trend line
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=f"{ticker} Close"))
    trend_fig.add_trace(go.Scatter(x=df.index, y=df["50_DAY_MA"], name="50-Day MA"))
    trend_fig.add_trace(go.Scatter(x=df.index, y=df["200_DAY_MA"], name="200-Day MA"))
    trend_fig.update_layout(title={"text":"<b>Closing Price Trend<b>"}, xaxis_title="<b>DATE<b>", yaxis_title="<b>PRICE<b>",
                            legend=dict(orientation="h",yanchor="bottom",y=-.03,xanchor="center",x=0.5))

    # Distribution Table
    df_close_dist = df['Close'].describe().to_frame(name='Closing Price').round(2)
    df_close_dist.reset_index(inplace=True)
    df_close_dist.rename(columns={'index':'Statistics'}, inplace=True)
    
    df_volume_dist = df['Volume'].describe().to_frame(name='Volume').round(2)
    df_volume_dist.reset_index(inplace=True)
    df_volume_dist.rename(columns={'index':'Statistics'}, inplace=True)
    
    df_close_dist = pd.concat(objs=[df_close_dist, df_volume_dist[['Volume']]], axis=1, join='inner')
    
    # --- CRITICAL FIX: Safe indexing using Column Names ---
    # This replaces the crash-prone iloc[-1,[0,4]]
    previous_data = df.iloc[[-1]][['Close', 'Volume']].copy()
    
    previous_data.rename(index={previous_data.index[0]:'previous data'}, inplace=True)
    previous_data.reset_index(inplace=True)
    previous_data.rename(columns={'index':'Statistics', 'Close':'Closing Price'}, inplace=True)
    previous_data['Closing Price'] = previous_data['Closing Price'].round(2)
    
    df_close_dist = pd.concat(objs=[df_close_dist, previous_data], axis=0, ignore_index=True)
    
    # Format volume safely
    df_close_dist['Volume'] = df_close_dist['Volume'].apply(lambda x: f"{x:.2e}" if pd.notnull(x) else x)

    dist_table = dash_table.DataTable(
        columns=[{'name':col, 'id':col} for col in df_close_dist.columns],
        data=df_close_dist.to_dict('records'),
        style_table={'overflowY': 'auto', 'overflowX':'auto'},
        style_cell={'textAlign': 'left', 'whitespace':'normal', "color":"black"},
        style_header={'backgroundColor': 'black', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'}
    )

    return news_table, financials_table, trend_fig, rsi_fig, forecast_table, eps_fig, macd_fig, dist_table