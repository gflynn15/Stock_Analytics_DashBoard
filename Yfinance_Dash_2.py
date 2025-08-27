# %% [markdown]
# - Creating a new Dash App using the yfinance API as the data pipeline

# %% [markdown]
# - Importing the necessary libraries to execute the application

# %%
import numpy as np
import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import openpyxl
import warnings

warnings.filterwarnings("ignore")

symbols_df = pd.read_excel(io=r"Sp500Symbols.xlsx",
                        sheet_name='Table 1',engine='openpyxl')
symbols = symbols_df['Symbol'].to_list()

# %% [markdown]
# - Creating the list for date filters to be used in the line charts

# %%
period = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

# %% [markdown]
# - Creating the dashboard framework

# %%
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        # Title
        html.H1(
            "Welcome to the Stock Analytics Dashboard",
            style={"textAlign": "center", "padding": "20px", "color": "#1a1a1a"},
        ),

        # -------------------- Filters --------------------
        html.Div(
            [
                html.Div(
                    [html.Label("Select Stock"), dcc.Dropdown(symbols, "AAPL", id="stock_symbols", clearable=False)],
                    style={"minWidth": 0},
                ),
                html.Div(
                    [html.Label("Select Period"), dcc.Dropdown(period, "5y", id="period", clearable=False)],
                    style={"minWidth": 0},
                ),
                html.Div(
                    [html.Label("Select Interval"), dcc.Dropdown(intervals, "1d", id="intervals", clearable=False)],
                    style={"minWidth": 0},
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr 1fr",
                "gap": "12px",
                "padding": "0 20px 20px 20px",
                "alignItems": "center",
            },
        ),

        html.Hr(),

        # -------------------- Top: Trend | RSI+MACD --------------------
        html.Div(
            [
                # Left: Price Trend
                html.Div(
                    [
                        html.H3("📈 Stock Price Trend", style={"margin": "0 0 8px 0"}),
                        dcc.Graph(id="trend_line", style={"height": "520px"}),
                    ],
                    style={"minWidth": 0},
                ),

                # Right: RSI | MACD (side-by-side)
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("📈 Relative Strength Index (RSI)", style={"margin": "0 0 8px 0"}),
                                dcc.Graph(id="RSI", style={"height": "400px"}),
                            ],
                            style={"minWidth": 0},
                        ),
                        html.Div(
                            [
                                html.H3("📈 MACD Signal Line", style={"margin": "0 0 8px 0"}),
                                dcc.Graph(id="MACD", style={"height": "400px"}),
                            ],
                            style={"minWidth": 0},
                        ),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "20px",
                        "minWidth": 0,
                        "alignItems": "start",
                    },
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "20px",
                "padding": "20px",
                "alignItems": "start",
            },
        ),

        # -------------------- News --------------------
        html.Div(
            [
                html.H3("📰 Latest News", style={"margin": "0 0 8px 0"}),
                html.Div(id="news_table"),
            ],
            style={"padding": "20px"},
        ),

        html.Hr(),

        # ------------- Bottom: Distribution+Forecast | Financials+EPS -------------
        html.Div(
            [
                # Left column
                html.Div(
                    [
                        html.H3("📈 Price & Volume Distribution", style={"margin": "0 0 8px 0"}),
                        html.Div(id="price_dis_table", style={"minHeight": "0"}),
                        html.H3("🔮 Earnings Forecast", style={"margin": "20px 0 8px 0"}),
                        html.Div(id="forecast", style={"minHeight": "0"}),
                    ],
                    style={"display": "flex", "flexDirection": "column", "gap": "12px", "minWidth": 0},
                ),

                # Right column
                html.Div(
                    [
                        html.H3("📊 Historical Financials", style={"margin": "0 0 8px 0"}),
                        html.Div(id="historical_financials", style={"minHeight": "0"}),
                        html.H3("📊 Quarterly Earnings Per Share", style={"margin": "20px 0 8px 0"}),
                        dcc.Graph(id="EPS", style={"height": "400px"}),
                    ],
                    style={"display": "flex", "flexDirection": "column", "gap": "12px", "minWidth": 0},
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "20px",
                "padding": "20px",
                "alignItems": "start",   # important: prevents tall child from stretching the row
            },
        ),
    ],
    style={"backgroundColor": "#f5f5f5", "fontFamily": "Arial, sans-serif", "paddingBottom": "40px"},
)

@callback(
    Output("news_table", "children"),
    Output("historical_financials", "children"),
    Input("stock_symbols", "value"),
)
def create_table(ticker: str):
    stock = yf.Ticker(ticker)

    # News (robust parsing)
    news = stock.get_news(count=10, tab="news", proxy=None) or []
    # Pull nested content dicts if present
    records = []
    for item in news:
        content = item.get("content", {})
        # Some fields may be at top-level; prefer content then fallback
        title = content.get("title") or item.get("title")
        summary = content.get("summary") or item.get("summary")
        pubDate = content.get("pubDate") or item.get("pubDate")
        url = None
        cu = content.get("canonicalUrl") or item.get("canonicalUrl")
        if isinstance(cu, dict):
            url = cu.get("url")
        elif isinstance(cu, str):
            url = cu
        records.append({"TITLE": title, "SUMMARY": summary, "PUBDATE": pubDate, "URL": url})

    news_df = pd.DataFrame(records)
    articles = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in news_df.columns],
        data=news_df.to_dict("records"),
        cell_selectable=True,
        page_size=5,
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": "400px",
            "border": "1px solid #ccc",
        },
        style_cell={
            "textAlign": "left",
            "whiteSpace": "normal",
            "minWidth": "150px",
            "width": "200px",
            "maxWidth": "300px",
            "padding": "6px",
            "overflowX": "auto",
            "overflowY": "auto",
        },
        style_header={
            "backgroundColor": "black",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "left",
            "overflowX": "auto",
            "overflowY": "auto",
        },
    )

    # Historical financials (guard for empty)
    financials = stock.financials

    financials = financials.T

    financials.index = pd.to_datetime(financials.index).year

    key_financials = financials[['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income','Normalized Income']]

    key_financials.reset_index(inplace=True)

    key_financials.rename(columns={'index':'Year'}, inplace=True)
    
    financials_table = dash_table.DataTable(
        columns=[{'name':col, 'id':col} for col in key_financials.columns],
        data=key_financials.to_dict('records'),
        cell_selectable=True,
        style_table={
        'overflowX': 'auto',         # Scroll horizontally if needed
        'overflowY': 'auto',         # Scroll vertically inside the box
        'maxHeight': '400px',        # Set max visible height
        'border': '2px solid #ccc'   # Optional: makes the box look nice
    },
    
    style_cell={
        'textAlign': 'left',
        'whiteSpace': 'normal',
        'minWidth': '150px',
        'width': '200px',
        'maxWidth': '300px',
        'padding': '6px',
        'overflowX':'auto',
        'overflowY':'auto',
    },

    style_header={
        'backgroundColor': 'black',
        'color': 'white',
        'fontWeight': 'bold',
        'textAlign': 'left',
        'overflowX':'auto',
        'overflowY':'auto'
    }
)

    return articles, financials_table

# -------- Trend line, RSI, forecast, EPS, MACD --------
@callback(
    Output("trend_line", "figure"),
    Output("RSI", "figure"),
    Output("forecast", "children"),
    Output("EPS", "figure"),
    Output("MACD", "figure"),
    Input("stock_symbols", "value"),
    Input("period", "value"),
    Input("intervals", "value"),
)
def trend_chart(ticker: str, period: str, intervals: str):
    company = yf.Ticker(ticker)
    df = yf.download(tickers=ticker, period=period, interval=intervals)

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
                title="Actual vs Estimate Earnings per Share",
                xaxis_title="Date",
                yaxis_title="EPS",
                barmode="group",
                legend=dict(title="Legend", orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
            )
        except Exception:
            pass

    # RSI figure
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=rs.index, y=rs["RSI"], name="RSI"))
    rsi_fig.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Date",
        yaxis_title="RSI",
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
    macd_fig.update_layout(title="MACD")

    # Trend line
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=f"{ticker} Close"))
    trend_fig.add_trace(go.Scatter(x=df.index, y=df["50_DAY_MA"], name="50-Day MA"))
    trend_fig.add_trace(go.Scatter(x=df.index, y=df["200_DAY_MA"], name="200-Day MA"))
    trend_fig.update_layout(title="Daily Closing Price Trend", xaxis_title="Date", yaxis_title="Price")

    # Earnings calendar / forecast table (guard empty)
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
            'padding': '2px'
        },
        style_header={
            'backgroundColor': 'black',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        }
    )

    return trend_fig, rsi_fig, forecast_table, eps_fig, macd_fig

# -------- Distribution table --------
@callback(
    Output("price_dis_table", "children"),
    Input("stock_symbols", "value"),
)
def dist_table(ticker):

    df = yf.download(tickers=ticker, period='5y')
    
    cols_new = []

    for x in df.columns:
        new = x[0]
        cols_new.append(new)
    
    df.columns = cols_new
    
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
    
    
    return dash_table.DataTable(
        columns=[{'name':col, 'id':col} for col in df_close_dist.iloc[1:,:].columns],
        data=df_close_dist.iloc[1:,:].to_dict('records'),
        style_table={'overflowY': 'auto'},
        style_cell={'textAlign': 'left',
                    'whitespace':'normal',
                    'height':'auto',
                    'width':'100px',
                    'padding':'8px'},
        style_header={'backgroundColor': 'black',
                      'color': 'white',
                      'fontWeight': 'bold',
                      'textAlign': 'center'}
    )
if __name__ == "__main__":
    app.run_server(debug=False)


















