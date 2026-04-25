# %%
###==========================================Library Imports=======================================================###
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
import sys
import os
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
from dotenv import load_dotenv
                                        ###Importing the Fred API key###
load_dotenv()
#fred_key = os.getenv("fred_api")
#from fredapi import Fred
                                ###Created a variable to execute the fred call###
#fred_trigger = Fred(api_key=fred_key)
# This finds the directory one level up from where this notebook is located
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Add that parent directory to the system path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from app_functions import make_plot
from app_functions import price_card_info
from app_functions import make_card
import warnings
warnings.filterwarnings("ignore") 


# %% [markdown]
# - Creating the database connection to the sqlite db
#     - This bd contains the closing price information
#     - Contains the fundamental data
#     - Contains the stock list

# %%
###=========================================Creating the SQLite dbconnection==========================================###
engine = create_engine("sqlite:///STOCK_DATA_WAREHOUSE.db")
conn=sqlite3.connect('STOCK_DATA_WAREHOUSE.db')

# %% [markdown]
# - Creating a dash table creation function

# %%
def dash_table_create(df):
    """
    Creates a styled Dash DataTable with dynamic column widths.
    Locks the first column to 70% width to ensure multiple tables align vertically.
    """
    if df is None or df.empty:
        return html.Div("No data available for this selection.", 
                        style={'color': 'white', 'fontSize': 20, 'textAlign': 'center', 'padding': '20px'})
    # --- DYNAMIC COLUMN ALIGNMENT LOGIC ---
    num_cols = len(df.columns)
    
    # 1. Lock the first column to 70%
    style_cell_conditional = [
        {
            'if': {'column_id': df.columns[0]},
            'width': '70%', 'minWidth': '70%', 'maxWidth': '70%',
        }
    ]
    
    # 2. Divide the remaining 30% among the rest of the columns
    if num_cols > 1:
        remaining_share = f"{30 / (num_cols - 1)}%"
        for col_id in df.columns[1:]:
            style_cell_conditional.append({
                'if': {'column_id': col_id},
                'width': remaining_share, 
                'minWidth': remaining_share, 
                'maxWidth': remaining_share,
            })
    # --- TABLE DEFINITION ---
    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("records"),
        
        # Behavior
        cell_selectable=True,
        sort_action="native",
        filter_action="native",
        
        # Alignment & Styling
        style_cell_conditional=style_cell_conditional,
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "border": "none",
            "borderRadius": "15px",
            "boxShadow": "0 8px 16px 0 rgba(0,0,0,0.5)",
        },
        style_data={
            "textAlign": "left",
            "whiteSpace": "normal",
            "color": "#e0e0e0",
            "backgroundColor": "#1a1a1c",
            "border": "1px solid #333",
            "fontSize": 22,
            "fontFamily": "Inter",
            "height":"auto"
        },
        style_header={
            "backgroundColor": "#FF4B2B",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "center",
            "fontSize": 26,
            "border": "none"
        },
    )
    return table


# %% [markdown]
# - Creating the extraction function for pulling closing price data from the sqlite database

# %%
def data_query(metrics_list, period, interval):
    if not isinstance(metrics_list, list):
        metrics_list = [metrics_list]
    else:
        metrics_list = metrics_list
    assets_query_list = "'"+"','".join(metrics_list)+"'"
    try:
        with engine.connect() as conn:
            closing_prices = pd.read_sql(text(f"SELECT DATE, CLOSE, COMPANY AS METRIC FROM HISTORICAL_STOCK_PRICES WHERE COMPANY in ({assets_query_list})"), con=conn)
            summary_pivot = closing_prices.pivot_table(index="DATE", columns="METRIC", values="CLOSE")
            summary_revised = summary_pivot.fillna(method="ffill")
            summary_revised.index = pd.to_datetime(summary_revised.index)
            latest_date = summary_revised.index.max()
        if period == "W":
            start_date = latest_date - pd.DateOffset(weeks=1)
        elif period == "M":
            start_date = latest_date - pd.DateOffset(months=1)
        elif period == "3M":
            start_date = latest_date - pd.DateOffset(months=7)
        elif period == "1Y":
            start_date = latest_date - pd.DateOffset(years=2)
        elif period == "2Y":
            start_date = latest_date - pd.DateOffset(years=3)
        elif period == "3Y":
            start_date = latest_date - pd.DateOffset(years=4)
        elif period == "5Y":
            start_date = latest_date - pd.DateOffset(years=6)
        elif period == "YTD":
            start_date = pd.to_datetime(f"{latest_date.year}-01-01")
        elif period == "MAX":
            start_date = summary_revised.index.min()
        else:
            start_date = summary_revised.index.min()
        summary_revised_filtered = summary_revised[summary_revised.index >= start_date]
            ###resampling the data based on the interval selected by the user###
        if interval == "D":
            summary_revised_filtered_resampled = summary_revised_filtered.resample("D").last()
        elif interval == "W":
            summary_revised_filtered_resampled = summary_revised_filtered.resample("W").last()
        elif interval == "M":
            summary_revised_filtered_resampled = summary_revised_filtered.resample("M").last()
        elif interval == "Q":
            summary_revised_filtered_resampled = summary_revised_filtered.resample("Q").last()
        elif interval == "Y":
            summary_revised_filtered_resampled = summary_revised_filtered.resample("Y").last()
        else:
            summary_revised_filtered_resampled = summary_revised_filtered
        summary_revised_filtered_resampled.dropna(axis=0, inplace=True)
        summary_final = summary_revised_filtered_resampled.round(2)
        return summary_final
    except Exception as e:
        print(f"Failed to Load {e}")

# %% [markdown]
# - Importing the stock list to add into the drop down menus
#     - List is derived from the sqlite database to ensure only stocks available in the sqlite database are available in the drop down menu
# - Importing the interval and period list for the drop down menus

# %%
###=========================Adding in a list of commodities and markets to prevent them from being pulled in to the drop down menus=========================================###
energy = ["CL=F-CrudeOil","BZ=F-BrentCrudeOil","NG=F-NaturalGas","HO=F-HeatingOil","RB=F-RBOBGasoline"]
metals = ["SI=F-Silver","PL=F-Platinum","PA=F-Palladium","GC=F-Gold"]
ag = ["ZC=F-Corn","ZW=F-Wheat","KC=F-Coffee","LE=F-LiveCattle","HE=F-LeanHogs","SB=F-Sugar"]
markets = [ '^GSPC-SP500','^DJI-DowJones','^IXIC-NasDaq']
symbols_removed_from_query = energy + metals + ag + markets
symbols_removed_query_preped = "'" + ("','").join(symbols_removed_from_query) + "'"

###==========================================================Symbols ist import==============================================================###
try:
    with engine.connect() as conn:
        stock_symbols = pd.read_sql(text(f"""SELECT DISTINCT COMPANY 
                                            FROM HISTORICAL_STOCK_PRICES
                                            WHERE COMPANY NOT IN ({symbols_removed_query_preped})"""), 
                                    con=conn)
        symbols = stock_symbols["COMPANY"].tolist()
except Exception as e:
    symbols = ['MMM-3M','AOS-A. O. Smith','ABT-Abbott Laboratories',
                        'ABBV-AbbVie','ACN-Accenture','ADBE-Adobe Inc.',
                        'AMD-Advanced Micro Devices','AES-AES Corporation','AFL-Aflac',
                        'A-Agilent Technologies','APD-Air Products']


###=========================================================List for date filters===============================================================###
period = ["W","M","3M","1Y", "2Y","3Y","5Y","YTD","MAX"]
intervals = ["D", "W", "M", "Q", "Y"]

# %%
#dash.register_page(__name__, name="Stock_Review", path="/Stock_Review", order=2)
#load_figure_template('simplex')

###========================================Initializing the application===============================================###
dash.register_page(__name__, name="COMPANY OVERVIEW", path="/Stock_Review_1_1", order=2, external_stylesheets=[dbc.themes.CYBORG, "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"])
load_figure_template('cyborg')



layout = dbc.Container([
    # --- Title Row ---
    dbc.Row([
        html.H1('COMPANY OVERVIEW', style={
            'textAlign': 'center', 
            'fontFamily': 'Inter', 
            'fontWeight': '900',
            'background': '-webkit-linear-gradient(45deg, #FF416C, #FF4B2B)',
            '-webkit-background-clip': 'text',
            '-webkit-text-fill-color': 'transparent',
            'paddingBottom': '20px',
            'paddingTop': '20px'
        })
    ], className="animate__animated animate__fadeInDown company-overview-header"),
    
    ###====================================Adding in the business summary===================================###
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Accordion([
                dbc.AccordionItem([
                            html.P(id="company-summary", className="animate__animated animate__fadeIn",
                            style={"fontSize":20, "textAlign":"center"})], 
                            title=html.Span("Click to Review the Company Summary", style={"fontSize":35}))],
                    start_collapsed=True)], 
                    width=12)], className="mb-4"),
    html.Br(),
    ###=======================================Drop Down Menu Row============================================###
    dbc.Row([
        dbc.Col([
            html.Label('Select Stock', style={"fontSize":35}),
            dcc.Dropdown(symbols, symbols[0], id='stock_symbols', clearable=False, style={"fontSize":25}),
        ], xs=12, md=4, lg=2),
        dbc.Col([
            html.Label('Select Period', style={"fontSize":35}),
            dcc.Dropdown(period, '1Y', id='period', clearable=False, style={"fontSize":25})
        ], xs=12, md=4, lg=2),
        dbc.Col([
            html.Label('Select Interval', style={"fontSize":35}),
            dcc.Dropdown(intervals, 'D', id='intervals', clearable=False, style={"fontSize":25})
        ], xs=12, md=4, lg=2)      
    ], className='mb-4'),

    ####===================================Price Trend Charts=================================###
    dbc.Row(dbc.Card(dbc.CardBody([
        dbc.Row([
            html.H3("📈 Pricing Trend", style={'textAlign': 'left', 'color': '#00f2fe', "fontSize":35})  
        ]),
        dbc.Row([dcc.Loading(
                id="loading-trend",
                type="default",
                children=dcc.Graph(
                    id='trend_line', 
                    config={'responsive': True},
                    style={'height': '60vh', 'minHeight': '500px'},
            ))
        ])
    ]), style={"boxShadow": "0 8px 16px 0 rgba(0,0,0,0.7)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #333", "padding": "10px"}), className="animate__animated animate__zoomIn"),
    html.Br(),
###====================================================RSI AND MACD LINE CHARTS=================================================###
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            dbc.Row([html.H3("📈 Relative Strength Index (RSI)", style={'textAlign': 'left', 'color': '#fa709a'})]),
            dbc.Row([
                    dcc.Loading(
                    id="loading-rsi",
                    type="default",
                    children=dcc.Graph(
                        id='RSI', 
                        config={'responsive': True},
                        style={'height': '40vh', 'minHeight': '350px'},
                    ))
                ])
            ]), style={"boxShadow": "0 8px 16px 0 rgba(0,0,0,0.7)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #333"}), xs=12, md=6, lg=6, className="animate__animated animate__fadeInLeft"),
        dbc.Col(dbc.Card(dbc.CardBody([
            dbc.Row([html.H3("📈 MACD Signals", style={'textAlign': 'left', 'color': '#fee140'})]),
            dbc.Row([
                    dcc.Loading(
                    id="loading-macd",
                    type="default",
                    children=dcc.Graph(
                        id='MACD', 
                        config={'responsive': True},
                        style={'height': '40vh', 'minHeight': '350px'}
                    ))
                ])
            ]), style={"boxShadow": "0 8px 16px 0 rgba(0,0,0,0.7)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #333"}), xs=12, md=6, lg=6, className="animate__animated animate__fadeInRight")
        ], className='mb-4'),
    html.Br(),
###===================================================NEWS SECTION======================================================###
    dbc.Row(dbc.Card(dbc.CardBody([
        dbc.Row(html.H3("📰 Latest News", style={'textAlign': 'left', 'color': '#FF4B2B'})),
        dbc.Row(html.Div(id='news_table'))
    ]), style={"boxShadow": "0 8px 16px 0 rgba(0,0,0,0.7)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #333"}), className='mb-4 animate__animated animate__fadeInUp'),

###=================================================Company Fundamentals================================================###
    html.Br(),
    dbc.Row([
        html.H1("COMPANY FUNDAMENTALS", style={
            "fontFamily":"Inter", 
            "textAlign":"center",
            'fontWeight': '900',
            'background': '-webkit-linear-gradient(45deg, #00f2fe, #4facfe)',
            '-webkit-background-clip': 'text',
            '-webkit-text-fill-color': 'transparent',
            'paddingBottom': '20px',
            'paddingTop': '20px'
        })
    ], className="animate__animated animate__fadeInUp fundamentals-header"),

    html.Br(),

    ###==============================================Fundamentals Tables=======================================================###
        ###Select company breakdown###
    dbc.Row([
        ###Company Risk Assesment###
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(id="company-risk")
        ]), style={"boxShadow": "0 8px 16px 0 rgba(0,0,0,0.7)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #333", "padding":"10px"}), xs=12, md=6, lg=4, className="animate__animated animate__zoomIn"),
        
        ###Company Financials##
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(id="profitability-performance")
        ]), style={"boxShadow": "0 8px 16px 0 rgba(0,0,0,0.7)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #333", "padding":"10px"}), xs=12, md=6, lg=4, className="animate__animated animate__zoomIn animate__delay-1s"),
        
        ###Company Performance###
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(id="financial-strength")
        ]), style={"boxShadow": "0 8px 16px 0 rgba(0,0,0,0.7)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #333", "padding":"10px"}), xs=12, md=6, lg=4, className="animate__animated animate__zoomIn animate__delay-2s"),
    ], align="start")
    ###Industry Average Breakdown###

], fluid=True)



# %%
@callback(
    Output("news_table", "children"),
    Output("trend_line", "figure"),
    Output("RSI", "figure"),
    Output("MACD", "figure"),
    Output("company-summary","children"),
    Output("company-risk","children"),
    Output("profitability-performance","children"),
    Output("financial-strength","children"),
    #Output("industry-risk","children"),
    Input("stock_symbols", "value"),
    Input("period", "value"),
    Input("intervals", "value"),
)
def trend_chart(ticker: str, period: str, intervals: str):
    news_query_conversion = ticker.split("-")[0]
    with engine.connect() as conn:
        articles_df = pd.read_sql(text(f"SELECT * FROM STOCK_NEWS_TABLE WHERE COMPANY = '{news_query_conversion}'"), con=conn)
    articles_df.drop(columns=["index"], inplace=True)
    articles_df["PUBDATE"] = pd.to_datetime(articles_df["PUBDATE"]).dt.date 
    articles_df.sort_values(by="PUBDATE",ascending=False, inplace=True)
    articles_df = articles_df.iloc[:-15,:]

    ##Creating the dash table
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
            style_table={
                "overflowX": "auto",
                "overflowY": "auto",
                "border": "none",
                "borderRadius": "15px",
                "boxShadow": "0 8px 16px 0 rgba(0,0,0,0.5)"
            },
            style_data={
                "textAlign": "left",
                "whiteSpace": "normal",
                "color":"#e0e0e0",
                "backgroundColor": "#1a1a1c",
                "border": "1px solid #333",
                "fontSize":22,
                "fontFamily":"Inter"
            },
            style_header={
                "backgroundColor": "#FF4B2B",
                "color": "white",
                "fontWeight": "bold",
                "textAlign": "center",
                "fontSize":26,
                "border": "none"
            },
        )


###===========================================FETCHing PRICING DATA FROM THE DB====================================================###
    with engine.connect() as conn:
        df = data_query(ticker, period, intervals).dropna(axis=0)
    # Clean Logic Removed: We handled cleaning inside get_price_data already!
    df.rename(columns={f"{ticker}":"Close"}, inplace=True)
    # Moving averages
    df["50_DAY_MA"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["200_DAY_MA"] = df["Close"].rolling(window=200, min_periods=1).mean()

###==============================================RSI CALCULATIONS==============================================###
    rs = df[["Close"]].copy()
    rs["Δ"] = rs["Close"].diff()
    gains = rs["Δ"].clip(lower=0)
    losses = (-rs["Δ"].clip(upper=0))
    rs["avg_gain"] = gains.ewm(com=14 - 1, adjust=False).mean()
    rs["avg_loss"] = losses.ewm(com=14 - 1, adjust=False).mean()
    rs["RSI"] = 100 - (100 / (1 + (rs["avg_gain"] / rs["avg_loss"].replace(0, np.nan))))
    rs["RSI"] = rs["RSI"].round(2)

###================================================RSI Figure====================================================###
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=rs.index, y=rs["RSI"], name="RSI", line=dict(color='#fa709a', width=3)))
    rsi_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        title={"text":f"<b>{ticker}-RSI<b>", 
                #"x":.5, 
                "y":.95, 
                "font": dict(color="#fa709a",size=18,family='Inter')
        },
        xaxis=dict(
                title="DATE",
                title_font=dict(size=25),
                tickfont=dict(size=18)
                ), 
        yaxis=dict(
                title="PRICE",
                title_font=dict(size=25),
                tickfont=dict(size=18),
                range=[0, 100]
                ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.20,
            xanchor="center",
            x=0.5,
            font=dict(size=18)
                    ),
        shapes=[
            dict(type="line", yref="y", y0=70, y1=70, xref="x", x0=df.index[0], x1=df.index[-1]),
            dict(type="line", yref="y", y0=30, y1=30, xref="x", x0=df.index[0], x1=df.index[-1]),
        ],
    )

###==================================================MACD========================================================###
    macd_df = df[["Close"]].copy()
    macd_df["EMA_12"] = macd_df["Close"].ewm(span=12, adjust=False).mean()
    macd_df["EMA_26"] = macd_df["Close"].ewm(span=26, adjust=False).mean()
    macd_df["MACD_LINE"] = macd_df["EMA_12"] - macd_df["EMA_26"]
    macd_df["SIGNAL_LINE"] = macd_df["MACD_LINE"].ewm(span=9, adjust=False).mean()
    macd_df["HISTOGRAM"] = macd_df["MACD_LINE"] - macd_df["SIGNAL_LINE"]

    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["MACD_LINE"], name="MACD", line=dict(color='#00f2fe', width=2)))
    macd_fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["SIGNAL_LINE"], name="Signal", line=dict(color='#f093fb', width=2)))
    macd_fig.add_trace(go.Bar(x=macd_df.index, y=macd_df["HISTOGRAM"], name="Histogram", marker_color='#fee140'))
    macd_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        title={"text":f"<b>{ticker}-MACD<b>", 
                #"x":.5, 
                "y":.95, 
                "font": dict(color="#fee140")}, 
        xaxis=dict(
                title="DATE",
                title_font=dict(size=25),
                tickfont=dict(size=18)
                ), 
        yaxis=dict(
                title="MACD",
                title_font=dict(size=25),
                tickfont=dict(size=18)
                ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.30,
            xanchor="center",
            x=0.5,
            font=dict(size=18)
                    )
                        )

###==============================================Trend line========================================================###
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=f"{ticker} Close", line=dict(color='#00f2fe', width=3)))
    trend_fig.add_trace(go.Scatter(x=df.index, y=df["50_DAY_MA"], name="50-Day MA", line=dict(color='#4facfe', width=2, dash='dot')))
    trend_fig.add_trace(go.Scatter(x=df.index, y=df["200_DAY_MA"], name="200-Day MA", line=dict(color='#f093fb', width=2, dash='dash')))
    trend_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        title={"text":f"<b>{ticker}-Closing Price Trend<b>", 
                "font":{'color':"#00f2fe",
                        'size':20,
                        'family': 'Inter'
                        }
                }, 
                xaxis=dict(
                    title="DATE",
                    title_font=dict(size=25),
                    tickfont=dict(size=18)
                ), 
                yaxis=dict(
                    title="PRICE",
                    title_font=dict(size=25),
                    tickfont=dict(size=18)
                ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.20,
            xanchor="center",
            x=0.5,
            font=dict(size=18)
                    ),
                        )

###=============================================FUNDAMENTALS BREAKDOWN==============================================###
    company = ticker.split("-")[0]
    with engine.connect() as conn:
        fundamentals = pd.read_sql(text(f"""SELECT * FROM FUNDAMENTAL_DATA WHERE "index" = '{company}'"""), con=conn).rename(columns={"index":"Company"})
        fundamentals.columns = fundamentals.columns.str.upper()
        sector = fundamentals["SECTOR"][0]
        sector_fundamentals = pd.read_sql(text(f"""SELECT * FROM FUNDAMENTAL_DATA WHERE sector = '{sector}'"""), con=conn).rename(columns={"index":"Company"})
        sector_fundamentals.columns = sector_fundamentals.columns.str.upper()
    ###Business Summary text###
    company_summary_text = fundamentals["LONGBUSINESSSUMMARY"]

    if not company_summary_text.empty:
        company_summary_display = company_summary_text.iloc[0]
    else:
        company_summary_display = "Company summary is unavailable. Please refer to the company website for more detials about their business"

    ###Risk Assesment Table###
    risk_analysis = fundamentals.loc[:,(fundamentals.columns.str.contains("RISK") | fundamentals.columns.str.contains("BETA"))].T
    risk_analysis.reset_index(inplace=True)
    risk_analysis.rename(columns={"index":"RISK ASSESMENT TYPE", 0:"RISK SCORE"}, inplace=True)

        ###Risk Assesment Industry###
    sector_risk_analysis = sector_fundamentals.loc[:,(sector_fundamentals.columns.str.contains("RISK") | sector_fundamentals.columns.str.contains("BETA"))].T
    sector_risk_analysis["INDUSTRY AVERAGE"] = (sector_risk_analysis.mean(axis=1)).round(2)
    sector_risk_analysis.reset_index(inplace=True)
    sector_risk_analysis.rename(columns={"index":"RISK ASSESMENT TYPE"}, inplace=True)
    sector_risk_summary = sector_risk_analysis[["RISK ASSESMENT TYPE","INDUSTRY AVERAGE"]]

    risk_analysis = risk_analysis.merge(sector_risk_summary, how="inner", on="RISK ASSESMENT TYPE")
    risk_table = dash_table_create(risk_analysis)

        ###Function created to create the company specific tables and the industry average tables
    def table_create_function(cols, title):
        df1 = fundamentals.loc[:,cols].T
        df1.dropna(axis=0, inplace=True)
        df1.reset_index(inplace=True)
        df1.rename(columns={"index":f"{title}", 0:"Value"}, inplace=True)
        df2 = sector_fundamentals.loc[:,cols].T
        #df2.dropna(axis=0, inplace=True)
        df2.rename(columns={"index":f"{title}"}, inplace=True)
        df2_avg = df2.mean(1)
        df2_avg.name = "Industry Average"
        df2_avg = df2_avg.to_frame()
        df2_avg.reset_index(inplace=True)
        df2_avg.rename(columns={"index":f"{title}"}, inplace=True)

        df2_table = df1.merge(df2_avg, on=f"{title}", how="left")

        df2_table['Industry Average'] = df2_table['Industry Average'].map("{:,.2f}".format)
        df2_table["Value"] = df2_table["Value"].map("{:,.2f}".format)
        
        return df2_table
        ###Profitability Table
    profitability_columns = [
                            'PROFITMARGINS', 'GROSSMARGINS', 'EBITDAMARGINS', 'OPERATINGMARGINS', 
                            'RETURNONASSETS', 'RETURNONEQUITY', 'EBITDA', 'GROSSPROFITS'
                            ]
    profitability_table = dash_table_create(table_create_function(profitability_columns,"Profitability Metrics"))

        ###Financial Strength Table
    financial_strength_cols = [
                            'TOTALCASH', 'TOTALCASHPERSHARE', 'TOTALDEBT', 'QUICKRATIO', 
                            'CURRENTRATIO', 'DEBTTOEQUITY', 'TOTALREVENUE', 'BOOKVALUE']
    financial_strength_table = dash_table_create(table_create_function(financial_strength_cols, "Financial Strength"))

    return news_table, trend_fig, rsi_fig, macd_fig, company_summary_display, risk_table, profitability_table, financial_strength_table

# %%
##Running the application
if __name__ == "__main__":
    app.run(jupyter_mode='external',debug=True)

# %% [markdown]
# #with engine.connect() as conn:
# #    fundamentals = pd.read_sql(text("""SELECT * FROM FUNDAMENTAL_DATA WHERE "index" = 'MMM'"""), con=conn)#.rename(columns={"index":"Company"})
# #fundamentals.columns = fundamentals.columns.str.upper()
# #help(pd.DataFrame.set_index)
# ticker = symbols[0].split("-")[0]
# with engine.connect() as conn:
#     fundamentals = pd.read_sql(text(f"""SELECT * FROM FUNDAMENTAL_DATA WHERE "index" = '{ticker}'"""), con=conn).rename(columns={"index":"Company"})
#     fundamentals.columns = fundamentals.columns.str.upper()
#     sector = fundamentals["SECTOR"][0]
#     sector_fundamentals = pd.read_sql(text(f"""SELECT * FROM FUNDAMENTAL_DATA WHERE sector = '{sector}'"""), con=conn).rename(columns={"index":"Company"})
#     sector_fundamentals.columns = sector_fundamentals.columns.str.upper()
# 
# risk_analysis = fundamentals.loc[:,(fundamentals.columns.str.contains("RISK") | fundamentals.columns.str.contains("BETA"))].T
# risk_analysis.reset_index(inplace=True)
# risk_analysis.rename(columns={"index":"RISK ASSESMENT TYPE", 0:"RISK SCORE"}, inplace=True)
# 
# sector_risk_analysis = sector_fundamentals.loc[:,(sector_fundamentals.columns.str.contains("RISK") | sector_fundamentals.columns.str.contains("BETA"))].T
# sector_risk_analysis["INDUSTRY AVERAGE SCORES"] = (sector_risk_analysis.mean(axis=1)).round(2)
# sector_risk_analysis.reset_index(inplace=True)
# sector_risk_analysis.rename(columns={"index":"RISK ASSESMENT TYPE"}, inplace=True)
# sector_risk_summary = sector_risk_analysis[["RISK ASSESMENT TYPE","INDUSTRY AVERAGE SCORES"]]
# 
# risk_analysis = risk_analysis.merge(sector_risk_summary, how="inner", on="RISK ASSESMENT TYPE")
# 
# profitability_columns = [
#                             'PROFITMARGINS', 'GROSSMARGINS', 'EBITDAMARGINS', 'OPERATINGMARGINS', 
#                             'RETURNONASSETS', 'RETURNONEQUITY', 'EBITDA', 'GROSSPROFITS'
#                             ]
#                             
# profitability_performance = fundamentals.loc[:,profitability_columns].T
# profitability_performance.reset_index(inplace=True)
# profitability_performance.dropna(axis=0,inplace=True)
# profitability_performance.rename(columns={"index":"Profitability Metric", 0:"Value"}, inplace=True)
# #profitability_table = dash_table_create(profitability_performance)
# 
# sector_fund = sector_fundamentals.loc[:,profitability_columns].round(2).T
# #sector_fund.dropna(axis=0, inplace=True)
# sector_fund.rename(columns={"index":"Profitability Metric"}, inplace=True)
# sector_avg = sector_fund.mean(1)
# sector_avg.name = "Industry Average"
# sector_avg = sector_avg.to_frame()
# sector_avg.reset_index(inplace=True)
# sector_avg.rename(columns={"index":"Profitability Metric"}, inplace=True)
# 
# profitability_table = profitability_performance.merge(sector_avg, on="Profitability Metric", how="left")
# 
# profitability_table['Industry Average'] = profitability_table['Industry Average'].map("{:,.2f}".format)
# profitability_table["Value"] = profitability_table["Value"].map("{:,.2f}".format)
# profitability_table


