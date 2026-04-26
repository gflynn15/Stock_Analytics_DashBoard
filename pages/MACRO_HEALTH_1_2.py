# %% [markdown]
# - Macro Economic Page
#     - This page will be dedicated to exploring macro economic dta that impacts all market performances
#     - Data we are interested in starting with
#         1. Jobs reports
#         2. consumer price index
#         3. GDP
#         4. Institute for supply management 
#         5. The Beige book

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
from plotly.subplots import make_subplots
import openpyxl
import threading
import sys
import os
import statsmodels
import statsmodels.api as sm
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
import psycopg2
from dotenv import load_dotenv
##Importing cloud url
load_dotenv()
render_url = os.getenv("render_db_url")

                                        ###Importing list of stocks###
engine = create_engine(render_url)

with engine.connect() as conn:
    stock_symbols = pd.read_sql(text('SELECT DISTINCT "COMPANY" FROM "HISTORICAL_STOCK_PRICES"'), con=conn)
    stock_symbols_list = stock_symbols["COMPANY"].tolist()
    macros_symbols = pd.read_sql(text('SELECT DISTINCT "LEADING_INDICATOR" FROM "MACRO_INDICATOR_VALUES"'), con=conn)
    macros_symbols_list = macros_symbols["LEADING_INDICATOR"].tolist()

stock_symbols_list.extend(macros_symbols_list)

##Creating a dictionary to store the values of the macro economic indicators
leading_indicators_list = ["GDP",
                            "INDUSTRIAL PRODUCTION",
                            "RETAIL SALES",
                            "BUSINESS INVENTORIES",
                            "UNEMPLOYEMENT",
                            "TOTAL NONFARM PAYROLLS",
                            "INITIAL UNEMPLOYMENT CLAIMS",
                            "CPI",
                            "PCE",
                            "FEDERAL FUNDS EFFECTIVE RATE",
                            "10-YEAR TREASURY YIELD",
                            "2-YEAR TREASURY YIELD",
                            "30-YEAR FIXED MORTAGAGE RATE",
                            "M2 MONEY SUPPLY",
                            "RECESSION INDICATOR"
                        ]
    
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
                            ###Adding in period and interval drop down list for the scatter plot###
period = ["W","M","3M","1Y", "2Y","3Y","5Y","YTD","MAX"]
interval = ["D", "W", "M", "Q", "Y"]
                            ###Creating SQL Query Function for Data Extraction###
def data_query(metrics_list, period, interval):
    if not isinstance(metrics_list, list):
        metrics_list = [metrics_list]
    macro_list = []
    assets_list = []
    for x in metrics_list:
        if x in leading_indicators_list:
            macro_list.append(x)
        else:
            assets_list.append(x)
            
    macro_query_list = "'"+"','".join(macro_list)+"'"
    assets_query_list = "'"+"','".join(assets_list)+"'"
    
    if len(macro_list) > 0 and len(assets_list) > 0:
        try:
            with engine.connect() as conn:
                    macro_df = pd.read_sql(text(f'SELECT "DATE", "VALUE" AS "CLOSE", "LEADING_INDICATOR" AS "METRIC" FROM "MACRO_INDICATOR_VALUES" WHERE "LEADING_INDICATOR" in ({macro_query_list})'), con=conn)
                    stock_df = pd.read_sql(text(f'SELECT "DATE", "CLOSE", "COMPANY" AS "METRIC" FROM "HISTORICAL_STOCK_PRICES" WHERE "COMPANY" in ({assets_query_list})'), con=conn) 
                    summary_df = pd.concat([macro_df, stock_df], ignore_index=True, join="inner")
        except Exception as e:
            print(f"An error occurred while processing {e}")   
    elif len(macro_list) > 0 and len(assets_list) == 0:
        try:
            with engine.connect() as conn:
                summary_df = pd.read_sql(text(f'SELECT "DATE", "VALUE" AS "CLOSE", "LEADING_INDICATOR" AS "METRIC" FROM "MACRO_INDICATOR_VALUES" WHERE "LEADING_INDICATOR" in ({macro_query_list})'), con=conn)
        except Exception as e:
            print(f"An error occurred while processing {e}")
    elif len(macro_list) == 0 and len(assets_list) > 0:
        try:            
            with engine.connect() as conn:
                summary_df = pd.read_sql(text(f'SELECT "DATE", "CLOSE", "COMPANY" AS "METRIC" FROM "HISTORICAL_STOCK_PRICES" WHERE "COMPANY" in ({assets_query_list})'), con=conn)
        except Exception as e:
            print(f"An error occurred while processing {e}")
    summary_pivot = summary_df.pivot_table(index="DATE", columns="METRIC", values="CLOSE")
    summary_revised = summary_pivot.fillna(method="ffill")
    summary_revised.index = pd.to_datetime(summary_revised.index)
        ###Filtering the data based on the period selected by the user###
    latest_date = summary_revised.index.max()
    if period == "W":
        start_date = latest_date - pd.DateOffset(weeks=1)
    elif period == "M":
        start_date = latest_date - pd.DateOffset(months=1)
    elif period == "3M":
        start_date = latest_date - pd.DateOffset(months=3)
    elif period == "1Y":
        start_date = latest_date - pd.DateOffset(years=1)
    elif period == "2Y":
        start_date = latest_date - pd.DateOffset(years=2)
    elif period == "3Y":
        start_date = latest_date - pd.DateOffset(years=3)
    elif period == "5Y":
        start_date = latest_date - pd.DateOffset(years=5)
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

# %% [markdown]
# 1. application layout

# %%
##Establishign the application variable
#dash.register_page(__name__, name="Market_Review", path="/", order=1)

from flask import Flask
from flask_caching import Cache

## 1. Initialize Flask Server FIRST
server = Flask(__name__)

##Creating a cache file to store the data
cache = Cache(server, config={
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': 'MACRO_HEALTH_cache_file',
    'CACHE_DEFAULT_TIMEOUT': 300,
    'CACHE_THRESHOLD': 500
})


dash.register_page(__name__, name="Macro Health",path="/", order=1, external_stylesheets=[dbc.themes.CYBORG, "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"])
load_figure_template('cyborg')


layout = dbc.Container([
    ##Title of page
    dbc.Row([
        dbc.Col([
            html.H1("MACRO HEALTH", style={
                'textAlign': 'center', 
                'fontFamily': 'Inter', 
                'fontWeight': '900',
                'background': '-webkit-linear-gradient(45deg, #00f2fe, #4facfe)',
                '-webkit-background-clip': 'text',
                '-webkit-text-fill-color': 'transparent',
                'fontSize': '6rem',
                'letterSpacing': '5px'
            }),
            html.H3("MARKET INTELLIGENCE COMMAND", style={
                'textAlign': 'center', 
                'color': '#ffffff', 
                'fontFamily': 'Inter', 
                'fontWeight': '300', 
                'opacity': '0.8',
                'letterSpacing': '10px',
                'marginTop': '-10px'
            })
        ], width=12, className="d-flex flex-column justify-content-center align-items-center")
    ], className="macro-health-header animate__animated animate__fadeInDown"),
    
    html.Br(),
    
    ###====================================Instructional Text===================================###
    dbc.Row([
        dbc.Col([
            html.P([
                "Analyze macroeconomic trends and discover how key financial metrics drive broad market performance.",
                html.Br(),
                "Use the tools below to quantify the mathematical lead/lag relationships between economic indicators."
            ], style={"textAlign": "center", "fontSize":"24px", "fontFamily": "Inter", "color": "#e0e0e0"})
        ], width=10, className="mx-auto mb-4")
    ], className="animate__animated animate__fadeIn animate__delay-1s"),
    html.Br(),
    ###=======================================Main Controls Card============================================###
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("INDICATOR SELECTION", style={"fontSize":20, "fontWeight":"bold", "color":"#00f2fe"}),
                dcc.Dropdown(stock_symbols_list, 
                         ["^GSPC-SP500","INDUSTRIAL PRODUCTION","M2 MONEY SUPPLY"], 
                         id="metrics_list", multi=True, optionHeight=50,
                         style={"fontSize":"18px"})
            ], xs=12, lg=6),
            
            dbc.Col([
                html.Label("TIMEFRAME", style={"fontSize":20, "fontWeight":"bold", "color":"#00f2fe"}),
                dcc.Dropdown(period,"5Y",id="period1_drop",multi=False, style={"fontSize":"18px"})
            ], xs=6, lg=3),
            
            dbc.Col([
                html.Label("INTERVAL", style={"fontSize":20, "fontWeight":"bold", "color":"#00f2fe"}),
                dcc.Dropdown(interval, "D",id="interval1_drop",multi=False, style={"fontSize":"18px"})
            ], xs=6, lg=3) 
        ], className="mb-4"),
        
        ###Lag Feature Implementation for the heat map###
        dbc.Row([
            dbc.Col([
                html.Label("FEATURES TO LAG", style={"fontSize":20, "fontWeight":"bold", "color":"#fa709a"}),
                dcc.Dropdown(
                    id="lag_feature_selector", 
                    multi=True, 
                    placeholder="Select indicators to shift...",
                    style={"fontSize":"18px"}
                )
            ], xs=12, lg=6),
        
            dbc.Col([
                html.Label("LAG DEPTH (MONTHS)", style={"fontSize":20, "fontWeight":"bold", "color":"#fa709a"}),
                dcc.Slider(
                    id="lag_slider",
                    min=0, max=12, step=1, value=0,
                    marks={i: {'label': str(i), 'style': {'color': 'white'}} for i in range(0, 13)},
                    tooltip={"always_visible": True, "placement": "bottom"}
                )
            ], xs=12, lg=6)
        ])
    ]), style={"boxShadow": "0 8px 32px 0 rgba(0,0,0,0.8)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #333", "padding": "20px"}, 
    className="animate__animated animate__zoomIn mb-5"),
    
    ##----Heat Map Graph--##
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3("📊 CROSS-ASSET CORRELATION MATRIX", style={'textAlign': 'left', 'color': '#00f2fe', "marginBottom":"20px"}),
            dcc.Loading(dcc.Graph(id="heat_map", style={'height': '70vh'}), color="#00f2fe", type="default")
        ]), style={"boxShadow": "0 8px 16px 0 rgba(0,0,0,0.7)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #333"}), width=12)
    ], className="animate__animated animate__fadeInUp mb-5"),
    ##Space between heat map and scatter plot
    html.Br(),
    ##---------------------------------Input Drop downs for scatter and line chart-----------------------------------------##
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([html.H3("🛠️ REGRESSION CONTROLS", style={'textAlign': 'center', 'color': '#00f2fe', "fontWeight":"bold", "marginBottom":"20px"})], width=12),
            dbc.Col([
                html.Label("METRIC 1 (X)", style={"fontSize":18, "fontWeight":"bold"}),
                dcc.Dropdown(stock_symbols_list, "^GSPC-SP500", id="scatter&line_input1", multi=False, style={"fontSize":"16px"})
            ], xs=12, md=3), 
            dbc.Col([
                html.Label("METRIC 2 (Y)", style={"fontSize":18, "fontWeight":"bold"}),
                dcc.Dropdown(stock_symbols_list, "AMD-Advanced Micro Devices", id="scatter&line_input2", multi=False, style={"fontSize":"16px"})
            ], xs=12, md=3),
            dbc.Col([
                html.Label("PERIOD", style={"fontSize":18, "fontWeight":"bold"}),
                dcc.Dropdown(period, "5Y",id="period2_drop",multi=False, style={"fontSize":"16px"})
            ], xs=6, md=3),
            dbc.Col([
                html.Label("INTERVAL", style={"fontSize":18, "fontWeight":"bold"}),
                dcc.Dropdown(interval, "D",id="interval2_drop",multi=False, style={"fontSize":"16px"})
            ], xs=6, md=3)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Label("FEATURE TO LAG", style={"fontSize":18, "fontWeight":"bold", "color":"#fa709a"}),
                dcc.Dropdown(
                    id="lag_feature_selector2", 
                    multi=False, 
                    placeholder="Select indicators to shift...",
                    style={"fontSize":"16px"}
                ),
                html.Label("LAG DEPTH (MONTHS)", style={"fontSize":18, "fontWeight":"bold", "color":"#fa709a", "marginTop":"10px"}),
                dcc.Slider(
                    id="lag_slider2",
                    min=0, max=12, step=1, value=0,
                    marks={i: {'label': str(i), 'style': {'color': 'white'}} for i in range(0, 13)},
                    tooltip={"always_visible": True, "placement": "bottom"}   
                )
            ], xs=12, md=8),
            dbc.Col([
                html.Label("ROLLING WINDOW", style={"fontSize":18, "fontWeight":"bold", "color":"#fee140"}),
                dbc.Input(id="rolling_win", value=30, type="number", style={"fontSize":"16px", "backgroundColor":"#2a2a2c", "color":"white"})
            ], xs=12, md=4)
        ])
    ]), style={"boxShadow": "0 8px 16px 0 rgba(0,0,0,0.7)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #444", "padding":"15px"},
    className="animate__animated animate__fadeInUp mb-4"),
    
    ##-------------------------------------Scatter Plot with Lag Feature---------------------------------------------------##
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3("🎯 LINEAR REGRESSION", style={'textAlign': 'center', 'color': '#fa709a'}),
            dcc.Loading(dcc.Graph(id="scatter_plot", style={'height': '50vh'}), color="#fa709a", type="default")
        ]), style={"boxShadow": "0 8px 16px 0 rgba(0,0,0,0.7)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #333"}), xs=12, lg=6, className="animate__animated animate__fadeInLeft"),
        
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3("🔗 ROLLING CORRELATION", style={'textAlign': 'center', 'color': '#fee140'}),
            dcc.Loading(dcc.Graph(id="rolling_cor", style={'height': '50vh'}), color="#fee140", type="default")
        ]), style={"boxShadow": "0 8px 16px 0 rgba(0,0,0,0.7)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #333"}), xs=12, lg=6, className="animate__animated animate__fadeInRight")
    ], className="mb-5"),
    ##-------------------------------------Overlay Graphs--------------------------------------------------------##
    html.Br(),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3("📈 RAW DATA OVERLAY CHART", style={'textAlign': 'center', 'color': '#00f2fe', "marginBottom":"20px"}),
            dcc.Loading(dcc.Graph(id="overlay_chart", style={'height': '60vh'}), color="#00f2fe", type="default")
        ]), style={"boxShadow": "0 8px 16px 0 rgba(0,0,0,0.7)", "borderRadius": "15px", "backgroundColor": "#1a1a1c", "border": "1px solid #333"}), width=12)
    ], className="animate__animated animate__fadeInUp mb-5"),
    ##-------------------------------------Regression Analytics---------------------------------------------------##
    html.Br(),
    
    dbc.Row([
        html.H1("REGRESSION ANALYTICS", style={
            'fontFamily': 'Inter', 
            'textAlign': 'center',
            'fontWeight': '900',
            'background': '-webkit-linear-gradient(45deg, #00f2fe, #4facfe)',
            '-webkit-background-clip': 'text',
            '-webkit-text-fill-color': 'transparent',
            'paddingBottom': '20px',
            'paddingTop': '20px'
        })
    ], className="animate__animated animate__fadeInUp fundamentals-header mb-4"),
    
    html.Br(),
    
    # NEW: The container that will hold the 3 KPI cards
    dbc.Row(id="regression_cards", justify="center", className="mb-5")
    
], fluid=True)

###Call back Numer 1: Utilize this callback to create the heat map in the beginning of the page. That heatmap is u[dated using the first drop down in the page. 
@callback(
    Output("heat_map", "figure"),
    Input("metrics_list", "value"),
    Input("period1_drop","value"), 
    Input("interval1_drop", "value"),
    Input("lag_feature_selector", "value"),
    Input("lag_slider", "value")
)
def data_recovery_call(macros, period1, interval1, lag_features, lag_value):
    # 1. Handle empty inputs: if nothing is selected, return an empty figure
    if not macros:
        return go.Figure()
    else:
        indi_df = data_query(macros, period1, interval1).pct_change().dropna()
        
        ##Adding in lag feature logic and slider variable
        if lag_features and lag_value != 0:
            for feature in lag_features:
                if feature in indi_df.columns:
                    # Shifting the selected columns
                    indi_df[feature] = indi_df[feature].shift(lag_value)
            
            # Drop NaNs created by shifting so correlation remains valid
            indi_df = indi_df.dropna()
        
        for x in indi_df.columns:
                if x in leading_indicators_dict.keys():
                    indi_df.rename(columns={x:leading_indicators_dict[x][0]}, inplace=True)
                else:
                    indi_df.rename(columns={x:x.split("-")[0]}, inplace=True)
        ###---------------------------Creating heat map figure----------------------------###
        macro_heat = px.imshow(
            indi_df.corr().round(3), 
            text_auto=".2f",                 # Limits text to 2 decimal places for a cleaner look
            aspect="auto", 
            x=indi_df.columns, 
            y=indi_df.columns,
            zmin=-1,                         # Locks the minimum value to -1
            zmax=1                           # Locks the maximum value to 1
        )
        # ADD THIS LINE: Target the text inside the cells
        macro_heat.update_traces(textfont_size=25)

        # Update layout for a cleaner, modern look
        macro_heat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title={
                "text": "<b>CROSS-ASSET CORRELATION ANALYSIS</b>",
                "x": 0.5,
                "y": 0.95,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 24, "family": "Inter", "color": "#00f2fe"}
            },
            font=dict(family="Inter, sans-serif", size=14, color="#e0e0e0"),
            margin=dict(l=50, r=50, t=80, b=50),
            coloraxis_colorbar=dict(
                title="Correlation",
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                tickfont=dict(size=12)
            ),
            hoverlabel=dict(
                font_size=16,
                font_family="Inter",
                bgcolor="rgba(26, 26, 28, 0.9)"
            )
        )

        macro_heat.update_xaxes(title_text="", showgrid=False, tickangle=45)
        macro_heat.update_yaxes(title_text="", showgrid=False)
        return macro_heat
        
    # Fallback just in case
    return go.Figure()

###Lag feature callback for the heat map###
@callback(
    Output("lag_feature_selector", "options"),
    Input("metrics_list", "value")
)
def update_lag_dropdown(selected_metrics):
    if not selected_metrics:
        return []
    # This sets the options of the second dropdown to the values of the first
    return [{"label": x, "value": x} for x in selected_metrics]

##------------------------Call back number 2: Utilized to update the scatter plot, rolling correlation, Regression Cards------------------------##
@callback(
    Output("scatter_plot","figure"),
    Output("rolling_cor","figure"),
    Output("overlay_chart","figure"),
    Output("regression_cards", "children"),
    Input("scatter&line_input1","value"),
    Input("scatter&line_input2","value"),
    Input("period2_drop","value"),
    Input("interval2_drop","value"),
    Input("lag_feature_selector2", "value"),
    Input("lag_slider2", "value"),
    Input("rolling_win","value")
)
def scatter_line(metric1, metric2, period2, interval2, lag_features2, lag_value2, rolling_win):
    # 1. Handle empty inputs: if nothing is selected, return an empty figure
    if not metric1 or not metric2:
        return go.Figure(), go.Figure(), go.Figure(), []
    metrics_list = [metric1, metric2]
    indi_df = data_query(metrics_list, period2, interval2).pct_change().dropna()
    ## Adding in lag feature logic and slider variable
    if lag_features2 and lag_value2 != 0:
        # Ensure it's a list even if one item is selected
        features_to_process = lag_features2 if isinstance(lag_features2, list) else [lag_features2]
        
        for feature in features_to_process:
            # Clean the feature name to match the short ticker columns in indi_df
            #clean_ticker = feature.split("-")[0]
            
            if feature in indi_df.columns:
                # Shifting the selected column
                indi_df[feature] = indi_df[feature].shift(lag_value2)
        
        # Drop NaNs created by shifting so the OLS trendline doesn't crash
        indi_df = indi_df.dropna()
    
    # Drop NaNs created by the rolling window
    indi_df = indi_df.dropna()
    ###-------------------------------------------------Building The scatter plot-------------------------------------###
    scatter = px.scatter(data_frame=indi_df, x=metric1, y=metric2, trendline="ols")
    scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis={'title':{'text':f"<b>{metric1}<b>", 'font':{"size":16, "color":"#fa709a"}}, 'gridcolor':'#333'}, 
        yaxis={'title':{'text':f"<b>{metric2}<b>", 'font':{"size":16, "color":"#fa709a"}}, 'gridcolor':'#333'},
        font=dict(family="Inter, sans-serif", size=14, color="#e0e0e0"),
        hoverlabel=dict(
            font_size=16,
            font_family="Inter",
            bgcolor="rgba(26, 26, 28, 0.9)"
        ))
    scatter.update_traces(marker=dict(size=10, color='#fa709a', opacity=0.7, line=dict(width=1, color='white')))
    
        # --- THE SAFETY VALVE ---
    # If the user clears the input box, default it back to 30 to prevent a crash.
    # Also ensure it is treated as a solid integer.
    if not rolling_win or int(rolling_win) <= 1:
        rolling_win = 30
    else:
        rolling_win = int(rolling_win)
    # ------------------------
    # Use the cleaned tickers (ticker1, ticker2) instead of the full metric names
    rolling_df = data_query(metrics_list, period2, interval2).pct_change().dropna()
    rolling_df["ROLLING_COR"] = rolling_df[metric1].rolling(window=rolling_win).corr(rolling_df[metric2])
    rolling_df = rolling_df.dropna()
    
    ###-------------------------------------------------Building The Rolling correlation chart-------------------------------------###
    # Use indi_df.index for the x-axis to show the timeline
    rolling_cor_fig = px.line(
        data_frame=rolling_df, 
        x=rolling_df.index, 
        y="ROLLING_COR",
        title=f"<b>{rolling_win}-{interval2} Rolling Correlation</b>"
    )
    
    rolling_cor_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis={"title":{"text":"Date", "font":{"size":16, "color":"#fee140"}}, 'gridcolor':'#333'},
        yaxis={"title":{"text":"Correlation Coefficient", "font":{"size":16, "color":"#fee140"}}, 'gridcolor':'#333', 'range':[-1.1, 1.1]},
        font=dict(family="Inter, sans-serif", size=14, color="#e0e0e0"),
        hoverlabel=dict(
            font_size=16,
            font_family="Inter",
            bgcolor="rgba(26, 26, 28, 0.9)"
            ),
        hovermode="x unified"
        )
    rolling_cor_fig.update_traces(line=dict(color='#fee140', width=3))
    
    ###------------------------------------------------Building a overlay chart of the raw values of the metrics--------------------###
    raw_df = data_query(metrics_list, period2, interval2)
    
    ###-------------------Building out the graph---------------###
    # 1. Initialize the figure with a secondary y-axis
    overlay_fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 2. Add the first metric (Left Y-Axis)
    overlay_fig.add_trace(
        go.Scatter(
            x=raw_df.index, 
            y=raw_df[metric1], 
            name=metric1,
            mode='lines',
            line=dict(width=2, color='#00d2ff') # A bright cyan for contrast
        ),
        secondary_y=False,
    )

    # 3. Add the second metric (Right Y-Axis)
    overlay_fig.add_trace(
        go.Scatter(
            x=raw_df.index, 
            y=raw_df[metric2], 
            name=metric2,
            mode='lines',
            line=dict(width=2, color='#ff003c') # A sharp red to match your theme
        ),
        secondary_y=True,
    )

    # 4. Update the layout to match your Cyborg theme
    overlay_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=f"<b>RAW VALUE COMPARISON: {metric1} VS {metric2}</b>",
        template="cyborg",
        font=dict(family="Inter, sans-serif", size=14, color="#e0e0e0"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.4, 
            xanchor="center", x=0.5,
            font=dict(size=12)
        ),
        hoverlabel=dict(
            bgcolor="rgba(26, 26, 28, 0.9)",
            font_size=16,
            font_family="Inter"
        )
    )

    # 5. Label your axes
    overlay_fig.update_yaxes(title_text=f"<b>{metric1}</b>", secondary_y=False, tickfont=dict(color='#00d2ff'), gridcolor='#333')
    overlay_fig.update_yaxes(title_text=f"<b>{metric2}</b>", secondary_y=True, tickfont=dict(color='#ff003c'), gridcolor='#333')
    overlay_fig.update_xaxes(title_text="Date", gridcolor='#333')
    
    import statsmodels.api as sm

    ###---------------- Building The Regression Analytics KPI Cards ----------------###
    # Create a clean dataframe specifically for the regression to avoid NaN crashes
    reg_df = indi_df.dropna()

    # Safety check: Ensure we have enough data points to run a valid regression
    if len(reg_df) > 5:
        # Define our X (Independent Variable) and Y (Dependent Variable)
        X = sm.add_constant(reg_df[metric1]) 
        y = reg_df[metric2]
        
        # Fit the Ordinary Least Squares (OLS) model
        model = sm.OLS(y, X).fit()
        
        # Extract the metrics (rounded to 4 decimals for a clean UI)
        r_squared = round(model.rsquared, 4)
        beta = round(model.params[metric1], 4)
        p_value = round(model.pvalues[metric1], 4)
        
        # Format the P-Value text (Green for significant, Red for noise)
        p_color = "#02D302" if p_value < 0.05 else "#df0202"
        p_text = f"{p_value} (Valid Signal)" if p_value < 0.05 else f"{p_value} (Process Noise)"

        # Build the UI Cards using Bootstrap
        regression_ui = [
            dbc.Col(dbc.Card([
                dbc.CardHeader("R-SQUARED (ACCURACY)", className="text-center", style={"fontWeight": "900", "fontSize": "18px", "color": "#00d2ff"}),
                dbc.CardBody(html.H2(f"{r_squared}", className="text-center", style={"color": "white", "fontWeight": "900"}))
            ], style={"backgroundColor": "rgba(26, 26, 28, 0.6)", "borderRadius": "15px", "border": "1px solid #444", "boxShadow": "0 4px 15px rgba(0,0,0,0.5)"}), xs=12, md=4),
            
            dbc.Col(dbc.Card([
                dbc.CardHeader(f"BETA (SENSITIVITY)", className="text-center", style={"fontWeight": "900", "fontSize": "18px", "color": "#fee140"}),
                dbc.CardBody(html.H2(f"{beta}", className="text-center", style={"color": "white", "fontWeight": "900"}))
            ], style={"backgroundColor": "rgba(26, 26, 28, 0.6)", "borderRadius": "15px", "border": "1px solid #444", "boxShadow": "0 4px 15px rgba(0,0,0,0.5)"}), xs=12, md=4),
            
            dbc.Col(dbc.Card([
                dbc.CardHeader("P-VALUE (SIGNIFICANCE)", className="text-center", style={"fontWeight": "900", "fontSize": "18px", "color": "#fa709a"}),
                dbc.CardBody(html.H2(p_text, className="text-center", style={"color": p_color, "fontWeight": "900"}))
            ], style={"backgroundColor": "rgba(26, 26, 28, 0.6)", "borderRadius": "15px", "border": "1px solid #444", "boxShadow": "0 4px 15px rgba(0,0,0,0.5)"}), xs=12, md=4),
        ]
    else:
        # Fallback if a massive lag pushes all data off the chart
        regression_ui = [html.H5("Insufficient data to calculate regression.", style={"color": "white", "textAlign": "center"})]

    # FINAL RETURN: Make sure all 4 outputs are passed back to the app!
    return scatter, rolling_cor_fig, overlay_fig, regression_ui
    
### Consolidated Lag Feature Callback for the Scatter Plot ###
@callback(
    Output("lag_feature_selector2", "options"),
    Input("scatter&line_input1", "value"),
    Input("scatter&line_input2", "value")
)
def update_lag_dropdown_consolidated(metric1, metric2):
    if not metric1 and not metric2:
        return []
    
    # Filter out None and return options
    selected_metrics = [m for m in [metric1, metric2] if m]
    return [{"label": x, "value": x} for x in selected_metrics]
