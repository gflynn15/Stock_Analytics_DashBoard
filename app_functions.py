##Importing necessary functions
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

##Price change function
def price_card_info(df, col):
        series = df[col]
        price_change = ((series.iloc[-1] - series.iloc[-2])/series.iloc[-2])
        price = series.iloc[-1]
        pricing_list = [price_change, price]
        return pricing_list
##Card design function
def make_card(change_header, change, price_header, price):
        status_class = "success" if change > 0 else "danger"
        card = dbc.CardGroup([
            dbc.Card(
                dbc.CardBody([
                html.H6(change_header, className="text-muted small"),
                html.H2(f"{change:.2%}", className=f"text-{status_class} mb-0")
                ])
                ),
            dbc.Card(
                dbc.CardBody([
                html.H6(price_header, className="text-muted small"),
                html.H2(f"${price:,.2f}", className=f"text-{status_class} mb-0")    
                ])
                )       
            ])
        return card
#Function to create line chart
def make_plot(df, ticker, title_text):
    # We grab the price column and the three MA columns for this specific ticker
    cols = [ticker, f"{ticker}_30_MA", f"{ticker}_50_MA", f"{ticker}_200_MA"]
        
    fig = px.line(
        df, 
        y=cols, 
        title=f"<b>{title_text}</b>",
        render_mode="svg"
    )
    fig.update_layout(title_x=0.5, legend_title_text="Trend Lines",
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=-.03,
                          xanchor="center",
                          x=0.5                          
                      ))
    return fig