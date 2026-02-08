## Importing necessary functions
import numpy as np
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import html

## Price change function (Robust Version)
def price_card_info(df, col):
    """
    Calculates percent change and current price safely.
    Handles empty dataframes and short history to prevent crashes.
    """
    if df is None or df.empty:
        return [0, 0]
    
    # Check if the column actually exists in the DataFrame
    if col not in df.columns:
        return [0, 0]

    # Drop NaNs to ensure we get the last VALID price (not a NaN)
    series = df[col].dropna()
    
    # If we don't have enough data for a comparison (need 2 days)
    if len(series) < 2:
        if len(series) == 1:
            return [0, series.iloc[-1]] # Return 0% change, current price
        return [0, 0]

    price = series.iloc[-1]
    prev_price = series.iloc[-2]
    
    # Prevent division by zero
    if prev_price == 0:
        price_change = 0
    else:
        price_change = (price - prev_price) / prev_price
        
    pricing_list = [price_change, price]
    return pricing_list

## Card design function
def make_card(change_header, change, price_header, price):
    # Fallback if inputs are None
    if change is None: change = 0
    if price is None: price = 0

    status_class = "success" if change >= 0 else "danger"
    
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

# Function to create line chart (Crash-Proof Version)
def make_plot(df, ticker, title_text):
    df_columns = [x for x in df.columns if f"{ticker}" in x]
    df_ticker = df[df_columns]
    fig = px.line(
        x=df_ticker.index, 
        y=df_ticker.columns, 
        title=f"<b>{title_text}</b>",
        render_mode="svg"
    )
    
    fig.update_layout(
        title_x=0.5, 
        legend_title_text="Trend Lines",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,  # Moved down slightly to prevent overlapping the X-axis
            xanchor="center",
            x=0.5                          
        ),
        margin=dict(l=20, r=20, t=40, b=50) # Added bottom margin for legend
    )
    return fig