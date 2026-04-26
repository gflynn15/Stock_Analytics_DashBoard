## Importing necessary functions
import numpy as np
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import html
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
render_url = os.getenv("render_db_url")
engine = create_engine(render_url)

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

    status_color = "#00ff88" if change >= 0 else "#ff3333" # Neon Green / Neon Red
    status_class = "success" if change >= 0 else "danger"
    
    card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # Left Side: Percent Change
                dbc.Col([
                    html.H6(change_header, style={"fontSize": "14px", "opacity": "0.6", "textTransform": "uppercase"}),
                    html.H2(f"{change:.2%}", style={"color": status_color, "fontWeight": "900", "fontSize": "30px"})
                ], width=6, style={"borderRight": "1px solid rgba(255,255,255,0.1)"}),
                
                # Right Side: Price
                dbc.Col([
                    html.H6(price_header, style={"fontSize": "14px", "opacity": "0.6", "textTransform": "uppercase"}),
                    html.H2(f"${price:,.2f}", style={"color": "white", "fontWeight": "900", "fontSize": "30px"})
                ], width=6)
            ])
        ])
    ], style={
        "backgroundColor": "rgba(255, 255, 255, 0.05)",
        "backdropFilter": "blur(15px)",
        "borderRadius": "15px",
        "border": f"1px solid {status_color}33", # Faint glowing border
        "boxShadow": f"0 4px 15px 0 {status_color}11",
        "marginBottom": "15px"
    }, className="animate__animated animate__fadeInUp")
    return card

# Function to create line chart (Crash-Proof Version)
def make_plot(df, ticker, title_text):
    df_columns = [x for x in df.columns if f"{ticker}" in x]
    df_ticker = df[df_columns]
    
    fig = px.line(data_frame=df_ticker,
        x=df_ticker.index, 
        y=df_ticker.columns, 
        title=f"<b>{title_text}</b>",
        render_mode="svg",
        template="cyborg"
    )
    
    fig.update_layout(
        title={"font":{"size":20, "family": "Inter", "color": "white"}},
        title_x=0.5,
        xaxis=dict(
            title="DATE",
            title_font=dict(size=16, family="Inter", color="rgba(255,255,255,0.6)"),
            tickfont=dict(size=12, color="rgba(255,255,255,0.8)"),
            gridcolor="rgba(255,255,255,0.05)"
            ),
        yaxis=dict(
            title="PRICE",
            title_font=dict(size=16, family="Inter", color="rgba(255,255,255,0.6)"),
            tickfont=dict(size=12, color="rgba(255,255,255,0.8)"),
            gridcolor="rgba(255,255,255,0.05)"
            ), 
        legend_title_text="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="white"),
            bgcolor="rgba(0,0,0,0)"
        ),
        hovermode="x unified",
        hoverlabel=dict(
            font_size=10, 
            font_family="Inter",
            bgcolor="rgba(30, 30, 30, 0.9)",
            bordercolor="rgba(255,255,255,0.1)"
        ),
        paper_bgcolor="rgba(0,0,0,0)", # Transparent background for glass look
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=80, b=80)
    )
    
    fig.update_xaxes(
        tickangle=0,
        automargin=True,
        showgrid=True
    )
    fig.update_yaxes(showgrid=True)
    
    # Modern smooth lines
    fig.update_traces(line=dict(width=3), hovertemplate="%{y:,.2f}")
    
    return fig

###=======================================Database Closing Price Query Function==========================================###
def data_query(metrics_list, period, interval):
    if not isinstance(metrics_list, list):
        metrics_list = [metrics_list]
    else:
        metrics_list = metrics_list
    assets_query_list = "'"+"','".join(metrics_list)+"'"
    try:
        with engine.connect() as conn:
            closing_prices = pd.read_sql(text(f"""SELECT "DATE", "CLOSE", "COMPANY" AS "METRIC" FROM "HISTORICAL_STOCK_PRICES" WHERE "COMPANY" in ({assets_query_list})"""), con=conn)
            summary_pivot = closing_prices.pivot_table(index="DATE", columns="METRIC", values="CLOSE")
            summary_revised = summary_pivot.ffill()
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
    
