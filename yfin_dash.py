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

##Visual Adjustments
# Set the display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)  # Prevent line wrapping

# %% [markdown]
# - Utilizing polygon io to provide a comprehensive list of stock symbols

# %%
import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Send a request to fetch the page content
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find the first table on the page (where the S&P 500 companies are listed)
table = soup.find('table', {'id': 'constituents'})

# Extract the table headers
headers = [th.text.strip() for th in table.find_all('th')]

# Extract table rows
rows = []
for tr in table.find_all('tr')[1:]:  # Skip the header row
    cells = [td.text.strip() for td in tr.find_all(['td', 'th'])]
    rows.append(cells)

# Create DataFrame
symbols = []

for x in range(len(rows)):
    symbols.append(rows[x][0])
    
    


##

# %% [markdown]
# - Creating he dash board frame work

# %%
app = dash.Dash(__name__)

app.layout = html.Div([
    ##Title of Page
    html.H1(
        children=['Hello and Welcome to my Dash Application'],
            style={'textAlign':'center'}),
    
    ##Creating stock Selection Dropdown Filter
    html.Div([
        dcc.Dropdown(symbols, 'AAPL',id='stock_symbols',
                     placeholder='Select A Stock',clearable=False,
                     )],style={'display':'inlineBlock', 'textAlign':'center',
                               'width':'100%'}),
    
    ##row 0 this row will have 2 columns
    ##News Table
    html.Div([
    ##This is the left side of the screen and it has the news table    
    html.Div(id='news_table', style={'width':'50%', 'flex':1, 'heigth':'100%'}),
    html.Div([dcc.Graph(id='trend_line')], style={'width':'50%', 'flex':1})
    ],style={'display':'flex', 'flexDirection':'row', 'flex':1}
             ),
    
    
    ##Creating the box plot distribution table
    html.Div([html.Div(id='price_dis_table', style={'width':'50%', 'flex':1}),
              html.Div([dcc.Graph(id='volume_trend_line')], style={'width':'50%', 'flex':1})
              ], style={'display':'flex', 'flexDirection':'row'})
    
    
    
    ],
##Style template for the overall template of the dashboard                      
                      style={'color':'black',
                             'backgroundColor':'silver'})

##Creating the call back for the news table
@callback(
    Output('news_table', 'children'),
    Input('stock_symbols','value'),
)
def create_table(ticker):
    
    stock = yf.Ticker(ticker)

    stock_json = stock.get_news(count=10, tab='news', proxy=None)

    stocks_dfs = {}

    for x in range(len(stock_json)):
        stocks_dfs[x] = stock_json[x]['content']


    stock_dictionary_list = list(stocks_dfs.values())

    news_articles = pd.DataFrame(stock_dictionary_list)

    keys_news_cols = news_articles[['title', 'summary', 'pubDate', 'canonicalUrl']]

    for x in list(range(0,10,1)):
        keys_news_cols['canonicalUrl'][x] = keys_news_cols['canonicalUrl'].iloc[x]['url']
    
    keys_news_cols.columns = keys_news_cols.columns.str.upper()

    
    return dash_table.DataTable(
        columns=[{'name':col, 'id':col} for col in keys_news_cols.columns],
        data=keys_news_cols.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left',
                    'whitespace':'normal',
                    'height':'auto',
                    'width':'100px',
                    'padding':'8px'}
    )
    
##Creating the trend line graph call back
@callback(
    Output('trend_line', 'figure'),
    Output('volume_trend_line', 'figure'),
    Input('stock_symbols','value'),
)


def trend_chart(ticker):
    from datetime import datetime, timedelta, date
    from polygon import RESTClient
    from dotenv import load_dotenv
    import os

    # Load environment variables from .env file
    load_dotenv()

    # Access your API key
    polygon_api_key = os.getenv("poly_api_key")

    client = RESTClient(api_key=polygon_api_key)
    today = pd.to_datetime(datetime.today()).date()
    
    yesterday = pd.to_datetime(datetime.today() - timedelta(days=1)).date()
    
    five_year = pd.to_datetime(yesterday - timedelta(days=3600)).date()
    
    # List Aggregates (Bars)
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="day",from_=five_year,to=yesterday, limit=50000):
        aggs.append(a)
    
    df = pd.DataFrame(aggs)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    
    ###Creating the moving average columns
    
    #Using pandas rolling function so the datetime column has to be set as the index
    df.set_index(keys='timestamp', drop=True, inplace=True)
    
    ##Creating the 50 day moving average column
    df['50_DAY_MA'] = df['close'].rolling(window=50).mean()
    ##Creating the 200 day moving average column
    df['200_DAY_MA'] = df['close'].rolling(window=200).mean()
    
    
##Creating the closing price trend chart
    stock_trend_line = go.Figure()
    stock_trend_line.add_trace(go.Scatter(x=df.index,y=df['close'],name=f"{ticker}_Closeing_Price"))
    ##Adding in the 50 day trace
    stock_trend_line.add_trace(go.Scatter(x=df.index, y=df['50_DAY_MA'], name=f'{ticker}_50_DAY_MA'))
    ##Adding in the 200 day moving average
    stock_trend_line.add_trace(go.Scatter(x=df.index, y=df['200_DAY_MA'], name=f'{ticker}_200_DAY_MA'))
    
    
    stock_trend_line.update_layout(title=str('Daily Closing Price Trend Chart Analysis').upper(),
                                   xaxis_title='Date',
                                   yaxis_title='Daily Closing Price',
                                   template='plotly_dark'
                                   )
    
    ##Adding in the volume chart to see if it will allow me to populate more than one chart in a single call back
    stock_volume_trend_line = go.Figure()
    stock_volume_trend_line.add_trace(go.Scatter(x=df.index,y=df['volume'],name=f"{ticker}_Volume"))
    #stock_trend_line.add_trace(go.Scatter(x=target_var['Date'],y=target_var[xaxis_col],name=xaxis_col))
    stock_volume_trend_line.update_layout(title=str('Volume Trend Chart Analysis').upper(),
                                   xaxis_title='Date',
                                   yaxis_title='Weekly Volume',
                                   template='plotly_dark'
                                   )
    
    
    return stock_trend_line, stock_volume_trend_line


##Creating the Distribution Table
##Creating the trend line graph call back
@callback(
    Output('price_dis_table', 'children'),
    Input('stock_symbols','value'),
)


def dist_table(ticker):
    from datetime import datetime, date, timedelta
    from polygon import RESTClient
    from dotenv import load_dotenv
    import os

    # Load environment variables from .env file
    load_dotenv()

    # Access your API key
    polygon_api_key = os.getenv("poly_api_key")

    client = RESTClient(api_key=polygon_api_key)
    
    today = datetime.today()
    
    yesterday = datetime.today() - timedelta(days=1)
    
    five_year = yesterday - timedelta(days=1800)
    
    # List Aggregates (Bars)
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="week",from_=five_year,to=yesterday):
        aggs.append(a)
    
    df = pd.DataFrame(aggs)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date

    df_close_dist = df['close'].describe()

    df_close_dist = df_close_dist.to_frame(name='Closing Price').round(2)
    
    df_close_dist.reset_index(inplace=True)
    
    df_close_dist.rename(columns={'index':'Statistics'}, inplace=True)
    
    df_volume_dist = df['volume'].describe()
    
    df_volume_dist = df_volume_dist.to_frame(name='Volume').round(2)
    
    df_volume_dist.reset_index(inplace=True)
    
    df_volume_dist.rename(columns={'index':'Statistics'}, inplace=True)
    
    df_close_dist = pd.concat(objs=[df_close_dist,df_volume_dist[['Volume']]], axis=1, join='inner',ignore_index=False)
    
    previous_data = df.iloc[-1,[3,4]].to_frame().transpose()

    previous_data.rename(index={previous_data.index[0]:'previous data'}, inplace=True)
    
    previous_data.reset_index(inplace=True)
    
    previous_data.rename(columns={'index':'Statistics',
                                  'close':'Closing Price',
                                  'volume':'Volume'}, inplace=True)
    
    df_close_dist = pd.concat(objs=[df_close_dist, previous_data], axis=0,ignore_index=True)
    
    df_close_dist['Volume'] = df_close_dist['Volume'].apply(lambda x: f"{x:.2e}")
    
    
    return dash_table.DataTable(
        columns=[{'name':col, 'id':col} for col in df_close_dist.iloc[1:,:].columns],
        data=df_close_dist.iloc[1:,:].to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left',
                    'whitespace':'normal',
                    'height':'auto',
                    'width':'100px',
                    'padding':'8px'}
    )
    
        

if __name__ == '__main__':
    app.run(debug=True)




