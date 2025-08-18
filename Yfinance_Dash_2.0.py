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
import warnings
warnings.filterwarnings("ignore")

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

# %% [markdown]
# - Creating the list for date filters to be used in the line charts

# %%
period = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
intervals = ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'] 

# %% [markdown]
# - Creating he dash board frame work

# %%
app = dash.Dash(__name__)

app.layout = html.Div([
    # Title
    html.H1('Welcome to the Stock Analytics Dashboard',
            style={'textAlign': 'center', 'padding': '20px', 'color': '#1a1a1a'}),

    # Dropdown Filters Row
    html.Div([
        html.Div([
            html.Label('Select Stock'),
            dcc.Dropdown(symbols, 'AAPL', id='stock_symbols', clearable=False)
        ], style={'flex': 1, 'marginRight': '10px'}),

        html.Div([
            html.Label('Select Period'),
            dcc.Dropdown(period, '5y', id='period', clearable=False)
        ], style={'flex': 1, 'marginRight': '10px'}),

        html.Div([
            html.Label('Select Interval'),
            dcc.Dropdown(intervals, '1d', id='intervals', clearable=False)
        ], style={'flex': 1})
    ], style={
        'display': 'flex', 'flexDirection': 'row',
        'justifyContent': 'space-between',
        'padding': '0px 20px 20px 20px'
    }),

    html.Hr(),

    # Main Graph Container for a 50/50 split
    html.Div([
        # Price Trend Line - Left half of the screen
        html.Div([
            html.H3("📈 Stock Price Trend", style={'textAlign': 'left'}),
            dcc.Graph(id='trend_line')
        ], style={'flex': '1 1 0%'}),

        # RSI and MACD graphs - Right half of the screen, stacked
        html.Div([
            # RSI Graph
            html.Div([
                html.H3("📈 Relative Strength Index (RSI)", style={'textAlign': 'left'}),
                dcc.Graph(id='RSI')
            ], style={'flex': '1 1 0%'}), # Use flex on the child to ensure it takes available space

            # MACD Graph
            html.Div([
                html.H3("📈 MACD Signal Line", style={'textAlign': 'left'}),
                dcc.Graph(id='MACD')
            ], style={'flex': '1 1 0%'}) # Use flex on the child to ensure it takes available space
        ],
        style={
            'display': 'flex',
            'flexDirection': 'row', # Stacks RSI and MACD vertically
            'flex': '1 1 0%',         # Allows this container to take up half the width
            'gap': '20px',            # Adds a consistent gap between the RSI and MACD graphs
        }),

    ], style={
        'display': 'flex',
        'flexDirection': 'column',  # Arranges the two main Divs horizontally
        'padding': '20px',
        'gap': '20px'            # Adds spacing between the two columns
    }),


    ##News Inquiry Table
    html.Div([
        html.Div([
            html.H3("📰 Latest News", style={'textAlign': 'left'}),
            html.Div(id='news_table')
        ], style={'flex': 1})
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'padding': '20px'
    }),

    html.Hr(),

    # Distribution and Forecast Table
    html.Div([
        
        html.Div([
            html.H3("📈 Price & Volume Distribution"),
            html.Div(id='price_dis_table'),
            html.H3("🔮 Earnings Forecast"),
            html.Div(id='forecast')
        ], style={'flex': 1, 'flexDirection':'column','marginRight': '20px'}),

        #html.Div([
        #    html.H3("🔮 Earnings Forecast"),
        #    html.Div(id='forecast')
        #], style={'flex': 1, 'marginRight': '20px'}),
        
        html.Div([html.H3("📊 Historical Financials"),
                  html.Div(id='historical_financials'),
                  html.H3(children='📊 Quarterly Earnings Per Share', style={',arginTop':'20px'}),
                  dcc.Graph(id='EPS', style={'marginTop': '20px'})
                  ], 
                 style={'flex': 1,'flexDirection':'column','marginRight': '20px'})
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'padding': '20px'
    })
], style={
    'backgroundColor': '#f5f5f5',
    'fontFamily': 'Arial, sans-serif',
    'paddingBottom': '40px'
})


##Creating the call back for the news table
@callback(
    Output('news_table', 'children'),
    Output('historical_financials', 'children'),
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
    
    keys_news_cols.rename(columns={'canonicalUrl':'URL'}, inplace=True)
    
    keys_news_cols.columns = keys_news_cols.columns.str.upper()

    articles = dash_table.DataTable(
        columns=[{'name':col, 'id':col} for col in keys_news_cols.columns],
        data=keys_news_cols.to_dict('records'),
        cell_selectable=True,
        page_size=5,
        style_table={
        'overflowX': 'auto',         # Scroll horizontally if needed
        'overflowY': 'auto',         # Scroll vertically inside the box
        'maxHeight': '400px',        # Set max visible height
        'border': '1px solid #ccc'   # Optional: makes the box look nice
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
    
    ##Creating the histroical financials table to be added into the bottom three tables
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
    
##Creating the trend line graph call back
@callback(
    Output('trend_line', 'figure'),
    Output('RSI', 'figure'),
    Output('forecast', 'children'),
    Output('EPS','figure'),
    Output('MACD','figure'),
    
    
    Input('stock_symbols','value'),
    Input('period','value'),
    Input('intervals','value')
)
def trend_chart(ticker, period, intervals):
    ##Creating a tick Module
    company = yf.Ticker(ticker)
    # Download historical stock data
    df = yf.download(tickers=ticker, period=period, interval=intervals)

    # Flatten multi-level column index if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Create moving average columns
    df['50_DAY_MA'] = df['Close'].rolling(window=50).mean()
    df['200_DAY_MA'] = df['Close'].rolling(window=200).mean()
    
    ##Creating the RSI Graph
    rs_data = df[['Close']]

    rs_data['Gains and Losses'] = rs_data['Close'].diff()
    
    rs_data['Gains Average'] = rs_data['Gains and Losses'].apply(lambda x: x if x > 0 else 0).ewm(com=14-1, adjust=False).mean()
    rs_data['Losses Average'] = rs_data['Gains and Losses'].apply(lambda x: -x if x < 0 else 0).ewm(com=14-1, adjust=False).mean()
    
    rs_data['RSI'] = 100 - (100 / (1 + rs_data['Gains Average'] / rs_data['Losses Average'].replace(0, np.nan)))

    rs_data['RSI'] = rs_data['RSI'].round(2)
    
    ##Earnings per share Bar Chart
    earns = company.earnings_history
    
    earns.index = earns.index.date
        
    ##Earnings bar chart
    eps_fig = go.Figure()
    eps_fig.add_trace(go.Bar(
        x=earns.index,
        y=earns['epsActual'],
        name='EPS Actual',
        marker_color='blue'
    ))
    eps_fig.add_trace(go.Bar(
        x=earns.index,
        y=earns['epsEstimate'],
        name='EPS Estimate',
       marker_color='red'
    ))
    eps_fig.update_layout(title='Actual vs Estimate Earnings per Share Comparison',
                          xaxis_title='Date',
                          yaxis_title='Earnings Per Share (EPS)',
                          barmode='group',
                          template='ggplot2',
                          xaxis=dict(tickformat='%Y-%m-%d'),
                          yaxis=dict(tickformat='.2f'),
                          legend=dict(title='Legend', orientation='h', yanchor='bottom', y=1.00, xanchor='right', x=1)
                            
                          )
    
    
    ##RSI Graph Creation
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=rs_data.index, y=rs_data['RSI'], name='RSI', line=dict(color='firebrick')))
    rsi_fig.update_layout(
        title='Relative Strength Index (RSI) Chart',
        xaxis_title='Date',
        yaxis_title='RSI',
        yaxis=dict(range=[0, 100]),
        template='plotly_dark',
        shapes=[
            dict(
                type='line',
                yref='y',
                y0=70,
                y1=70,
                xref='x',
                x0=df.index[0],
                x1=df.index[-1],
                line=dict(color='white', width=1, dash='solid')
            ),
            dict(
                type='line',
                yref='y',
                y0=30,
                y1=30,
                xref='x',
                x0=df.index[0],
                x1=df.index[-1],
                line=dict(color='white', width=1, dash='solid')
            )
        ],
        annotations=[
            dict(
                x=df.index[-1],
                y=70,
                text='Overbought',
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
                font=dict(color='red')
            ),
            dict(
                x=df.index[-1],
                y=30,
                text='Oversold',
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=40,
                font=dict(color='orange')
            )
        ]
    )
    
    ###MACD GRAPH CREATION
    df_macd = df[['Close']]

    df_macd['EMA_12'] = df_macd['Close'].ewm(span=12, adjust=False).mean()

    df_macd['EMA_26'] = df_macd['Close'].ewm(span=26, adjust=False).mean()

    df_macd['MACD_LINE'] = df_macd['EMA_12'] - df_macd['EMA_26']

    df_macd['SIGNAL_LINE'] = df_macd['MACD_LINE'].ewm(span=9, adjust=False).mean()

    df_macd['HISTOGRAM'] = df_macd['MACD_LINE'] - df_macd['SIGNAL_LINE']
    
        ##Creating the MACD chart
    macd_fig = go.Figure()

    macd_fig.add_trace(go.Scatter(x=df_macd.index, y=df_macd['MACD_LINE'], name='MACD_LINE', line=dict(color='green')))

    macd_fig.add_trace(go.Scatter(x=df_macd.index, y=df_macd['SIGNAL_LINE'], name='SIGNAL_LINE', line=dict(color='firebrick')))

    macd_fig.add_trace(go.Bar(x=df_macd.index, y=df_macd['HISTOGRAM'], name='HISTOGRAM', marker_color='White'))

    macd_fig.update_layout(
        title='MACD Chart',
        template='plotly_dark',
        yaxis2=dict(
        title='Closing Price', # Title for the secondary y-axis
        titlefont=dict(color='red'),
        tickfont=dict(color='red'),
        overlaying='y', # IMPORTANT: Overlay yaxis2 on yaxis
        side='right',
         # IMPORTANT: Place yaxis2 on the right side
    ))    

    
    # Create the trend line figure
    stock_trend_line = go.Figure()
    stock_trend_line.add_trace(go.Scatter(x=df.index, y=df['Close'], name=f"{ticker} Closing Price"))
    stock_trend_line.add_trace(go.Scatter(x=df.index, y=df['50_DAY_MA'], name=f'{ticker} 50-Day MA'))
    stock_trend_line.add_trace(go.Scatter(x=df.index, y=df['200_DAY_MA'], name=f'{ticker} 200-Day MA'))

    stock_trend_line.update_layout(
        title='Daily Closing Price Trend Chart Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )

    # Create earnings forecast table
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

    return stock_trend_line, rsi_fig, forecast_table, eps_fig, macd_fig  



##Creating the Distribution Table
##Creating the trend line graph call back
@callback(
    Output('price_dis_table', 'children'),
    Input('stock_symbols','value'),
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
    
        

if __name__ == '__main__':
    app.run(jupyter_mode='external',debug=True)

# %%
import plotly.graph_objects as go

df = yf.download(tickers='AAPL', period='5y')

df.columns = [col[0] for col in df.columns]

df_macd = df[['Close']]

df_macd['EMA_12'] = df_macd['Close'].ewm(span=12, adjust=False).mean()

df_macd['EMA_26'] = df_macd['Close'].ewm(span=26, adjust=False).mean()

df_macd['MACD_LINE'] = df_macd['EMA_12'] - df_macd['EMA_26']

df_macd['SIGNAL_LINE'] = df_macd['MACD_LINE'].ewm(span=9, adjust=False).mean()

df_macd['HISTOGRAM'] = df_macd['MACD_LINE'] - df_macd['SIGNAL_LINE']

macd_fig = go.Figure()

macd_fig.add_trace(go.Scatter(x=df_macd.index, y=df_macd['MACD_LINE'], name='MACD_LINE', line=dict(color='blue')))

macd_fig.add_trace(go.Scatter(x=df_macd.index, y=df_macd['SIGNAL_LINE'], name='SIGNAL_LINE', line=dict(color='red')))

macd_fig.add_trace(go.Bar(x=df_macd.index, y=df_macd['HISTOGRAM'], name='HISTOGRAM', marker_color='darkred'))

macd_fig.update_layout(
    title='MACD Chart',
    yaxis2=dict(
        title='Closing Price', # Title for the secondary y-axis
        titlefont=dict(color='red'),
        tickfont=dict(color='red'),
        overlaying='y', # IMPORTANT: Overlay yaxis2 on yaxis
        side='right'    # IMPORTANT: Place yaxis2 on the right side
    ))

macd_fig.show()

# %%



