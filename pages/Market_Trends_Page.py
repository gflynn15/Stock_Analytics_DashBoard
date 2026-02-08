@cache.memoize(timeout=600) # 10 Minutes
def get_market_data(tickers, period, interval):
    # Ensure unique list
    tickers = list(set(tickers))
    
    # CHANGE: threads=False
    # This prevents the "2 of 3 completed" errors by downloading one by one.
    try:
        df = yf.download(
            tickers=tickers, 
            period=period, 
            interval=interval, 
            group_by='ticker', 
            threads=False,   # <--- THE FIX
            auto_adjust=True
        )
    except Exception:
        return pd.DataFrame().to_json()

    # Initialize result DataFrame with the same index (Dates)
    result_df = pd.DataFrame(index=df.index)
    
    for t in tickers:
        try:
            # CASE 1: MultiIndex DataFrame
            if isinstance(df.columns, pd.MultiIndex):
                if t in df.columns:
                    if 'Close' in df[t].columns:
                        result_df[t] = df[t]['Close']
                    elif 'Adj Close' in df[t].columns:
                        result_df[t] = df[t]['Adj Close']
            
            # CASE 2: Flat DataFrame (Single Ticker result)
            else:
                if 'Close' in df.columns:
                    result_df[t] = df['Close']
                elif 'Adj Close' in df.columns:
                    result_df[t] = df['Adj Close']
        except Exception:
            continue
            
    return result_df.to_json(date_format='iso')