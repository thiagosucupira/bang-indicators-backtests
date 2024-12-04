import yfinance as yf

def fetch_yfinance_data(symbol, interval, start_date, end_date):
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)
    df.reset_index(inplace=True)
    df.rename(columns={'Datetime': 'Date'}, inplace=True)
    return df