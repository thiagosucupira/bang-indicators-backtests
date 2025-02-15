import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from scipy.stats import zscore
import mplfinance as mpf
import io, base64

# -------------------------------
# Helper function
# -------------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

# -------------------------------
# DEBUG FLAG
# -------------------------------
DEBUG = True


def calculate_markov_features(df):
    """
    Calculate log returns and volatility, and normalize them using zscore.
    """
    # If 'Adj Close' is not present, try to use 'Close'
    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            raise ValueError("DataFrame must contain 'Adj Close' or 'Close' column.")
    
    # Sort the DataFrame by its index to ensure chronological order
    df = df.sort_index()

    # Calculate log returns and volatility then remove the missing values.
    df['log_returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['volatility'] = df['log_returns'].rolling(window=20).std()
    df = df.dropna().copy()

    # Normalize features using zscore
    features = ['log_returns', 'volatility']
    df[features] = df[features].apply(zscore)
    return df

def backtest_markov_strategy(df):
    """
    Apply Gaussian HMM on the provided features and generate trade signals
    based on state changes. Returns a DataFrame of trades.
    """
    features = ['log_returns', 'volatility']
    best_model = None
    best_score = float('-inf')
    
    # Fit HMM model
    for n_states in range(2, 4):
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42)
        model.fit(df[features])
        score = model.score(df[features])
        if score > best_score:
            best_score = score
            best_model = model

    # Generate state predictions and assign signals
    df['state'] = best_model.predict(df[features])
    state_means = {i: best_model.means_[i, 0] for i in range(best_model.n_components)}
    bullish_state = max(state_means, key=state_means.get)
    bearish_state = min(state_means, key=state_means.get)
    
    df['signal'] = 0
    df.loc[df['state'] == bullish_state, 'signal'] = 1
    df.loc[df['state'] == bearish_state, 'signal'] = -1

    trades = []
    position = None

    for i in range(1, len(df)):
        today = df.iloc[i]
        yesterday = df.iloc[i-1]

        # Entry conditions
        if position is None:
            if yesterday['signal'] != today['signal']:
                entry_price = today['Open'] if 'Open' in df.columns else today['Adj Close']
                daily_range = yesterday['High'] - yesterday['Low'] if 'High' in df.columns else 0
                
                if today['signal'] == 1:  # Long entry
                    stop_loss = entry_price - 1.5 * daily_range if daily_range > 0 else entry_price * 0.98
                    take_profit = entry_price + 3 * daily_range if daily_range > 0 else entry_price * 1.04
                    
                    position = {
                        'Entry_Time': today.name,
                        'Entry_Price': float(entry_price),
                        'Stop_Loss': float(stop_loss),
                        'Take_Profit': float(take_profit),
                        'Exit_Time': None,
                        'Exit_Price': None,
                        'Result': None,
                        'Type': 'Long'
                    }
                    trades.append(position)
                
                elif today['signal'] == -1:  # Short entry
                    stop_loss = entry_price + 1.5 * daily_range if daily_range > 0 else entry_price * 1.02
                    take_profit = entry_price - 3 * daily_range if daily_range > 0 else entry_price * 0.96
                    
                    position = {
                        'Entry_Time': today.name,
                        'Entry_Price': float(entry_price),
                        'Stop_Loss': float(stop_loss),
                        'Take_Profit': float(take_profit),
                        'Exit_Time': None,
                        'Exit_Price': None,
                        'Result': None,
                        'Type': 'Short'
                    }
                    trades.append(position)

        # Exit conditions
        elif position is not None:
            current_price = today['Close']
            
            if position['Type'] == 'Long':
                if current_price >= position['Take_Profit']:
                    position['Exit_Time'] = today.name
                    position['Exit_Price'] = float(position['Take_Profit'])
                    position['Result'] = 'Win'
                    position = None
                elif current_price <= position['Stop_Loss']:
                    position['Exit_Time'] = today.name
                    position['Exit_Price'] = float(position['Stop_Loss'])
                    position['Result'] = 'Loss'
                    position = None
                elif yesterday['signal'] == 1 and today['signal'] != 1:
                    position['Exit_Time'] = today.name
                    position['Exit_Price'] = float(today['Open'])
                    position['Result'] = 'Win' if today['Open'] > position['Entry_Price'] else 'Loss'
                    position = None
            
            elif position['Type'] == 'Short':
                if current_price <= position['Take_Profit']:
                    position['Exit_Time'] = today.name
                    position['Exit_Price'] = float(position['Take_Profit'])
                    position['Result'] = 'Win'
                    position = None
                elif current_price >= position['Stop_Loss']:
                    position['Exit_Time'] = today.name
                    position['Exit_Price'] = float(position['Stop_Loss'])
                    position['Result'] = 'Loss'
                    position = None
                elif yesterday['signal'] == -1 and today['signal'] != -1:
                    position['Exit_Time'] = today.name
                    position['Exit_Price'] = float(today['Open'])
                    position['Result'] = 'Win' if today['Open'] < position['Entry_Price'] else 'Loss'
                    position = None

    # Close any open positions at the end
    if position is not None:
        last_row = df.iloc[-1]
        position['Exit_Time'] = last_row.name
        position['Exit_Price'] = float(last_row['Close'])
        if position['Type'] == 'Long':
            position['Result'] = 'Win' if position['Exit_Price'] > position['Entry_Price'] else 'Loss'
        else:
            position['Result'] = 'Win' if position['Exit_Price'] < position['Entry_Price'] else 'Loss'

    trades_df = pd.DataFrame(trades)
    return trades_df

def calculate_markov_metrics(df, trades_df, start_date, end_date):
    """
    Calculate performance metrics for the Markov strategy.
    """
    # Initial buy and hold calculation
    buy_and_hold_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100) if len(df) > 0 else 0

    if trades_df.empty:
        return {
            'Buy and Hold Return (%)': round(buy_and_hold_return, 2),
            'Max Drawdown Buy and Hold (%)': 0,
            'Strategy Return (%)': 0,
            'Max Drawdown Strategy (%)': 0,
            'Exposure Time (%)': 0,
            'Number of Trades': 0,
            'Win Rate (%)': 0,
            'Average Win (%)': 0,
            'Average Loss (%)': 0,
            'Profit Factor': 0
        }

    # Calculate strategy returns
    strategy_returns = trades_df.apply(
        lambda x: (x['Exit_Price'] - x['Entry_Price']) / x['Entry_Price']
        if x['Type'] == 'Long'
        else (x['Entry_Price'] - x['Exit_Price']) / x['Entry_Price']
        if x['Type'] == 'Short'
        else 0, axis=1
    ).fillna(0)
    
    strategy_return = strategy_returns.sum() * 100

    # Calculate drawdowns
    cumulative_strategy_returns = (1 + strategy_returns).cumprod()
    peak_strategy = cumulative_strategy_returns.expanding(min_periods=1).max()
    drawdown_strategy = ((cumulative_strategy_returns / peak_strategy) - 1)
    max_drawdown_strategy = min(drawdown_strategy.min() * 100, 0)

    # Calculate buy and hold drawdown
    returns = df['Close'].pct_change()
    cumulative_bnh_returns = (1 + returns).cumprod()
    peak_bnh = cumulative_bnh_returns.expanding(min_periods=1).max()
    drawdown_bnh = ((cumulative_bnh_returns / peak_bnh) - 1)
    max_drawdown_bnh = min(drawdown_bnh.min() * 100, 0)

    # Calculate exposure time
    total_time = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds()
    trade_intervals = trades_df[['Entry_Time', 'Exit_Time']].dropna().sort_values(by='Entry_Time').values.tolist()
    
    merged_intervals = []
    for interval in trade_intervals:
        if not merged_intervals:
            merged_intervals.append(interval)
        else:
            last = merged_intervals[-1]
            if interval[0] <= last[1]:
                merged_intervals[-1][1] = max(last[1], interval[1])
            else:
                merged_intervals.append(interval)
    
    trade_time = sum((pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() 
                     for start, end in merged_intervals)
    exposure_time = (trade_time / total_time) * 100 if total_time > 0 else 0

    # Calculate trade statistics
    num_trades = len(trades_df)
    winning_trades = trades_df[trades_df['Result'] == 'Win']
    losing_trades = trades_df[trades_df['Result'] == 'Loss']
    
    win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0
    
    avg_win = strategy_returns[trades_df['Result'] == 'Win'].mean() * 100
    avg_win = 0 if pd.isna(avg_win) else avg_win
    
    avg_loss = strategy_returns[trades_df['Result'] == 'Loss'].mean() * 100
    avg_loss = 0 if pd.isna(avg_loss) else avg_loss

    total_profit = strategy_returns[trades_df['Result'] == 'Win'].sum()
    total_loss = abs(strategy_returns[trades_df['Result'] == 'Loss'].sum())
    profit_factor = (total_profit / total_loss) if total_loss != 0 else 0

    metrics = {
        'Buy and Hold Return (%)': round(buy_and_hold_return, 2),
        'Max Drawdown Buy and Hold (%)': round(max_drawdown_bnh, 2),
        'Strategy Return (%)': round(strategy_return, 2),
        'Max Drawdown Strategy (%)': round(max_drawdown_strategy, 2),
        'Exposure Time (%)': round(exposure_time, 2),
        'Number of Trades': int(num_trades),
        'Win Rate (%)': round(win_rate, 2),
        'Average Win (%)': round(avg_win, 2),
        'Average Loss (%)': round(avg_loss, 2),
        'Profit Factor': round(profit_factor, 2)
    }
    
    return metrics

def plot_trades_markov(df, trades, title="Markov Strategy"):
    """
    Plot a candlestick chart with trade markers overlaid.
    Returns a base64 encoded PNG image string.
    """
    # Prepare additional plots for signal markers if available
    ap_long = None
    ap_short = None
    if 'Open' in df.columns:
        # Create a Series with the same index as df for proper alignment
        long_signals = pd.Series(np.nan, index=df.index, dtype=float)
        short_signals = pd.Series(np.nan, index=df.index, dtype=float)
        
        # Only set values where signals exist
        long_signals.loc[df['signal'] == 1] = df.loc[df['signal'] == 1, 'Open'].astype(float)
        short_signals.loc[df['signal'] == -1] = df.loc[df['signal'] == -1, 'Open'].astype(float)
        
        # Create addplots with non-NaN values only
        ap_long = mpf.make_addplot(long_signals, type='scatter',
                                 markersize=100, marker='^', color='green')
        ap_short = mpf.make_addplot(short_signals, type='scatter',
                                  markersize=100, marker='v', color='red')
    
    additional_plots = [ap_long, ap_short] if (ap_long is not None and ap_short is not None) else None
    
    fig, axlist = mpf.plot(df, type='candle', style='charles', title=title,
                          addplot=additional_plots,
                          returnfig=True)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img

def run_markov_strategy(df):
    """
    Wrapper function to run the complete Markov strategy.
    It prepares features, performs backtesting, calculates metrics, and generates a plot.
    Returns a dict with keys: 'plotImage', 'metrics', 'trades'.
    """
    # If a 'Date' column exists, convert and set it as the index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index("Date", inplace=True)
    
    # Get start and end dates for exposure time calculation
    start_date = df.index[0]
    end_date = df.index[-1]
    
    # Process data (calculate log returns, volatility, and normalize)
    df_features = calculate_markov_features(df)
    
    # Generate trade signals via Gaussian HMM
    trades_df = backtest_markov_strategy(df_features)
    
    # Calculate performance metrics
    metrics = calculate_markov_metrics(df_features, trades_df, start_date, end_date)
    
    # Generate candlestick plot with trade markers if OHLC data is available
    plot_image = None
    if all(col in df_features.columns for col in ['Open', 'High', 'Low', 'Close']):
        plot_image = plot_trades_markov(df_features, trades_df)
    
    # Convert trades DataFrame to a format suitable for JSON serialization
    trades_dict = None
    if not trades_df.empty:
        trades_df['Entry_Time'] = trades_df['Entry_Time'].astype(str)
        trades_df['Exit_Time'] = trades_df['Exit_Time'].astype(str)
        trades_dict = trades_df.to_dict('records')
    
    return {
        'plotImage': plot_image,
        'metrics': metrics,
        'trades': trades_dict if trades_dict is not None else []
    }

# NEW FUNCTION: Detailed Markov Strategy with SL/TP and risk management

def run_markov_strategy_detailed(df):
    try:
        # Prepare the data
        df = df.copy()
        df['Returns'] = df['Close'].pct_change()
        df = df.dropna()

        # Fit HMM model
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
        observations = np.column_stack([df['Returns']])
        model.fit(observations)

        # Get hidden states
        hidden_states = model.predict(observations)
        
        # Calculate state means to identify bullish/bearish states
        state_means = []
        for state in range(2):
            mask = (hidden_states == state)
            state_means.append(df['Returns'][mask].mean())
        
        bullish_state = np.argmax(state_means)
        bearish_state = np.argmin(state_means)

        # Generate trading signals
        df['State'] = hidden_states
        df['Signal'] = 0
        df.loc[df['State'] == bullish_state, 'Signal'] = 1
        df.loc[df['State'] == bearish_state, 'Signal'] = -1

        # Backtest the strategy
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()

        # Calculate metrics
        total_return = df['Cumulative_Returns'].iloc[-1] - 1
        sharpe_ratio = np.sqrt(252) * df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()
        max_drawdown = (df['Cumulative_Returns'] / df['Cumulative_Returns'].cummax() - 1).min()

        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'signals': df['Signal'].tolist(),
            'cumulative_returns': df['Cumulative_Returns'].tolist()
        }

    except Exception as e:
        print(f"Error in Markov strategy: {str(e)}")
        return {
            'error': str(e)
        }
