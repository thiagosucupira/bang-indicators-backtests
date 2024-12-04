import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import logging
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to calculate SMA and EMA
def calculate_sma_ema(df, sma_period=50, ema_period=20):
    if df.empty:
        return df
    df['SMA'] = df['Close'].rolling(window=sma_period).mean()
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    return df

# Function to implement the trading strategy
def backtest_sma_ema_strategy(df):
    if df.empty:
        return pd.DataFrame()
    
    # Calculate largest candle size for stop loss
    df['CandleSize'] = abs(df['High'] - df['Low'])
    stop_size = df['CandleSize'].max()
    
    trades = []
    position = None
    long_crossovers = 0
    short_crossovers = 0

    for i in range(1, len(df)):
        today = df.iloc[i]
        yesterday = df.iloc[i - 1]

        # Ensure SMA and EMA are not NaN
        if pd.isna(today['SMA']) or pd.isna(today['EMA']) or pd.isna(yesterday['SMA']) or pd.isna(yesterday['EMA']):
            continue

        # Check for stop loss or take profit if in a position
        if position is not None:
            exit_trade = False
            exit_reason = None

            if position['Type'] == 'Long':
                # Stop Loss for Long (1x candle size below entry)
                if today['Low'] <= (position['Entry_Price'] - stop_size):
                    exit_trade = True
                    exit_price = position['Entry_Price'] - stop_size  # Assuming we get filled at stop level
                    exit_reason = 'Stop Loss'
                # Take Profit for Long (2x candle size above entry)
                elif today['High'] >= (position['Entry_Price'] + (stop_size * 2)):
                    exit_trade = True
                    exit_price = position['Entry_Price'] + (stop_size * 2)  # Assuming we get filled at TP level
                    exit_reason = 'Take Profit'
                # Original crossover exit condition
                elif yesterday['EMA'] >= yesterday['SMA'] and today['EMA'] < today['SMA']:
                    exit_trade = True
                    exit_price = today['Open']
                    exit_reason = 'Crossover or Limit'
                    short_crossovers += 1

            else:  # Short position
                # Stop Loss for Short (1x candle size above entry)
                if today['High'] >= (position['Entry_Price'] + stop_size):
                    exit_trade = True
                    exit_price = position['Entry_Price'] + stop_size
                    exit_reason = 'Stop Loss'
                # Take Profit for Short (2x candle size below entry)
                elif today['Low'] <= (position['Entry_Price'] - (stop_size * 2)):
                    exit_trade = True
                    exit_price = position['Entry_Price'] - (stop_size * 2)
                    exit_reason = 'Take Profit'
                # Original crossover exit condition
                elif yesterday['EMA'] <= yesterday['SMA'] and today['EMA'] > today['SMA']:
                    exit_trade = True
                    exit_price = today['Open']
                    exit_reason = 'Crossover or Limit'
                    long_crossovers += 1

            if exit_trade:
                position['Exit_Time'] = today.name
                position['Exit_Price'] = exit_price
                position['Exit_Reason'] = exit_reason
                if position['Type'] == 'Long':
                    position['Result'] = 'Win' if exit_price > position['Entry_Price'] else 'Loss'
                else:  # Short
                    position['Result'] = 'Win' if exit_price < position['Entry_Price'] else 'Loss'
                trades.append(position)
                
                # Only enter new position if exit was due to crossover
                if exit_reason == 'Crossover':
                    entry_price = today['Open']
                    position = {
                        'Entry_Time': today.name,
                        'Entry_Price': entry_price,
                        'Exit_Time': None,
                        'Exit_Price': None,
                        'Exit_Reason': None,
                        'Result': None,
                        'Type': 'Short' if position['Type'] == 'Long' else 'Long'
                    }
                else:
                    position = None

        # Entry Conditions (only if not in a position)
        elif position is None:
            # Long Entry: EMA crosses above SMA
            if yesterday['EMA'] <= yesterday['SMA'] and today['EMA'] > today['SMA']:
                long_crossovers += 1
                entry_price = today['Open']
                position = {
                    'Entry_Time': today.name,
                    'Entry_Price': entry_price,
                    'Exit_Time': None,
                    'Exit_Price': None,
                    'Exit_Reason': None,
                    'Result': None,
                    'Type': 'Long'
                }

            # Short Entry: EMA crosses below SMA
            elif yesterday['EMA'] >= yesterday['SMA'] and today['EMA'] < today['SMA']:
                short_crossovers += 1
                entry_price = today['Open']
                position = {
                    'Entry_Time': today.name,
                    'Entry_Price': entry_price,
                    'Exit_Time': None,
                    'Exit_Price': None,
                    'Exit_Reason': None,
                    'Result': None,
                    'Type': 'Short'
                }

    # Close any open positions at the end
    if position is not None and position['Exit_Time'] is None:
        last_row = df.iloc[-1]
        position['Exit_Time'] = last_row.name  # Using the index as the timestamp
        position['Exit_Price'] = last_row['Close']
        if position['Type'] == 'Long':
            position['Result'] = 'Win' if position['Exit_Price'] > position['Entry_Price'] else 'Loss'
        else:  # Short
            position['Result'] = 'Win' if position['Exit_Price'] < position['Entry_Price'] else 'Loss'
        trades.append(position)

    trades_df = pd.DataFrame(trades)
    
    logger.info(f"Number of potential long entries: {long_crossovers}")
    logger.info(f"Number of potential short entries: {short_crossovers}")
    logger.info(f"Number of actual trades taken: {len(trades_df)}")
    
    return trades_df

# Function to calculate performance metrics
def calculate_sma_ema_metrics(df, trades_df):
    # Default metrics dictionary
    default_metrics = {
        'Buy and Hold Return (%)': 0,
        'Max Drawdown Buy and Hold (%)': 0,
        'Strategy Return (%)': 0,
        'Max Drawdown Strategy (%)': 0,
        'Number of Trades': 0,
        'Win Rate (%)': 0,
        'Average Win (%)': 0,
        'Average Loss (%)': 0,
        'Profit Factor': 0
    }

    # Return defaults if no data
    if df.empty or trades_df.empty:
        return default_metrics

    try:
        # Ensure 'Result' column exists
        if 'Result' not in trades_df.columns:
            print("Error: 'Result' column is missing from trades_df.")
            return default_metrics

        # Calculate Buy and Hold Return
        first_price = float(df['Close'].iloc[0])
        last_price = float(df['Close'].iloc[-1])
        buy_and_hold_return = ((last_price - first_price) / first_price) * 100
        buy_and_hold_return = 0 if pd.isna(buy_and_hold_return) else buy_and_hold_return

        # Calculate Buy and Hold Max Drawdown
        peak = df['Close'].expanding(min_periods=1).max()
        drawdown = ((df['Close'] - peak) / peak) * 100
        max_drawdown_bnh = float(drawdown.min()) if not pd.isna(drawdown.min()) else 0

        # Calculate Strategy Returns and Stats
        strategy_returns = trades_df.apply(
            lambda x: (x['Exit_Price'] - x['Entry_Price']) / x['Entry_Price']
            if x['Type'] == 'Long'
            else (x['Entry_Price'] - x['Exit_Price']) / x['Entry_Price']
            if x['Type'] == 'Short'
            else 0, axis=1
        ).fillna(0)
        
        strategy_return = strategy_returns.sum() * 100

        # Calculate drawdowns safely
        cumulative_strategy_returns = (1 + strategy_returns).cumprod()
        peak_strategy = cumulative_strategy_returns.expanding(min_periods=1).max()
        drawdown_strategy = ((cumulative_strategy_returns - peak_strategy) / peak_strategy).fillna(0)
        max_drawdown_strategy = float(drawdown_strategy.min()) * 100 if not pd.isna(drawdown_strategy.min()) else 0

        cumulative_bnh_returns = (1 + df['Close'].pct_change().fillna(0)).cumprod()
        peak_bnh = cumulative_bnh_returns.expanding(min_periods=1).max()
        drawdown_bnh = ((cumulative_bnh_returns - peak_bnh) / peak_bnh).fillna(0)
        max_drawdown_bnh = float(drawdown_bnh.min()) * 100 if not pd.isna(drawdown_bnh.min()) else 0

        # Calculate trade statistics
        num_trades = len(trades_df)
        winning_trades = trades_df[trades_df['Result'] == 'Win']
        win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0

        # Calculate average win/loss
        winning_returns = strategy_returns[trades_df['Result'] == 'Win']
        losing_returns = strategy_returns[trades_df['Result'] == 'Loss']
        
        avg_win = winning_returns.mean() * 100 if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() * 100 if len(losing_returns) > 0 else 0

        # Calculate profit factor
        gross_profits = winning_returns.sum()
        gross_losses = abs(losing_returns.sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')

        metrics = {
            'Buy and Hold Return (%)': round(float(buy_and_hold_return), 2),
            'Max Drawdown Buy and Hold (%)': round(float(max_drawdown_bnh), 2),
            'Strategy Return (%)': round(float(strategy_return), 2),
            'Max Drawdown Strategy (%)': round(float(max_drawdown_strategy), 2),
            'Number of Trades': int(num_trades),
            'Win Rate (%)': round(float(win_rate), 2),
            'Average Win (%)': round(float(avg_win), 2),
            'Average Loss (%)': round(float(avg_loss), 2),
            'Profit Factor': round(float(profit_factor), 2) if profit_factor != float('inf') else np.inf
        }

        # Replace any remaining NaN or infinite values with 0 (except Profit Factor)
        for key, value in metrics.items():
            if pd.isna(value) or (isinstance(value, float) and np.isinf(value)):
                if key == 'Profit Factor' and value == np.inf:
                    continue  # Keep as infinity
                metrics[key] = 0

        return metrics

    except Exception as e:
        print(f"Error in calculate_sma_ema_metrics: {str(e)}")
        return default_metrics

# Function to plot the trades
def plot_trades_sma_ema(df, trades_df, title):
    if df.empty:
        logger.error("DataFrame is empty. Cannot generate plot.")
        return ""
    
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Plot candlestick chart
    width = 0.6
    width2 = 0.05

    up = df[df.Close >= df.Open]
    down = df[df.Close < df.Open]

    # Plot Up Candles
    ax1.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='green', edgecolor='black', linewidth=0.5)
    ax1.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color='black', edgecolor='black', linewidth=0.5)
    ax1.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color='black', edgecolor='black', linewidth=0.5)

    # Plot Down Candles
    ax1.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color='red', edgecolor='black', linewidth=0.5)
    ax1.bar(down.index, down.High - down.Open, width2, bottom=down.Open, color='black', edgecolor='black', linewidth=0.5)
    ax1.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, color='black', edgecolor='black', linewidth=0.5)

    # Plot SMA and EMA
    ax1.plot(df.index, df['SMA'], label='SMA', color='blue', linestyle='--')
    ax1.plot(df.index, df['EMA'], label='EMA', color='orange', linestyle='--')

    # Plot Long and Short Entry/Exit Signals
    long_entry_plotted = False
    long_exit_plotted = False
    short_entry_plotted = False
    short_exit_plotted = False

    for _, trade in trades_df.iterrows():
        try:
            entry_time = trade['Entry_Time']
            exit_time = trade['Exit_Time']
            entry_price = trade['Entry_Price']
            exit_price = trade['Exit_Price']
            trade_type = trade['Type']
            trade_result = trade['Result']

            # Check if entry and exit times exist in the DataFrame index
            if entry_time not in df.index:
                logger.warning(f"Entry time {entry_time} not found in DataFrame index.")
                continue
            if exit_time not in df.index:
                logger.warning(f"Exit time {exit_time} not found in DataFrame index.")
                continue

            if trade_type == 'Long':
                # Long Entry: Upward-pointing triangle
                ax1.scatter(entry_time, entry_price, marker='^', color='green', s=100, label='Long Entry' if not long_entry_plotted else "")
                long_entry_plotted = True

                # Exit plotting based on reason
                if trade.get('Exit_Reason') == 'Stop Loss':
                    exit_color = 'red'
                elif trade.get('Exit_Reason') == 'Take Profit':
                    exit_color = 'green'
                else:  # Crossover
                    exit_color = 'blue' if trade_result == 'Win' else 'magenta'
                ax1.scatter(exit_time, exit_price, marker='v', color=exit_color, s=100, 
                          label=f'Long Exit ({trade.get("Exit_Reason", "Unknown")})' if not long_exit_plotted else "")
                long_exit_plotted = True

            elif trade_type == 'Short':
                # Short Entry: Square
                ax1.scatter(entry_time, entry_price, marker='s', color='red', s=100, label='Short Entry' if not short_entry_plotted else "")
                short_entry_plotted = True

                # Exit plotting based on reason
                if trade.get('Exit_Reason') == 'Stop Loss':
                    exit_color = 'red'
                elif trade.get('Exit_Reason') == 'Take Profit':
                    exit_color = 'green'
                else:  # Crossover
                    exit_color = 'cyan' if trade_result == 'Win' else 'purple'
                ax1.scatter(exit_time, exit_price, marker='s', color=exit_color, s=100,
                          label=f'Short Exit ({trade.get("Exit_Reason", "Unknown")})' if not short_exit_plotted else "")
                short_exit_plotted = True

        except Exception as e:
            logger.error(f"Error processing trade: {trade}, Error: {e}")
            continue

    # Handle Legend to Avoid Duplicates
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(0.1, 0.9))

    # Set titles and labels
    ax1.set_title(title)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')

    # Use Matplotlib's DateFormatter for x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Convert plot to Base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return image_base64
    