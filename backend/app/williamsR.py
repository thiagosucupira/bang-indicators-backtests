import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def calculate_williams_r(df, period=14):
    if df.empty:
        return df

    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    williams_r = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
    df['Williams_%R'] = williams_r.fillna(0)  # Handle NaN values
    return df


def backtest_williams_r_strategy(df):
    if df.empty:
        return pd.DataFrame()

    trades = []
    position = None
    partial_exit = False

    for i in range(1, len(df)):
        today = df.iloc[i]
        yesterday = df.iloc[i - 1]

        # Entry Conditions
        if position is None:
            # Long Entry
            if yesterday['Williams_%R'] < -80 and today['Williams_%R'] >= -80:
                entry_price = today['Open']
                daily_range = yesterday['High'] - yesterday['Low']
                stop_loss = entry_price - 1.5 * daily_range
                take_profit = entry_price + 3 * daily_range

                position = {
                    'Entry_Time': today['Date'],
                    'Entry_Price': float(entry_price),
                    'Stop_Loss': float(stop_loss),
                    'Take_Profit': float(take_profit),
                    'Exit_Time': None,
                    'Exit_Price': None,
                    'Result': None,
                    'Type': 'Long',
                    'Partial': False
                }
                trades.append(position)

            # Short Entry
            elif yesterday['Williams_%R'] > -20 and today['Williams_%R'] <= -20:
                entry_price = today['Open']
                daily_range = yesterday['High'] - yesterday['Low']
                stop_loss = entry_price + 1.5 * daily_range
                take_profit = entry_price - 3 * daily_range

                position = {
                    'Entry_Time': today['Date'],
                    'Entry_Price': float(entry_price),
                    'Stop_Loss': float(stop_loss),
                    'Take_Profit': float(take_profit),
                    'Exit_Time': None,
                    'Exit_Price': None,
                    'Result': None,
                    'Type': 'Short',
                    'Partial': False
                }
                trades.append(position)

        # Exit Conditions
        elif position is not None:
            if position['Type'] == 'Long':
                # Partial profit taking
                half_profit_target = position['Entry_Price'] + (position['Take_Profit'] - position['Entry_Price']) / 2
                if not partial_exit and today['High'] >= half_profit_target:
                    partial_trade = position.copy()
                    partial_trade['Exit_Time'] = today['Date']
                    partial_trade['Exit_Price'] = float(half_profit_target)
                    partial_trade['Result'] = 'Win'
                    partial_trade['Partial'] = True
                    trades.append(partial_trade)
                    
                    position['Stop_Loss'] = position['Entry_Price']
                    partial_exit = True
                    continue

                # Regular exits
                if today['High'] >= position['Take_Profit']:
                    position['Exit_Time'] = today['Date']
                    position['Exit_Price'] = float(position['Take_Profit'])
                    position['Result'] = 'Win'
                    position = None
                    partial_exit = False
                elif today['Low'] <= position['Stop_Loss']:
                    position['Exit_Time'] = today['Date']
                    position['Exit_Price'] = float(position['Stop_Loss'])
                    position['Result'] = 'Loss'
                    position = None
                    partial_exit = False
                elif yesterday['Williams_%R'] <= -20 and today['Williams_%R'] > -20:
                    position['Exit_Time'] = today['Date']
                    position['Exit_Price'] = float(today['Open'])
                    position['Result'] = 'Win' if today['Open'] > position['Entry_Price'] else 'Loss'
                    position = None

            elif position['Type'] == 'Short':
                # Partial profit taking
                half_profit_target = position['Entry_Price'] - (position['Entry_Price'] - position['Take_Profit']) / 2
                if not partial_exit and today['Low'] <= half_profit_target:
                    partial_trade = position.copy()
                    partial_trade['Exit_Time'] = today['Date']
                    partial_trade['Exit_Price'] = float(half_profit_target)
                    partial_trade['Result'] = 'Win'
                    partial_trade['Partial'] = True
                    trades.append(partial_trade)
                    
                    position['Stop_Loss'] = position['Entry_Price']
                    partial_exit = True
                    continue

                # Regular exits
                if today['Low'] <= position['Take_Profit']:
                    position['Exit_Time'] = today['Date']
                    position['Exit_Price'] = float(position['Take_Profit'])
                    position['Result'] = 'Win'
                    position = None
                    partial_exit = False
                elif today['High'] >= position['Stop_Loss']:
                    position['Exit_Time'] = today['Date']
                    position['Exit_Price'] = float(position['Stop_Loss'])
                    position['Result'] = 'Loss'
                    position = None
                    partial_exit = False
                elif yesterday['Williams_%R'] >= -80 and today['Williams_%R'] < -80:
                    position['Exit_Time'] = today['Date']
                    position['Exit_Price'] = float(today['Open'])
                    position['Result'] = 'Win' if today['Open'] < position['Entry_Price'] else 'Loss'
                    position = None

    # Close any open positions at the end
    if position is not None and position['Exit_Time'] is None:
        last_row = df.iloc[-1]
        position['Exit_Time'] = last_row['Date']
        position['Exit_Price'] = float(last_row['Close'])
        if position['Type'] == 'Long':
            position['Result'] = 'Win' if position['Exit_Price'] > position['Entry_Price'] else 'Loss'
        else:
            position['Result'] = 'Win' if position['Exit_Price'] < position['Entry_Price'] else 'Loss'

    trades_df = pd.DataFrame(trades)
    
    # Clean NaN values from trades DataFrame
    if not trades_df.empty:
        trades_df = trades_df.fillna(0)
        numeric_columns = trades_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            trades_df[col] = trades_df[col].apply(lambda x: float(0) if pd.isna(x) or np.isinf(x) else float(x))

    return trades_df


def calculate_williams_r_metrics(df, trades_df, start_date, end_date):
    print("Starting metrics calculation...")
    print(f"Number of trades: {len(trades_df)}")
    
    # Initial buy and hold calculation
    buy_and_hold_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100) if len(df) > 0 else 0
    print(f"Buy and hold return: {buy_and_hold_return}")

    if trades_df.empty:
        print("No trades found, returning default values")
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

    # Calculate strategy returns with NaN handling
    strategy_returns = trades_df.apply(
        lambda x: (x['Exit_Price'] - x['Entry_Price']) / x['Entry_Price']
        if x['Type'] == 'Long'
        else (x['Entry_Price'] - x['Exit_Price']) / x['Entry_Price']
        if x['Type'] == 'Short'
        else 0, axis=1
    ).fillna(0)
    
    strategy_return = strategy_returns.sum() * 100

    # Calculate drawdowns with NaN handling
    cumulative_strategy_returns = (1 + strategy_returns).cumprod().fillna(1)
    peak_strategy = cumulative_strategy_returns.expanding(min_periods=1).max()
    drawdown_strategy = ((cumulative_strategy_returns / peak_strategy) - 1).fillna(0)
    max_drawdown_strategy = min(drawdown_strategy.min() * 100, 0)

    # Calculate buy and hold drawdown with NaN handling
    returns = df['Close'].pct_change().fillna(0)
    cumulative_bnh_returns = (1 + returns).cumprod()
    peak_bnh = cumulative_bnh_returns.expanding(min_periods=1).max()
    drawdown_bnh = ((cumulative_bnh_returns / peak_bnh) - 1).fillna(0)
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

    # Calculate trade statistics with NaN handling
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
    profit_factor = 0 if pd.isna(profit_factor) else profit_factor

    def safe_float(value):
        if pd.isna(value) or value is None or isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return 0.0
        return float(value)

    metrics = {
        'Buy and Hold Return (%)': safe_float(buy_and_hold_return),
        'Max Drawdown Buy and Hold (%)': safe_float(max_drawdown_bnh),
        'Strategy Return (%)': safe_float(strategy_return),
        'Max Drawdown Strategy (%)': safe_float(max_drawdown_strategy),
        'Exposure Time (%)': safe_float(exposure_time),
        'Number of Trades': int(num_trades),
        'Win Rate (%)': safe_float(win_rate),
        'Average Win (%)': safe_float(avg_win),
        'Average Loss (%)': safe_float(avg_loss),
        'Profit Factor': safe_float(profit_factor)
    }

    metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items()}
    
    return metrics


def plot_trades_williams_r(df, title):
    if df.empty:
        return ""

    # Clean NaN values from the dataframe
    df = df.fillna(method='ffill').fillna(0)
    
    # Ensure all numeric columns are clean
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        df[col] = df[col].apply(lambda x: float(0) if pd.isna(x) or np.isinf(x) else float(x))

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

    # Plot Buy and Sell Signals
    buys = df[df['Williams_%R'] < -80]
    sells = df[df['Williams_%R'] > -20]
    ax1.scatter(buys.index, buys['Close'], marker='^', color='green', label='Buy Signal', zorder=5)
    ax1.scatter(sells.index, sells['Close'], marker='v', color='red', label='Sell Signal', zorder=5)

    # Plot Williams %R on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['Williams_%R'], label='Williams %R', color='orange')
    ax2.axhline(-80, color='red', linestyle='--', linewidth=1)
    ax2.axhline(-20, color='green', linestyle='--', linewidth=1)

    # Set titles and labels
    ax1.set_title(title)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Williams %R')

    # Format x-axis
    step = max(len(df) // 10, 1)
    date_ticks = df['Date'][::step]
    ax1.set_xticks(range(0, len(df), step))
    ax1.set_xticklabels([dt.strftime('%Y-%m-%d') for dt in date_ticks], rotation=45)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", bbox_to_anchor=(0.1, 0.9))

    plt.tight_layout()

    # Convert plot to Base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return image_base64
