import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def calculate_williams_r(df, period=14):
    if df.empty:
        return df

    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    williams_r = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
    df['Williams_%R'] = williams_r
    return df


def backtest_williams_r_strategy(df):
    if df.empty:
        return pd.DataFrame()

    trades = []
    position = None  # No position initially

    for i in range(1, len(df)):
        today = df.iloc[i]
        yesterday = df.iloc[i - 1]

        # Entry Conditions
        if position is None:
            # Long Entry: Williams %R crosses above -80 (from oversold to neutral/overbought)
            if yesterday['Williams_%R'] < -80 and today['Williams_%R'] >= -80:
                entry_price = today['Open']
                daily_range = today['High'] - today['Low']  # Simple volatility measure
                stop_loss = entry_price - daily_range  # Stop loss set at one day's range below entry
                take_profit = entry_price + 2 * daily_range  # Take profit at 2x the risk

                position = {
                    'Entry_Time': today['Date'],
                    'Entry_Price': entry_price,
                    'Stop_Loss': stop_loss,
                    'Take_Profit': take_profit,
                    'Exit_Time': None,
                    'Exit_Price': None,
                    'Result': None,
                    'Type': 'Long'
                }
                trades.append(position)

            # Short Entry: Williams %R crosses below -20 (from overbought to neutral/oversold)
            elif yesterday['Williams_%R'] > -20 and today['Williams_%R'] <= -20:
                entry_price = today['Open']
                daily_range = today['High'] - today['Low']  # Simple volatility measure
                stop_loss = entry_price + daily_range  # Stop loss set at one day's range above entry
                take_profit = entry_price - 2 * daily_range  # Take profit at 2x the risk

                position = {
                    'Entry_Time': today['Date'],
                    'Entry_Price': entry_price,
                    'Stop_Loss': stop_loss,
                    'Take_Profit': take_profit,
                    'Exit_Time': None,
                    'Exit_Price': None,
                    'Result': None,
                    'Type': 'Short'
                }
                trades.append(position)

        # Check for exit conditions if in a position
        elif position is not None:
            if position['Type'] == 'Long':
                # Exit Condition 1: Take Profit hit
                if today['High'] >= position['Take_Profit']:
                    position['Exit_Time'] = today['Date']
                    position['Exit_Price'] = position['Take_Profit']
                    position['Result'] = 'Win'
                    position = None
                # Exit Condition 2: Stop Loss hit
                elif today['Low'] <= position['Stop_Loss']:
                    position['Exit_Time'] = today['Date']
                    position['Exit_Price'] = position['Stop_Loss']
                    position['Result'] = 'Loss'
                    position = None
                # Exit Condition 3: Williams %R > -20 (Overbought)
                elif yesterday['Williams_%R'] <= -20 and today['Williams_%R'] > -20:
                    position['Exit_Time'] = today['Date']
                    position['Exit_Price'] = today['Open']
                    position['Result'] = 'Win' if today['Open'] > position['Entry_Price'] else 'Loss'
                    position = None

            elif position['Type'] == 'Short':
                # Exit Condition 1: Take Profit hit
                if today['Low'] <= position['Take_Profit']:
                    position['Exit_Time'] = today['Date']
                    position['Exit_Price'] = position['Take_Profit']
                    position['Result'] = 'Win'
                    position = None
                # Exit Condition 2: Stop Loss hit
                elif today['High'] >= position['Stop_Loss']:
                    position['Exit_Time'] = today['Date']
                    position['Exit_Price'] = position['Stop_Loss']
                    position['Result'] = 'Loss'
                    position = None
                # Exit Condition 3: Williams %R < -80 (Oversold)
                elif yesterday['Williams_%R'] >= -80 and today['Williams_%R'] < -80:
                    position['Exit_Time'] = today['Date']
                    position['Exit_Price'] = today['Open']
                    position['Result'] = 'Win' if today['Open'] < position['Entry_Price'] else 'Loss'
                    position = None

    # Close any open positions at the end
    if position is not None and position['Exit_Time'] is None:
        last_row = df.iloc[-1]
        position['Exit_Time'] = last_row['Date']
        position['Exit_Price'] = last_row['Close']
        if position['Type'] == 'Long':
            position['Result'] = 'Win' if position['Exit_Price'] > position['Entry_Price'] else 'Loss'
        else:  # Short
            position['Result'] = 'Win' if position['Exit_Price'] < position['Entry_Price'] else 'Loss'

    trades_df = pd.DataFrame(trades)
    return trades_df


def calculate_williams_r_metrics(df, trades_df, start_date, end_date):
    buy_and_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100

    strategy_returns = trades_df.apply(
        lambda x: (x['Exit_Price'] - x['Entry_Price']) / x['Entry_Price']
        if x['Type'] == 'Long'
        else (x['Entry_Price'] - x['Exit_Price']) / x['Entry_Price']
        if x['Type'] == 'Short'
        else 0, axis=1
    )
    strategy_return = strategy_returns.sum() * 100

    cumulative_strategy_returns = (1 + strategy_returns).cumprod()
    peak_strategy = cumulative_strategy_returns.expanding(min_periods=1).max()
    drawdown_strategy = (cumulative_strategy_returns / peak_strategy) - 1
    max_drawdown_strategy = drawdown_strategy.min() * 100

    cumulative_bnh_returns = (1 + df['Close'].pct_change().fillna(0)).cumprod()
    peak_bnh = cumulative_bnh_returns.expanding(min_periods=1).max()
    drawdown_bnh = (cumulative_bnh_returns / peak_bnh) - 1
    max_drawdown_bnh = drawdown_bnh.min() * 100

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
    
    trade_time = sum((pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() for start, end in merged_intervals)
    exposure_time = (trade_time / total_time) * 100 if total_time > 0 else 0

    num_trades = len(trades_df)

    win_rate = len(trades_df[trades_df['Result'] == 'Win']) / num_trades * 100 if num_trades > 0 else 0

    avg_win = strategy_returns[trades_df['Result'] == 'Win'].mean() * 100 if len(trades_df[trades_df['Result'] == 'Win']) > 0 else 0
    avg_loss = strategy_returns[trades_df['Result'] == 'Loss'].mean() * 100 if len(trades_df[trades_df['Result'] == 'Loss']) > 0 else 0

    total_profit = strategy_returns[trades_df['Result'] == 'Win'].sum()
    total_loss = abs(strategy_returns[trades_df['Result'] == 'Loss'].sum())
    profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')

    return {
        'Buy and Hold Return (%)': round(buy_and_hold_return, 2),
        'Max Drawdown Buy and Hold (%)': round(max_drawdown_bnh, 2),
        'Strategy Return (%)': round(strategy_return, 2),
        'Max Drawdown Strategy (%)': round(max_drawdown_strategy, 2),
        'Exposure Time (%)': round(exposure_time, 2),
        'Number of Trades': num_trades,
        'Win Rate (%)': round(win_rate, 2),
        'Average Win (%)': round(avg_win, 2),
        'Average Loss (%)': round(avg_loss, 2),
        'Profit Factor': round(profit_factor, 2)
    }


def plot_trades_williams_r(df, title):
    if df.empty:
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

    # Format x-axis to show date labels without gaps for weekends
    step = max(len(df) // 10, 1)
    date_ticks = df['Date'][::step]
    ax1.set_xticks(range(0, len(df), step))
    ax1.set_xticklabels([dt.strftime('%Y-%m-%d') for dt in date_ticks], rotation=45)

    # Combine legends from both axes
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
