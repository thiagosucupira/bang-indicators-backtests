import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from scipy.stats import zscore
import mplfinance as mpf

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

# -------------------------------
# 1. Download Data
# -------------------------------
symbol = "EURUSD=X"
data = yf.download(symbol, interval="1h", start="2024-10-01", end="2025-01-01")
if data.empty:
    raise ValueError("Failed to download EUR/USD data. Try adjusting the timeframe.")

# -------------------------------
# 2. Compute Log Returns & Volatility
# -------------------------------
data['log_returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
data['volatility'] = data['log_returns'].rolling(window=20).std()
data = data.dropna().copy()

# -------------------------------
# 3. HMM Analysis (using only HMM features)
# -------------------------------
features = ['log_returns', 'volatility']
data.loc[:, features] = data.loc[:, features].apply(zscore)

best_model = None
best_score = float('-inf')
for n_states in range(2, 4):
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(data[features])
    score = model.score(data[features])
    if score > best_score:
        best_score = score
        best_model = model

data.loc[:, 'state'] = best_model.predict(data[features])
state_means = {i: best_model.means_[i, 0] for i in range(best_model.n_components)}
bullish_state = max(state_means, key=state_means.get)
bearish_state = min(state_means, key=state_means.get)
data.loc[:, 'signal'] = 0
data.loc[data['state'] == bullish_state, 'signal'] = 1
data.loc[data['state'] == bearish_state, 'signal'] = -1

# -------------------------------
# 4. Define SL/TP Levels (tight thresholds)
# -------------------------------
SL_pct_long = 0.001   # 0.1% stop-loss for long trades
TP_pct_long = 0.002   # 0.2% take-profit for long trades
SL_pct_short = 0.001  # 0.1% stop-loss for short trades
TP_pct_short = 0.002  # 0.2% take-profit for short trades

# -------------------------------
# 5. Simulate Strategy (Regime-Change-Only Entries)
# -------------------------------
# We shift the signal so that we enter at the open of the candle following a signal.
data['entry_signal'] = data['signal'].shift(1)

initial_capital = 10000
capital = initial_capital
equity_curve = [capital]
trades = []      # List to store trade details

# We'll mark the trade entry in the DataFrame.
data['trade_entry'] = np.nan

# To reduce overtrading, only trade on regime changes.
last_trade_signal = 0  
risk_per_trade = 0.005  # risk 0.5% of capital per trade

for i in range(1, len(data)):
    trade_executed = False
    entry_signal = data['entry_signal'].iloc[i]
    
    if (not np.isnan(entry_signal)) and (entry_signal != 0) and (entry_signal != last_trade_signal):
        trade_signal = int(entry_signal)
        entry_price = data['Open'].iloc[i]  # Enter at current candle's open.
        
        if trade_signal == 1:  # Long trade
            sl = entry_price * (1 - SL_pct_long)
            tp = entry_price * (1 + TP_pct_long)
        else:  # Short trade
            sl = entry_price * (1 + SL_pct_short)
            tp = entry_price * (1 - TP_pct_short)
        
        if DEBUG:
            print(f"DEBUG: Trade signal from {data.index[i-1]}: {trade_signal}, "
                  f"enter at {data.index[i]} open {entry_price:.5f}, High: {data['High'].iloc[i]:.5f}, "
                  f"Low: {data['Low'].iloc[i]:.5f}, SL: {sl:.5f}, TP: {tp:.5f}")
        
        if trade_signal == 1:  # Long
            if data['Low'].iloc[i] <= sl:
                exit_price = sl
                exit_type = 'SL'
                trade_executed = True
            elif data['High'].iloc[i] >= tp:
                exit_price = tp
                exit_type = 'TP'
                trade_executed = True
        elif trade_signal == -1:  # Short
            if data['High'].iloc[i] >= sl:
                exit_price = sl
                exit_type = 'SL'
                trade_executed = True
            elif data['Low'].iloc[i] <= tp:
                exit_price = tp
                exit_type = 'TP'
                trade_executed = True
        
        if trade_executed:
            if trade_signal == 1:
                trade_return = ((exit_price - entry_price) / (entry_price * SL_pct_long)) * 100
            else:
                trade_return = ((entry_price - exit_price) / (entry_price * SL_pct_short)) * 100
            
            risk = abs(entry_price - sl)
            position_size = capital * risk_per_trade / risk
            if trade_signal == 1:
                profit = position_size * (exit_price - entry_price)
            else:
                profit = position_size * (entry_price - exit_price)
            capital += profit

            trade = {
                'date': data.index[i],
                'signal': trade_signal,
                'entry': entry_price,
                'exit': exit_price,
                'exit_type': exit_type,
                'profit': profit,
                'trade_return_pct': trade_return
            }
            trades.append(trade)
            data.at[data.index[i], 'trade_entry'] = entry_price
            
            last_trade_signal = trade_signal
            
            if DEBUG:
                print(f"DEBUG: Trade executed on {data.index[i]} | Exit: {exit_price:.5f} via {exit_type} | Profit: {profit:.2f} | Trade Return: {trade_return:.2f}%")
        else:
            if DEBUG:
                print(f"DEBUG: No trade executed on {data.index[i]} despite regime change signal {trade_signal}.")
    equity_curve.append(capital)

data.loc[:, 'equity_curve'] = equity_curve

# -------------------------------
# 6. Compute Backtest Metrics
# -------------------------------
initial_price = data['Adj Close'].iloc[0]
final_price = data['Adj Close'].iloc[-1]
buy_and_hold_return = ((final_price / initial_price) - 1) * 100

running_max_price = data['Adj Close'].cummax()
drawdown_bnh = (data['Adj Close'] - running_max_price) / running_max_price
max_drawdown_bnh = drawdown_bnh.min() * 100

final_capital = equity_curve[-1]
strategy_return = ((final_capital / initial_capital) - 1) * 100
equity_array = np.array(equity_curve)
running_max_equity = np.maximum.accumulate(equity_array)
drawdowns_strategy = (equity_array - running_max_equity) / running_max_equity
max_drawdown_strategy = drawdowns_strategy.min() * 100

num_candles = len(data)
num_trades = len(trades)
exposure_time = (num_trades / num_candles) * 100

wins = [t for t in trades if t['profit'] > 0]
losses = [t for t in trades if t['profit'] < 0]
win_rate = (len(wins) / num_trades) * 100 if num_trades > 0 else 0
avg_win = np.mean([t['trade_return_pct'] for t in wins]) if wins else 0
avg_loss = np.mean([t['trade_return_pct'] for t in losses]) if losses else 0
total_profit_wins = sum(t['profit'] for t in wins)
total_loss_abs = abs(sum(t['profit'] for t in losses))
profit_factor = (total_profit_wins / total_loss_abs) if total_loss_abs != 0 else np.nan

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

print("\nBacktest Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

# -------------------------------
# 7. Prepare Marker Columns for Plotting
# -------------------------------
# We'll create separate columns for long and short trade entries and exits.
data['long_entry_marker'] = np.nan
data['long_exit_marker'] = np.nan
data['short_entry_marker'] = np.nan
data['short_exit_marker'] = np.nan

for trade in trades:
    idx = trade['date']
    if trade['signal'] == 1:
        data.at[idx, 'long_entry_marker'] = trade['entry']
        data.at[idx, 'long_exit_marker'] = trade['exit']
    else:
        data.at[idx, 'short_entry_marker'] = trade['entry']
        data.at[idx, 'short_exit_marker'] = trade['exit']

# Create addplots for trade entry markers:
ap_long_entry = mpf.make_addplot(data['long_entry_marker'], type='scatter',
                                 markersize=150, marker='^', color='green')
ap_short_entry = mpf.make_addplot(data['short_entry_marker'], type='scatter',
                                  markersize=150, marker='v', color='red')

# -------------------------------
# 8. Plot Candlestick Chart with Trade Annotations
# -------------------------------
# Plot the candlestick chart and get the figure and axis.
volume_flag = True
if 'Volume' not in data or data['Volume'].isna().all():
    volume_flag = False

fig, axlist = mpf.plot(data, type='candle', style='charles',
                       title="EUR/USD 1H Candlestick with Trade Entries and SL/TP Labels",
                       addplot=[ap_long_entry, ap_short_entry],
                       volume=volume_flag,
                       ylabel="Price",
                       figsize=(14,8),
                       returnfig=True)
ax = axlist[0]  # Main axes

# Annotate each trade with text labels:
for trade in trades:
    # Find x coordinate (the index position) for the trade date.
    x = data.index.get_loc(trade['date'])
    if trade['signal'] == 1:
        color = 'green'
        # For long trades, place the "E" slightly below the entry and "X" slightly above the exit.
        ax.text(x, trade['entry'], 'E', color=color, fontsize=12, fontweight='bold',
                ha='center', va='bottom')
        ax.text(x, trade['exit'], 'X', color=color, fontsize=12, fontweight='bold',
                ha='center', va='top')
    else:
        color = 'red'
        # For short trades, place the "E" slightly above the entry and "X" slightly below the exit.
        ax.text(x, trade['entry'], 'E', color=color, fontsize=12, fontweight='bold',
                ha='center', va='top')
        ax.text(x, trade['exit'], 'X', color=color, fontsize=12, fontweight='bold',
                ha='center', va='bottom')

# -------------------------------
# 9. Plot the Equity Curve
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['equity_curve'], label='Equity Curve', color='blue')
plt.axhline(y=initial_capital, color='black', linestyle='--', label='Starting Capital')
plt.title("Strategy Equity Curve")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.legend()
plt.show()
