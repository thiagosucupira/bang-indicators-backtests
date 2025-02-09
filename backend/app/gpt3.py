import time
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, time as dtime
import numpy as np

# -------------------
# 1) Configurable Parameters
# -------------------
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
DAYS_LOOKBACK = 120          # Look back 30 days to build pattern
PATTERN_HOUR = 8            # Hour of the day (server time) to exploit
PATTERN_MINUTE = 0          # Minute of the day (e.g. 08:00)
SIGNIFICANCE_STD = 0.5      # Mean return must exceed 0.5 std dev to be considered significant
MAX_DAILY_DD_PERCENT = 3.0
DEFAULT_LOT_SIZE = 1.0
MAX_TRADES_PER_DAY = 1

# Risk Management for the trade:
RISK_MULTIPLIER = 1.0       # SL distance factor, e.g., 1x average move
REWARD_MULTIPLIER = 2.0     # TP at twice the SL distance for a 1:2 RR

_daily_start_equity = None
_daily_drawdown_exceeded = False
_current_day = None
_trades_opened_today = 0

def initialize_mt5(login=None, password=None, server=None):
    if not mt5.initialize():
        print(f"initialize() failed. Error code: {mt5.last_error()}")
        quit()
    print("MetaTrader5 package initialized.")

    if login and password and server:
        authorized = mt5.login(login=login, password=password, server=server)
        if not authorized:
            print(f"MT5 login failed. Login={login}, Server={server}")
            mt5.shutdown()
            quit()
        else:
            print(f"Logged in to account #{login} on server {server}")

def shutdown_mt5():
    mt5.shutdown()
    print("MetaTrader5 connection shut down.")

def place_order(symbol, order_type, lot_size, stop_loss=None, take_profit=None, comment="TIME_PATTERN"):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[Error] Symbol {symbol} not found!")
        return None

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"[Error] Failed to select symbol {symbol}")
            return None

    price = symbol_info.ask if order_type == mt5.ORDER_TYPE_BUY else symbol_info.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "deviation": 10,
        "magic": 123456,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
    }

    if stop_loss is not None:
        request["sl"] = stop_loss
    if take_profit is not None:
        request["tp"] = take_profit

    result = mt5.order_send(request)
    if result is None:
        print(f"order_send() failed, error code: {mt5.last_error()}")
        return None
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"order_send() failed, retcode={result.retcode}")
        print(f"Request: {request}")
    else:
        print(f"Order placed! Ticket: {result.order}, Lots: {lot_size}, Type: {order_type}")
        print(f"Entry Price: {result.price}, SL: {stop_loss}, TP: {take_profit}")

    return result

def close_all_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    for position in positions:
        close_trade(symbol, position)

def close_trade(symbol, position, volume=None):
    if volume is None:
        volume = position.volume
    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_price = mt5.symbol_info_tick(symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask
    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "position": position.ticket,
        "price": close_price,
        "comment": "TIME_PATTERN_EXIT",
    }
    result = mt5.order_send(close_request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Closed {volume} lots of position {position.ticket} on {symbol}.")
    else:
        print(f"Failed to close position {position.ticket} on {symbol}. Retcode={result.retcode if result else 'N/A'}")

def has_daily_drawdown_exceeded():
    global _daily_drawdown_exceeded, _daily_start_equity
    if _daily_drawdown_exceeded or _daily_start_equity is None:
        return _daily_drawdown_exceeded

    account_info = mt5.account_info()
    if account_info is None:
        return False

    current_equity = account_info.equity
    threshold_equity = _daily_start_equity * (1 - MAX_DAILY_DD_PERCENT / 100)
    if current_equity < threshold_equity:
        print(f"[Drawdown Exceeded] Current equity {current_equity:.2f} < daily threshold {threshold_equity:.2f}")
        _daily_drawdown_exceeded = True
        return True
    return False

def reset_daily_equity_baseline():
    global _daily_start_equity, _daily_drawdown_exceeded, _trades_opened_today
    account_info = mt5.account_info()
    if account_info is not None:
        _daily_start_equity = account_info.equity
        _daily_drawdown_exceeded = False
        _trades_opened_today = 0
        print(f"[Daily Reset] Daily baseline equity set to {_daily_start_equity:.2f}")
    else:
        print("[Daily Reset] Could not fetch account info!")

def get_next_candle_time(timeframe):
    now = datetime.now()
    if timeframe == mt5.TIMEFRAME_M1:
        minutes = 1
    elif timeframe == mt5.TIMEFRAME_M5:
        minutes = 5
    elif timeframe == mt5.TIMEFRAME_M15:
        minutes = 15
    elif timeframe == mt5.TIMEFRAME_M30:
        minutes = 30
    elif timeframe == mt5.TIMEFRAME_H1:
        minutes = 60
    else:
        minutes = 15

    minutes_to_add = minutes - (now.minute % minutes)
    if minutes_to_add == minutes and now.second == 0:
        minutes_to_add = 0
    next_candle = now.replace(second=0, microsecond=0) + pd.Timedelta(minutes=minutes_to_add)
    return next_candle

def build_intraday_pattern(symbol, days=120, timeframe=mt5.TIMEFRAME_M15):
    """
    Build a DataFrame of average returns for each (hour, minute) slot of the day.
    We'll assume a 24-hour market and M15 bars => 96 bars per day.
    For each bar of the day, compute mean return and std of returns over the last N days.
    Return a DataFrame indexed by (hour, minute) with columns: mean_return, std_return
    """
    # Get historical data
    end = datetime.now()
    start = end - pd.Timedelta(days=days+1)  # buffer extra 1 day
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['Date'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('Date', inplace=True)

    # Compute returns: (close/close_prev - 1)
    df['ret'] = df['close'].pct_change()

    # Group by hour&minute to compute mean and std of returns
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    pattern = df.groupby(['hour','minute'])['ret'].agg(['mean','std','count'])
    return pattern

def check_pattern_significance(pattern, hour, minute, significance_std=0.5):
    """
    Check if the mean return at given hour & minute is significantly positive or negative.
    Significantly positive: mean > significance_std * std
    Significantly negative: mean < -significance_std * std
    Return "BUY" if positive, "SELL" if negative, else None.
    """
    if (hour, minute) not in pattern.index:
        return None
    mean_ret = pattern.loc[(hour, minute), 'mean']
    std_ret = pattern.loc[(hour, minute), 'std']
    if std_ret == 0 or np.isnan(std_ret):
        return None
    if mean_ret > significance_std * std_ret:
        return "BUY"
    elif mean_ret < -significance_std * std_ret:
        return "SELL"
    return None

def define_sl_tp(symbol, timeframe=mt5.TIMEFRAME_M15):
    """
    Define SL and TP based on historical intraday volatility for the last 20 bars:
    - SL = average absolute return * RISK_MULTIPLIER
    - TP = SL * REWARD_MULTIPLIER
    We'll convert returns to price pips to define actual SL/TP levels.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 20)
    if rates is None or len(rates)<2:
        return None, None
    df = pd.DataFrame(rates)
    df['ret'] = df['close'].pct_change()
    avg_abs_ret = df['ret'].abs().mean()
    # Convert this average return to pips:
    # If avg_abs_ret ~ 0.0001 => 1 pip for EURUSD
    # We'll just multiply current price by avg_abs_ret to get approx pip value
    current_price = mt5.symbol_info_tick(symbol).bid
    sl_distance = current_price * avg_abs_ret * RISK_MULTIPLIER
    tp_distance = sl_distance * REWARD_MULTIPLIER
    return sl_distance, tp_distance

def run_intraday_pattern_strategy():
    account_info = mt5.account_info()
    if account_info is None:
        print("[Error] Unable to fetch account info")
        return
    print(f"Account balance: {account_info.balance}, equity: {account_info.equity}")

    if has_daily_drawdown_exceeded():
        print("[Risk] Daily drawdown exceeded. No new trades.")
        return

    global _trades_opened_today

    # If we already traded today, skip
    if _trades_opened_today >= MAX_TRADES_PER_DAY:
        print("Max trades per day reached. No more entries today.")
        return

    # Build pattern
    pattern = build_intraday_pattern(SYMBOL, days=DAYS_LOOKBACK, timeframe=TIMEFRAME)
    if pattern is None:
        print("Could not build intraday pattern. No data.")
        return

    # Check current time
    now = datetime.now()
    # We trade EXACTLY at PATTERN_HOUR:PATTERN_MINUTE
    # If we haven't reached that time yet, do nothing.
    # If we are exactly at that time (or just a minute after), check pattern:
    target_time = now.replace(hour=PATTERN_HOUR, minute=PATTERN_MINUTE, second=0, microsecond=0)

    # Only attempt a trade if we are within a short window of the pattern time
    # say we check every M15 bar, so if now >= target_time and now < target_time+bar_interval
    # Actually, let's just check if now is within the same 15-min candle or just slightly after:
    if now < target_time or (now - target_time).total_seconds() > 60:
        # Not time to trade yet or we missed it by more than a minute.
        return

    direction = check_pattern_significance(pattern, PATTERN_HOUR, PATTERN_MINUTE, SIGNIFICANCE_STD)
    if direction is None:
        print("No significant pattern at this time. No trade.")
        return

    # direction = "BUY" or "SELL"
    sl_distance, tp_distance = define_sl_tp(SYMBOL, timeframe=TIMEFRAME)
    if sl_distance is None or tp_distance is None:
        print("Could not define SL/TP distances.")
        return

    # Get current price
    current_bid = mt5.symbol_info_tick(SYMBOL).bid
    current_ask = mt5.symbol_info_tick(SYMBOL).ask

    if direction == "BUY":
        entry_price = current_ask
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
        order_type = mt5.ORDER_TYPE_BUY
    else:
        entry_price = current_bid
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance
        order_type = mt5.ORDER_TYPE_SELL

    result = place_order(SYMBOL, order_type, DEFAULT_LOT_SIZE, stop_loss=sl, take_profit=tp)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        _trades_opened_today += 1

def main():
    # Update with your credentials
    MY_LOGIN = 5032225458
    MY_PASSWORD = "@6IzQoQz"
    MY_SERVER = "MetaQuotes-Demo"

    initialize_mt5(login=MY_LOGIN, password=MY_PASSWORD, server=MY_SERVER)
    global _current_day
    _current_day = datetime.now().day
    reset_daily_equity_baseline()

    print(f"Starting Intraday Seasonal Pattern Strategy on {SYMBOL}, timeframe={TIMEFRAME}...")

    try:
        while True:
            now = datetime.now()
            if now.day != _current_day:
                _current_day = now.day
                reset_daily_equity_baseline()

            run_intraday_pattern_strategy()

            next_candle = get_next_candle_time(TIMEFRAME)
            wait_seconds = (next_candle - datetime.now()).total_seconds()
            wait_seconds = max(1, wait_seconds+1)
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Cycle complete. Next candle at {next_candle.strftime('%H:%M:%S')}, sleeping {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)

    except KeyboardInterrupt:
        print("Manual shutdown signal received.")
    finally:
        shutdown_mt5()

if __name__ == "__main__":
    main()
