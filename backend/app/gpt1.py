import time
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import numpy as np

# -------------------
# 1) Configurable Parameters
# -------------------
MAX_DAILY_DD_PERCENT = 3.0
DEFAULT_LOT_SIZE = 2.8
SYMBOLS = ["EURUSD", "EURGBP", "EURJPY"]
TIMEFRAME = mt5.TIMEFRAME_M15

# Volatility Squeeze Settings
SQUEEZE_LOOKBACK = 20   # Number of bars to check the range
ATR_PERIOD = 100         # ATR calculation period
RANGE_ATR_THRESHOLD = 0.5  # If the range < 0.5*ATR(100), we consider it a squeeze

MAX_TRADES_PER_DAY_PER_SYMBOL = 5
_trades_opened_today = {symbol: 0 for symbol in SYMBOLS}

_daily_start_equity = None
_daily_drawdown_exceeded = False
_current_day = None

# Partial TP & Breakeven Settings
# After reaching half the TP distance, close half the position and move SL to breakeven
def calculate_atr(rates, period=14):
    df = pd.DataFrame(rates)
    df.columns = ["Time","Open","High","Low","Close","Tick_volume","Spread","Real_volume"]
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                     abs(df['Low'] - df['Close'].shift(1))))
    atr = df['TR'].rolling(period).mean()
    return atr.iloc[-1] if len(atr.dropna())>0 else None

def initialize_mt5(login=None, password=None, server=None):
    if not mt5.initialize():
        print(f"initialize() failed. Error code: {mt5.last_error()}")
        quit()
    print("MetaTrader5 package initialized.")

    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        print("Failed to get terminal info")
        quit()
    print(f"Trade allowed: {terminal_info.trade_allowed}")

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

def place_order(symbol, order_type, lot_size, stop_loss=None, take_profit=None, price=None, comment="SQUEEZE_TRADE"):
    """
    Place immediate or pending orders. If price is given and order_type is buy_stop or sell_stop type, 
    we place a pending order. If no price given, place a market order.
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[Error] Symbol {symbol} not found!")
        return None

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"[Error] Failed to select symbol {symbol}")
            return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL if price is None else mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "deviation": 10,
        "magic": 123456,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    if stop_loss is not None:
        request["sl"] = stop_loss
    if take_profit is not None:
        request["tp"] = take_profit
    if price is not None:
        request["price"] = price
        request["type_filling"] = mt5.ORDER_FILLING_FOK
    else:
        # Market order
        request["type_filling"] = mt5.ORDER_FILLING_FOK

    result = mt5.order_send(request)
    if result is None:
        print(f"order_send() failed, error code: {mt5.last_error()}")
        return None
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"order_send() failed, retcode={result.retcode}")
        print(f"Request: {request}")
    else:
        print(f"Order placed! Ticket: {result.order}, Lots: {lot_size}, Type: {order_type}")
        if price:
            print(f"Pending Order at Price: {price}, SL: {stop_loss}, TP: {take_profit}")
        else:
            print(f"Market Order, SL: {stop_loss}, TP: {take_profit}")

    return result

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
        "comment": "SQUEEZE_EXIT",
    }
    result = mt5.order_send(close_request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Closed {volume} lots of position {position.ticket} on {symbol}.")
    else:
        print(f"Failed to close position {position.ticket} on {symbol}. Retcode={result.retcode if result else 'N/A'}")

def move_sl_to_breakeven(symbol, position):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": position.ticket,
        "sl": position.price_open,
        "tp": position.tp,
        "magic": 123456
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"SL moved to breakeven at {position.price_open} for position {position.ticket}.")
    else:
        print(f"Failed to move SL to breakeven for position {position.ticket}. Retcode={result.retcode if result else 'N/A'}")

def manage_positions():
    """
    Manage open positions:
    - If position reached halfway to TP, close half and move SL to breakeven.
    """
    for symbol in SYMBOLS:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            continue

        for position in positions:
            if position.tp == 0:
                continue
            # Halfway point
            if position.type == mt5.POSITION_TYPE_BUY:
                half_way_point = position.price_open + (position.tp - position.price_open)/2
                current_price = mt5.symbol_info_tick(symbol).bid
                if current_price >= half_way_point and position.volume > 0.5:
                    half_volume = position.volume / 2
                    close_trade(symbol, position, volume=half_volume)
                    move_sl_to_breakeven(symbol, position)
            else:
                half_way_point = position.price_open - (abs(position.tp - position.price_open)/2)
                current_price = mt5.symbol_info_tick(symbol).ask
                if current_price <= half_way_point and position.volume > 0.5:
                    half_volume = position.volume / 2
                    close_trade(symbol, position, volume=half_volume)
                    move_sl_to_breakeven(symbol, position)

def check_volatility_squeeze_signal(symbol, timeframe=mt5.TIMEFRAME_M15):
    """
    Identify a volatility squeeze:
    1. Get the last N bars.
    2. Compute range = High - Low over N bars.
    3. Compute ATR(100).
    4. If range < RANGE_ATR_THRESHOLD * ATR(100), consider it a squeeze.
    If squeeze found, return the details needed to place pending orders.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 200)
    if rates is None or len(rates) < 100:
        return None

    # ATR(100)
    atr_value = calculate_atr(rates, ATR_PERIOD)
    if atr_value is None:
        return None

    df = pd.DataFrame(rates)
    df['Date'] = pd.to_datetime(df['time'], unit='s')
    # Last N bars
    recent = df.iloc[-SQUEEZE_LOOKBACK:]
    squeeze_high = recent['High'].max()
    squeeze_low = recent['Low'].min()
    squeeze_range = squeeze_high - squeeze_low

    print(f"{symbol} Squeeze Range: {squeeze_range:.5f}, ATR(100): {atr_value:.5f}")

    if squeeze_range < (RANGE_ATR_THRESHOLD * atr_value):
        # We have a squeeze
        # We'll set buy stop above squeeze_high and sell stop below squeeze_low
        return squeeze_low, squeeze_high
    return None

def place_squeeze_orders(symbol, squeeze_low, squeeze_high):
    """
    Place buy stop above squeeze_high and sell stop below squeeze_low.
    SL and TP based on squeeze range.
    """
    squeeze_range = squeeze_high - squeeze_low
    lot = DEFAULT_LOT_SIZE
    if lot <= 0:
        print(f"[Warning] Invalid lot size for {symbol}. Skip.")
        return

    # Let's say we set orders 2 pips above and below
    # Adjust for 5-digit brokers: 2 pips = 0.0002 for EURUSD
    point = mt5.symbol_info(symbol).point
    buffer = 20 * point  # 2 pip buffer if symbol is EURUSD-like

    buy_stop_price = squeeze_high + buffer
    sell_stop_price = squeeze_low - buffer

    # Stop Loss and Take Profit:
    # SL at opposite side of the range, TP double the range
    # For the buy scenario (once triggered): SL = squeeze_low - buffer, TP = buy_stop_price + 2*squeeze_range
    # For the sell scenario (once triggered): SL = squeeze_high + buffer, TP = sell_stop_price - 2*squeeze_range
    # We have to assign them now for pending orders.

    buy_sl = squeeze_low - buffer
    buy_tp = buy_stop_price + 2 * squeeze_range

    sell_sl = squeeze_high + buffer
    sell_tp = sell_stop_price - 2 * squeeze_range

    # Place Buy Stop
    place_order(symbol, mt5.ORDER_TYPE_BUY_STOP, lot, stop_loss=buy_sl, take_profit=buy_tp, price=buy_stop_price, comment="SQUEEZE_BUY_STOP")

    # Place Sell Stop
    place_order(symbol, mt5.ORDER_TYPE_SELL_STOP, lot, stop_loss=sell_sl, take_profit=sell_tp, price=sell_stop_price, comment="SQUEEZE_SELL_STOP")

def run_volatility_squeeze_strategy(timeframe=mt5.TIMEFRAME_M15):
    account_info = mt5.account_info()
    if account_info is None:
        print("[Error] Unable to fetch account info")
        return
    print(f"Account balance: {account_info.balance}, equity: {account_info.equity}")

    if has_daily_drawdown_exceeded():
        print("[Risk] Daily drawdown exceeded. No new trades.")
        return

    # Manage open positions (partial exit and breakeven)
    manage_positions()

    # Check for conditions to place pending orders if no trades active for that symbol
    for symbol in SYMBOLS:
        # Check if we still have daily trade allowance
        if _trades_opened_today[symbol] >= MAX_TRADES_PER_DAY_PER_SYMBOL:
            print(f"Max trades per day reached for {symbol}. No more entries today.")
            continue

        # Check if we already have pending orders or open positions related to this symbol
        # If we do, skip placing new orders to avoid stacking orders.
        orders = mt5.orders_get(symbol=symbol)
        positions = mt5.positions_get(symbol=symbol)
        if orders or positions:
            print(f"{symbol} already has positions or pending orders. Skipping new squeeze orders.")
            continue

        squeeze_result = check_volatility_squeeze_signal(symbol, timeframe)
        if squeeze_result is not None:
            squeeze_low, squeeze_high = squeeze_result
            print(f"{symbol} volatility squeeze detected. Placing breakout orders.")
            place_squeeze_orders(symbol, squeeze_low, squeeze_high)
            _trades_opened_today[symbol] += 1

def reset_daily_equity_baseline():
    global _daily_start_equity, _daily_drawdown_exceeded, _trades_opened_today
    account_info = mt5.account_info()
    if account_info is not None:
        _daily_start_equity = account_info.equity
        _daily_drawdown_exceeded = False
        for s in SYMBOLS:
            _trades_opened_today[s] = 0
        print(f"[Daily Reset] Daily baseline equity set to {_daily_start_equity:.2f}")
    else:
        print("[Daily Reset] Could not fetch account info!")

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

def main():
    MY_LOGIN = 5032225458
    MY_PASSWORD = "@6IzQoQz"
    MY_SERVER = "MetaQuotes-Demo"

    initialize_mt5(login=MY_LOGIN, password=MY_PASSWORD, server=MY_SERVER)

    global _current_day
    _current_day = datetime.now().day
    reset_daily_equity_baseline()

    print(f"Starting Volatility Squeeze strategy on {SYMBOLS}, timeframe={TIMEFRAME}...")

    try:
        while True:
            now = datetime.now()
            if now.day != _current_day:
                _current_day = now.day
                reset_daily_equity_baseline()

            run_volatility_squeeze_strategy(TIMEFRAME)

            next_candle = get_next_candle_time(TIMEFRAME)
            wait_seconds = (next_candle - datetime.now()).total_seconds()
            wait_seconds = max(1, wait_seconds + 1)
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Cycle complete. Next candle at {next_candle.strftime('%H:%M:%S')}, sleeping {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)

    except KeyboardInterrupt:
        print("Manual shutdown signal received.")
    finally:
        shutdown_mt5()

if __name__ == "__main__":
    main()
