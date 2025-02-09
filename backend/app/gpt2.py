import time
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import numpy as np

# -------------------
# 1) Configurable Parameters
# -------------------
MAX_DAILY_DD_PERCENT = 3.0
DEFAULT_LOT_SIZE = 1.0
PAIR_A = "EURUSD"
PAIR_B = "GBPUSD"
SYMBOLS = [PAIR_A, PAIR_B]
TIMEFRAME = mt5.TIMEFRAME_M15

CORRELATION_PERIOD = 100        # Number of bars to compute correlation
CORRELATION_THRESHOLD = 0.9     # Minimum correlation to trust lead-lag
LOOKBACK_BREAKOUT = 20          # Breakout lookback
MAX_TRADES_PER_DAY_PER_SYMBOL = 3

# Partial TP & Breakeven Settings
_trades_opened_today = {PAIR_A: 0, PAIR_B: 0}
_daily_start_equity = None
_daily_drawdown_exceeded = False
_current_day = None

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

def place_order(symbol, order_type, lot_size, stop_loss=None, take_profit=None, comment="DYNAMIC_LEAD_LAG"):
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
        "comment": "DYNAMIC_LAG_EXIT",
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
    - Partial take profit at halfway to TP, move SL to breakeven.
    We apply this logic to both pairs, just in case.
    """
    for symbol in [PAIR_A, PAIR_B]:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            continue

        for position in positions:
            if position.tp == 0:
                continue
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

def compute_lagged_correlation(rates_a, rates_b, period=100):
    """
    Compute lagged correlation:
    - a_leads_b: correlation of a(t) returns with b(t+1) returns
    - b_leads_a: correlation of b(t) returns with a(t+1) returns

    Return (a_leads_b_corr, b_leads_a_corr)
    """
    if len(rates_a) < period+2 or len(rates_b) < period+2:
        return None, None

    df_a = pd.DataFrame(rates_a, columns=["time","open","high","low","close","tick_volume","spread","real_volume"])
    df_b = pd.DataFrame(rates_b, columns=["time","open","high","low","close","tick_volume","spread","real_volume"])

    a_rets = df_a['close'].pct_change().dropna().tail(period+1).reset_index(drop=True)
    b_rets = df_b['close'].pct_change().dropna().tail(period+1).reset_index(drop=True)

    # a_leads_b: correl a_rets[0:period] with b_rets[1:period+1]
    if len(a_rets) <= period or len(b_rets) <= period:
        return None, None
    a_leads_b = pd.DataFrame({'a': a_rets.iloc[:-1], 'b': b_rets.iloc[1:]})
    b_leads_a = pd.DataFrame({'a': a_rets.iloc[1:], 'b': b_rets.iloc[:-1]})

    corr_a_leads_b = a_leads_b.corr().iloc[0,1]
    corr_b_leads_a = b_leads_a.corr().iloc[0,1]

    return corr_a_leads_b, corr_b_leads_a

def check_breakout(rates):
    """
    Check if there's a breakout in the last LOOKBACK_BREAKOUT bars of the given symbol.
    Similar logic as before:
    - If close is near recent highs, signal BUY
    - If close is near recent lows, signal SELL
    """
    df = pd.DataFrame(rates, columns=["time","open","high","low","close","tick_volume","spread","real_volume"])
    recent = df.tail(LOOKBACK_BREAKOUT)
    current_price = recent['close'].iloc[-1]
    highest = recent['high'].max()
    lowest = recent['low'].min()

    buffer = (highest - lowest)*0.05
    if current_price >= highest - buffer:
        return "BUY"
    if current_price <= lowest + buffer:
        return "SELL"
    return None

def define_trade_parameters(direction, follower_rates):
    """
    Define SL and TP for the follower symbol based on recent volatility.
    Use last 20 bars range as SL distance, TP = 2*SL
    """
    df = pd.DataFrame(follower_rates, columns=["time","open","high","low","close","tick_volume","spread","real_volume"])
    recent = df.tail(20)
    follower_symbol_info = mt5.symbol_info(df['time'].name)
    # Actually 'df['time'].name' isn't correct for symbol_info, we need the symbol name. 
    # We'll just assume we know the symbol name from context.
    # Let's pass symbol as a parameter instead:
    # Actually let's handle it by passing symbol to define_trade_parameters:

def define_trade_parameters_for_symbol(direction, symbol, follower_rates):
    df = pd.DataFrame(follower_rates, columns=["time","open","high","low","close","tick_volume","spread","real_volume"])
    recent = df.tail(20)
    follower_symbol_info = mt5.symbol_info(symbol)
    if follower_symbol_info is None:
        return None, None, None

    point = follower_symbol_info.point
    rng = recent['high'].max() - recent['low'].min()
    sl_distance = rng
    tp_distance = 2*rng

    current_bid = mt5.symbol_info_tick(symbol).bid
    current_ask = mt5.symbol_info_tick(symbol).ask
    if direction == "BUY":
        entry_price = current_ask
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:
        entry_price = current_bid
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance
    return entry_price, sl, tp

def run_dynamic_lead_lag_strategy(timeframe=mt5.TIMEFRAME_M15):
    account_info = mt5.account_info()
    if account_info is None:
        print("[Error] Unable to fetch account info")
        return
    print(f"Account balance: {account_info.balance}, equity: {account_info.equity}")

    if has_daily_drawdown_exceeded():
        print("[Risk] Daily drawdown exceeded. No new trades.")
        return

    # Manage positions
    manage_positions()

    # If we have open or pending trades on either symbol, skip new trades for that symbol
    # We'll only trade on the "follower" symbol once determined
    # But first we need to determine leader/follower relationship:
    rates_a = mt5.copy_rates_from_pos(PAIR_A, timeframe, 0, CORRELATION_PERIOD+LOOKBACK_BREAKOUT+50)
    rates_b = mt5.copy_rates_from_pos(PAIR_B, timeframe, 0, CORRELATION_PERIOD+LOOKBACK_BREAKOUT+50)
    if rates_a is None or rates_b is None:
        print("Not enough data to determine relationship.")
        return

    corr_a_leads_b, corr_b_leads_a = compute_lagged_correlation(rates_a, rates_b, CORRELATION_PERIOD)
    if corr_a_leads_b is None or corr_b_leads_a is None:
        print("Not enough data to compute lagged correlation.")
        return

    print(f"Lagged correlation: {PAIR_A} leads {PAIR_B}: {corr_a_leads_b:.2f}, {PAIR_B} leads {PAIR_A}: {corr_b_leads_a:.2f}")

    # Determine leader and follower
    if corr_a_leads_b > CORRELATION_THRESHOLD and corr_a_leads_b > corr_b_leads_a:
        leader_symbol = PAIR_A
        follower_symbol = PAIR_B
        leader_leads = True
    elif corr_b_leads_a > CORRELATION_THRESHOLD and corr_b_leads_a > corr_a_leads_b:
        leader_symbol = PAIR_B
        follower_symbol = PAIR_A
        leader_leads = True
    else:
        print("No stable lead-lag relationship found above threshold.")
        return

    # Check if we can place a trade on the follower
    if _trades_opened_today[follower_symbol] >= MAX_TRADES_PER_DAY_PER_SYMBOL:
        print(f"Max trades per day reached for {follower_symbol}. No more entries.")
        return

    follower_positions = mt5.positions_get(symbol=follower_symbol)
    follower_orders = mt5.orders_get(symbol=follower_symbol)
    if follower_positions or follower_orders:
        print(f"{follower_symbol} already has positions or pending orders. Skipping new trades.")
        return

    # Check leader breakout
    leader_rates = rates_a if leader_symbol == PAIR_A else rates_b
    direction = check_breakout(leader_rates)
    if direction:
        # If leader breaks out, we trade follower in the same direction
        follower_rates = rates_b if follower_symbol == PAIR_B else rates_a
        entry_price, sl, tp = define_trade_parameters_for_symbol(direction, follower_symbol, follower_rates)
        if entry_price is None:
            return
        lot = DEFAULT_LOT_SIZE
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        result = place_order(follower_symbol, order_type, lot, stop_loss=sl, take_profit=tp)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            _trades_opened_today[follower_symbol] += 1
    else:
        print("No breakout detected in the leader symbol. No trade.")

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

    print(f"Starting Dynamic Lead-Lag Strategy with {PAIR_A} and {PAIR_B}")

    try:
        while True:
            now = datetime.now()
            if now.day != _current_day:
                _current_day = now.day
                reset_daily_equity_baseline()

            run_dynamic_lead_lag_strategy(TIMEFRAME)

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
