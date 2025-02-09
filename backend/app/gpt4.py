import time
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime

# -------------------
# Configurable Parameters
# -------------------
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = mt5.TIMEFRAME_M15
NUM_BARS = 200           # Number of bars to compute PCA
PC_ZSCORE_THRESHOLD = 2.0 # Z-score threshold for extremes
DEFAULT_LOT = 1.0
MAX_DAILY_DD_PERCENT = 3.0
MAX_TRADES_PER_DAY = 1

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
        print(f"[Drawdown Exceeded] Current equity {current_equity:.2f} < {threshold_equity:.2f}")
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

def place_order(symbol, order_type, lot_size, sl=None, tp=None, comment="PCA_TRADE"):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[Error] Symbol {symbol} not found!")
        return None
    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)
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
    if sl is not None:
        request["sl"] = sl
    if tp is not None:
        request["tp"] = tp
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Order placed: {symbol}, {order_type}, lot={lot_size}, SL={sl}, TP={tp}")
    else:
        print(f"Failed to place order: {result.retcode if result else mt5.last_error()}")
    return result

def close_all_positions():
    positions = mt5.positions_get()
    if not positions:
        return
    for pos in positions:
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(pos.symbol).ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "comment": "PCA_EXIT"
        }
        mt5.order_send(request)
        print(f"Closed position on {pos.symbol}")

def get_returns_matrix(symbols, timeframe, bars=200):
    """
    Returns a (bars-1, len(symbols)) matrix of returns
    """
    data = []
    for sym in symbols:
        rates = mt5.copy_rates_from_pos(sym, timeframe, 0, bars)
        if rates is None or len(rates)<2:
            return None
        df = pd.DataFrame(rates)
        df['ret'] = df['close'].pct_change()
        data.append(df['ret'].iloc[1:].values)  # discard first NaN
    # shape: len(symbols), bars-1
    returns = np.column_stack(data) # shape: (bars-1, len(symbols)) after transpose
    # Actually np.column_stack puts arrays as columns
    # we appended each ret array as rows, we want symbols in columns:
    # data list is symbols dimension, each data[i] is a 1D array of length bars-1
    # column_stack(data) => (bars-1, len(symbols)) which is correct: rows=times, cols=symbols
    return returns

def pca(returns):
    """
    Compute PCA using SVD on covariance matrix.
    returns: (T, N) matrix (T times, N symbols)
    """
    # Mean-center
    R = returns - np.nanmean(returns, axis=0)
    # Compute covariance
    C = np.cov(R, rowvar=False)  # shape (N, N)
    # SVD
    U, S, Vt = np.linalg.svd(C)
    # principal components in Vt
    # PC loadings = Vt
    # Scores = R @ V (Projections)
    return Vt, S, np.nanmean(R, axis=0), np.nanstd(R, axis=0)

def compute_pc_score(returns, loadings):
    """
    Project today's returns onto the first PC.
    returns is (T, N), take last row to represent today's returns
    loadings is from PCA, loadings[0,:] is PC1
    """
    R = returns - np.nanmean(returns, axis=0)
    today_ret = R[-1, :]  # last row
    pc1 = loadings[0, :]
    score = np.dot(today_ret, pc1)
    return score

def run_pca_strategy():
    account_info = mt5.account_info()
    if account_info is None:
        print("[Error] Unable to fetch account info")
        return
    print(f"Balance: {account_info.balance}, Equity: {account_info.equity}")

    if has_daily_drawdown_exceeded():
        print("Daily drawdown exceeded. No trades.")
        return

    global _trades_opened_today
    if _trades_opened_today >= MAX_TRADES_PER_DAY:
        print("Max trades per day reached.")
        return

    # Close any positions at the start of a new day or strategy cycle if you prefer fresh start
    # close_all_positions() # uncomment if desired daily reset of positions

    retmat = get_returns_matrix(SYMBOLS, TIMEFRAME, bars=NUM_BARS)
    if retmat is None:
        print("Not enough data for PCA.")
        return

    # PCA
    Vt, S, mean_ret, std_ret = pca(retmat)
    score = compute_pc_score(retmat, Vt)
    # Compute historical scores to get mean and std of PC1 projection
    # All projections:
    R = retmat - np.nanmean(retmat, axis=0)
    pc1 = Vt[0,:]
    all_scores = R @ pc1
    score_mean = np.mean(all_scores)
    score_std = np.std(all_scores)

    zscore = (score - score_mean)/score_std
    print(f"PC1 z-score: {zscore:.2f}")

    if abs(zscore) > PC_ZSCORE_THRESHOLD:
        # Extreme condition
        # If zscore > 0 => PC1 is positive extreme => Sell the pairs with positive loading, buy those with negative loading
        # If zscore < 0 => PC1 negative extreme => Buy pairs with positive loading, sell those with negative

        # Determine sign of each symbol's loading on PC1
        loadings = Vt[0,:]  # PC1 loadings per symbol
        # We'll split a small lot among symbols based on sign
        # Let's say we do a simplistic approach: 
        # For zscore>0: we short symbols with positive loadings, long symbols with negative loadings
        # For zscore<0: we do the opposite
        direction_factor = 1 if zscore < 0 else -1 
        # direction_factor=1 means positive loading => BUY, negative loading => SELL
        # direction_factor=-1 means positive loading => SELL, negative loading => BUY

        # Risk management: define SL/TP based on average volatility of returns:
        avg_abs_ret = np.nanmean(np.abs(retmat))
        # If avg_abs_ret ~0.0005 means about 5 pips for EURUSD-level pairs
        # We'll just pick a SL = current_price * avg_abs_ret * 2, TP = SL * 2 for demonstration
        # We'll assume similar magnitude for all pairs
        # This is naive, but just to show concept
        positions = []
        for i, sym in enumerate(SYMBOLS):
            load = loadings[i]
            sym_info = mt5.symbol_info_tick(sym)
            if sym_info is None:
                continue
            price = sym_info.bid
            sl_dist = price * avg_abs_ret * 2
            tp_dist = sl_dist * 2
            if load * direction_factor > 0:
                # BUY this symbol
                sl = price - sl_dist
                tp = price + tp_dist
                res = place_order(sym, mt5.ORDER_TYPE_BUY, DEFAULT_LOT, sl, tp)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    positions.append(res.order)
            else:
                # SELL this symbol
                sl = price + sl_dist
                tp = price - tp_dist
                res = place_order(sym, mt5.ORDER_TYPE_SELL, DEFAULT_LOT, sl, tp)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    positions.append(res.order)

        if positions:
            _trades_opened_today += 1
            print(f"Opened a PCA-based market-neutral trade basket. Orders: {positions}")
        else:
            print("No orders placed (some error occurred).")

    else:
        print("No extreme deviation in PC1. No trade.")

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
    MY_LOGIN = 12345678
    MY_PASSWORD = "password"
    MY_SERVER = "Broker-Server"

    initialize_mt5(login=MY_LOGIN, password=MY_PASSWORD, server=MY_SERVER)

    global _current_day
    _current_day = datetime.now().day
    reset_daily_equity_baseline()

    print("Starting PCA-based Market-Neutral Strategy on:", SYMBOLS)

    try:
        while True:
            now = datetime.now()
            if now.day != _current_day:
                _current_day = now.day
                reset_daily_equity_baseline()
                # Also consider closing all positions daily for a fresh start
                # close_all_positions()

            run_pca_strategy()

            next_candle = get_next_candle_time(TIMEFRAME)
            wait_seconds = (next_candle - datetime.now()).total_seconds()
            wait_seconds = max(1, wait_seconds+1)
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Cycle done. Sleeping {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)

    except KeyboardInterrupt:
        print("Manual shutdown.")
    finally:
        shutdown_mt5()

if __name__ == "__main__":
    main()
