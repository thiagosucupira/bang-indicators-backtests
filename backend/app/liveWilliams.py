import time
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime

# -------------------
# 1) Configurable Risk Parameters
# -------------------
RISK_PER_TRADE_PERCENT = 2.0   # Keeping it for reference; no longer used
MAX_DAILY_DD_PERCENT   = 3.0   # If equity drops more than 3% in a day, stop new trades
DEFAULT_LOT_SIZE = 2         # Changed to 5.0 lots
MIN_RISK_REWARD = 1.5        # Minimum risk:reward ratio for new trades

SYMBOLS = ["EURUSD", 
           #"EURGBP", 
           "EURJPY",
           "XAUUSD"]  # Replace the single SYMBOL constant
TIMEFRAME = mt5.TIMEFRAME_M15
WILLIAMS_R_PERIOD = 14
CHECK_INTERVAL = 900  # seconds to wait between checks

# Define default lot sizes for each symbol
DEFAULT_LOT_SIZES = {
    "EURUSD": 2,
  #  "EURGBP": 2,
    "XAUUSD": 1,
    "EURJPY": 2  # Adjust this value based on your preference
}

# -------------------
# 2) Williams %R Function
# -------------------
def calculate_williams_r(df, period=14):
    """
    Calculates Williams %R for the given DataFrame.
    df must have columns: 'High', 'Low', 'Close'.
    """
    if df.empty:
        return df

    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    williams_r = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
    df['Williams_%R'] = williams_r
    return df

# -------------------
# 3) MT5 Utility Functions
# -------------------
def initialize_mt5(login=None, password=None, server=None):
    """Initialize and log into the MT5 terminal."""
    if not mt5.initialize():
        print(f"initialize() failed. Error code: {mt5.last_error()}")
        quit()
    print("MetaTrader5 package initialized.")

    # Add terminal info checks
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
    """Shut down the MT5 connection cleanly."""
    mt5.shutdown()
    print("MetaTrader5 connection shut down.")

def place_order(symbol, order_type, lot_size, stop_loss=None, take_profit=None, comment="WR"):
    """
    Places a single order with the specified lot size
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[Error] Symbol {symbol} not found!")
        return None

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"[Error] Failed to select symbol {symbol}")
            return None

    # Add volume validation
    print(f"Attempting to place order with volume: {lot_size}")
    print(f"Symbol minimum lot: {symbol_info.volume_min}")
    print(f"Symbol maximum lot: {symbol_info.volume_max}")
    print(f"Symbol lot step: {symbol_info.volume_step}")

    # Validate lot size
    if lot_size < symbol_info.volume_min:
        print(f"[Error] Lot size {lot_size} is below minimum {symbol_info.volume_min}")
        return None
    if lot_size > symbol_info.volume_max:
        print(f"[Error] Lot size {lot_size} is above maximum {symbol_info.volume_max}")
        return None

    # Round lot size to nearest valid step
    lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
    print(f"Adjusted lot size: {lot_size}")

    # Get the filling mode for the symbol
    filling_type = None
    if symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
        filling_type = mt5.ORDER_FILLING_FOK
    elif symbol_info.filling_mode & mt5.ORDER_FILLING_IOC:
        filling_type = mt5.ORDER_FILLING_IOC
    else:
        filling_type = mt5.ORDER_FILLING_RETURN  # Use this as last resort

    print(f"Using filling type: {filling_type}")

    # Get symbol info for minimum stops distance
    min_stops_distance = symbol_info.trade_stops_level * symbol_info.point
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid

    print(f"Minimum stops distance: {min_stops_distance}")
    print(f"Original SL: {stop_loss}, Original TP: {take_profit}")

    # Adjust SL and TP to respect minimum distance  
    if stop_loss is not None:
        if order_type == mt5.ORDER_TYPE_BUY:
            min_sl = price - min_stops_distance
            stop_loss = min(stop_loss, min_sl)
        else:  # SELL
            min_sl = price + min_stops_distance
            stop_loss = max(stop_loss, min_sl)

    if take_profit is not None:
        if order_type == mt5.ORDER_TYPE_BUY:
            min_tp = price + min_stops_distance
            take_profit = max(take_profit, min_tp)
        else:  # SELL
            min_tp = price - min_stops_distance
            take_profit = min(take_profit, min_tp)

    print(f"Adjusted SL: {stop_loss}, Adjusted TP: {take_profit}")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot_size),
        "type": order_type,
        "magic": 123456,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
        "price": price,
        "deviation": 30,
        "sl": stop_loss if stop_loss is not None else 0.0,
        "tp": take_profit if take_profit is not None else 0.0
    }

    # Try up to 3 times to place the order
    max_tries = 3
    for attempt in range(max_tries):
        print(f"Order attempt {attempt + 1} of {max_tries}")
        
        # Update the price for each attempt
        request["price"] = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        
        # Print order details for debugging
        print(f"Placing order with:")
        print(f"Price: {request['price']}")
        print(f"Stop Loss: {request['sl']}")
        print(f"Take Profit: {request['tp']}")
        
        result = mt5.order_send(request)
        if result is None:
            print(f"order_send() failed, error code: {mt5.last_error()}")
            continue
        elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
            print("Requote received, trying again with new price...")
            continue
        elif result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"order_send() failed, retcode={result.retcode}")
            print(f"Request: {request}")
            continue
        else:
            print(f"Order successfully placed! Ticket: {result.order}, Lots: {lot_size}")
            print(f"Entry Price: {result.price}")
            print(f"Stop Loss: {stop_loss}")
            print(f"Take Profit: {take_profit}")
            return result
            
    print("Failed to place order after maximum attempts")
    return None

# -------------------
# 4) Real-Time Strategy Logic
# -------------------
def close_partial_position(position, close_percentage):
    """
    Closes a specified percentage of an open position
    Returns True if successful, False otherwise
    """
    if position is None:
        return False
        
    close_volume = position.volume * (close_percentage / 100)
    # Round to symbol's volume step
    symbol_info = mt5.symbol_info(position.symbol)
    if symbol_info:
        close_volume = round(close_volume / symbol_info.volume_step) * symbol_info.volume_step
    
    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": close_volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
        "comment": "PARTIAL_CLOSE",
    }
    
    result = mt5.order_send(close_request)
    return result and result.retcode == mt5.TRADE_RETCODE_DONE

def modify_sl_to_breakeven(position):
    """
    Modifies the stop loss of a position to breakeven
    Returns True if successful, False otherwise
    """
    if position is None:
        return False
        
    modify_request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": position.ticket,
        "sl": position.price_open,  # Set SL to entry price
        "tp": position.tp  # Keep existing take profit
    }
    
    result = mt5.order_send(modify_request)
    return result and result.retcode == mt5.TRADE_RETCODE_DONE

def check_williams_r_signal(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M15, period=14):
    """
    Modified to include partial close and breakeven stop loss logic
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 50)
    if rates is None or len(rates) < 2:
        return None, None, None

    df = pd.DataFrame(rates)
    df.rename(columns={"time": "Date", "open": "Open", "high": "High",
                      "low": "Low", "close": "Close"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], unit='s')

    df = calculate_williams_r(df, period=period)
    if 'Williams_%R' not in df.columns:
        return None, None, None

    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    # Add Williams %R values to console output
    print(f"Current Williams %R: {today['Williams_%R']:.2f}")
    print(f"Previous Williams %R: {yesterday['Williams_%R']:.2f}")

    # Calculate daily range for dynamic SL/TP
    daily_range = yesterday['High'] - yesterday['Low']

    # Add this before the signal checks
    print(f"\nDaily Range Analysis:")
    print(f"Yesterday High: {yesterday['High']:.5f}")
    print(f"Yesterday Low: {yesterday['Low']:.5f}")
    print(f"Daily Range: {daily_range:.5f}")

    # Get spread information
    symbol_info_tick = mt5.symbol_info_tick(symbol)
    spread = symbol_info_tick.ask - symbol_info_tick.bid
    min_distance = 3 * spread
    
    print(f"Current spread: {spread:.5f}")
    print(f"Minimum required distance (3x spread): {min_distance:.5f}")

    # Check for entry signals with risk:reward validation
    if yesterday['Williams_%R'] < -80 and today['Williams_%R'] >= -80:
        entry_price = mt5.symbol_info_tick(symbol).ask
        
        # Ensure SL is at least 3x spread away from entry
        stop_loss = min(entry_price - daily_range, entry_price - min_distance)
        take_profit = entry_price + (2 * daily_range)
        
        # Ensure TP is at least 3x spread away from entry
        if take_profit - entry_price < min_distance:
            take_profit = entry_price + min_distance
        
        # Calculate risk:reward ratio
        risk = abs(entry_price - stop_loss)  # Use abs() to ensure positive value
        reward = abs(take_profit - entry_price)  # Use abs() to ensure positive value
        risk_reward_ratio = reward/risk
        
        print(f"BUY Signal Analysis:")
        print(f"Entry: {entry_price:.5f}")
        print(f"SL: {stop_loss:.5f} (Risk: {risk:.5f} pips)")
        print(f"TP: {take_profit:.5f} (Reward: {reward:.5f} pips)")
        print(f"R:R Ratio: {risk_reward_ratio:.2f}")
        
        if risk_reward_ratio < MIN_RISK_REWARD:
            print(f"[Skip] BUY signal - Poor Risk:Reward {risk_reward_ratio:.2f} < {MIN_RISK_REWARD}")
            return None, None, None
            
        return "BUY", stop_loss, take_profit
        
    if yesterday['Williams_%R'] > -20 and today['Williams_%R'] <= -20:
        entry_price = mt5.symbol_info_tick(symbol).bid
        
        # Ensure SL is at least 3x spread away from entry
        stop_loss = max(entry_price + daily_range, entry_price + min_distance)
        take_profit = entry_price - (2 * daily_range)
        
        # Ensure TP is at least 3x spread away from entry
        if entry_price - take_profit < min_distance:
            take_profit = entry_price - min_distance
        
        # Calculate risk:reward ratio
        risk = abs(stop_loss - entry_price)  # Use abs() to ensure positive value
        reward = abs(entry_price - take_profit)  # Use abs() to ensure positive value
        risk_reward_ratio = reward/risk
        
        print(f"SELL Signal Analysis:")
        print(f"Entry: {entry_price:.5f}")
        print(f"SL: {stop_loss:.5f} (Risk: {risk:.5f} pips)")
        print(f"TP: {take_profit:.5f} (Reward: {reward:.5f} pips)")
        print(f"R:R Ratio: {risk_reward_ratio:.2f}")
        
        if risk_reward_ratio < MIN_RISK_REWARD:
            print(f"[Skip] SELL signal - Poor Risk:Reward {risk_reward_ratio:.2f} < {MIN_RISK_REWARD}")
            return None, None, None
            
        return "SELL", stop_loss, take_profit

    # Check for Williams %R based exit signals
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for position in positions:
            if position.type == mt5.POSITION_TYPE_BUY:
                # Exit long if Williams %R crosses above -20 (overbought)
                if yesterday['Williams_%R'] <= -20 and today['Williams_%R'] > -20:
                    return "CLOSE_BUY", None, None
                    
            elif position.type == mt5.POSITION_TYPE_SELL:
                # Exit short if Williams %R crosses below -80 (oversold)
                if yesterday['Williams_%R'] >= -80 and today['Williams_%R'] < -80:
                    return "CLOSE_SELL", None, None

    # Check for partial close opportunities
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for position in positions:
            current_price = mt5.symbol_info_tick(symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
            
            # Calculate price movement
            if position.type == mt5.POSITION_TYPE_BUY:
                price_movement = current_price - position.price_open
                total_target = position.tp - position.price_open
            else:
                price_movement = position.price_open - current_price
                total_target = position.price_open - position.tp
            
            # Check if we've reached 50% of the target
            if total_target != 0:  # Avoid division by zero
                movement_percentage = (price_movement / abs(total_target)) * 100
                
                if movement_percentage >= 50 and position.sl != position.price_open:
                    print(f"Position reached 50% of target. Closing half and moving SL to breakeven.")
                    if close_partial_position(position, 50):
                        modify_sl_to_breakeven(position)

    return None, None, None

def run_williams_r_strategy(timeframe=mt5.TIMEFRAME_M15):
    """
    Main strategy function that now handles multiple symbols
    """
    # Check account info
    account_info = mt5.account_info()
    if account_info is None:
        print("[Error] Unable to fetch account info")
        return

    if has_daily_drawdown_exceeded():
        print("[Risk] Daily drawdown exceeded. Not opening new trades.")
        return

    # Loop through each symbol
    for symbol in SYMBOLS:
        print(f"\nChecking signals for {symbol}...")
        signal, stop_loss, take_profit = check_williams_r_signal(symbol, timeframe, period=WILLIAMS_R_PERIOD)
        if not signal:
            continue

        # Handle exit signals
        if signal == "CLOSE_BUY":
            positions = mt5.positions_get(symbol=symbol)
            for position in positions:
                if position.type == mt5.POSITION_TYPE_BUY:
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": position.volume,
                        "type": mt5.ORDER_TYPE_SELL,
                        "position": position.ticket,
                        "price": mt5.symbol_info_tick(symbol).bid,
                        "comment": "WR_EXIT",
                    }
                    mt5.order_send(close_request)
            continue

        elif signal == "CLOSE_SELL":
            positions = mt5.positions_get(symbol=symbol)
            for position in positions:
                if position.type == mt5.POSITION_TYPE_SELL:
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": position.volume,
                        "type": mt5.ORDER_TYPE_BUY,
                        "position": position.ticket,
                        "price": mt5.symbol_info_tick(symbol).ask,
                        "comment": "WR_EXIT",
                    }
                    mt5.order_send(close_request)
            continue

        # Handle entry signals
        lot = DEFAULT_LOT_SIZES[symbol]
        print(f"Attempting to open {symbol} with lot size: {lot}")

        if signal == "BUY":
            if lot <= 0:
                print("[Warning] Lot size calculation returned 0 or negative. Trade skipped.")
                continue

            place_order(symbol, mt5.ORDER_TYPE_BUY, lot, stop_loss=stop_loss, take_profit=take_profit,
                       comment="WR_BUY")

        elif signal == "SELL":
            if lot <= 0:
                print("[Warning] Lot size calculation returned 0 or negative. Trade skipped.")
                continue

            place_order(symbol, mt5.ORDER_TYPE_SELL, lot, stop_loss=stop_loss, take_profit=take_profit,
                       comment="WR_SELL")

# -------------------
# 5) Daily Drawdown Logic
# -------------------
_daily_start_equity = None
_daily_drawdown_exceeded = False
_current_day = None

def reset_daily_equity_baseline():
    global _daily_start_equity, _daily_drawdown_exceeded
    account_info = mt5.account_info()
    if account_info is not None:
        _daily_start_equity = account_info.equity
        _daily_drawdown_exceeded = False
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

# Add this new function after the existing imports
def get_next_candle_time(timeframe):
    """Calculate the timestamp of the next candle based on the timeframe."""
    now = datetime.now()
    
    # Convert timeframe (in minutes) from MT5 constant
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
        minutes = 15  # default to M15 if unknown timeframe
    
    # Calculate next candle time
    minutes_to_add = minutes - (now.minute % minutes)
    if minutes_to_add == minutes and now.second == 0:
        minutes_to_add = 0
    
    next_candle = now.replace(second=0, microsecond=0) + pd.Timedelta(minutes=minutes_to_add)
    return next_candle

# -------------------
# 6) Main Loop
# -------------------
def main():
    # Update with your credentials
    MY_LOGIN = 5032225458
    MY_PASSWORD = "@6IzQoQz"
    MY_SERVER = "MetaQuotes-Demo"  # Update this with your broker's server

    initialize_mt5(login=MY_LOGIN, password=MY_PASSWORD, server=MY_SERVER)

    global _current_day
    _current_day = datetime.now().day
    reset_daily_equity_baseline()

    print(f"Starting Williams %R strategy on {SYMBOLS}, timeframe={TIMEFRAME} with risk management...")

    try:
        while True:
            now = datetime.now()
            if now.day != _current_day:
                _current_day = now.day
                reset_daily_equity_baseline()

            # Run the strategy (now without symbol parameter)
            run_williams_r_strategy(TIMEFRAME)

            # Calculate time until next candle
            next_candle = get_next_candle_time(TIMEFRAME)
            wait_seconds = (next_candle - datetime.now()).total_seconds()
            
            # Add 1 second buffer to ensure the candle is fully formed
            wait_seconds = max(1, wait_seconds + 1)
            
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Checked all symbols for signals. Next candle at {next_candle.strftime('%H:%M:%S')}, sleeping {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)

    except KeyboardInterrupt:
        print("Manual shutdown signal received.")
    finally:
        shutdown_mt5()

if __name__ == "__main__":
    main()
