import time
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from typing import Dict, Deque
from threading import Thread, Lock
from queue import Queue
import queue

# -------------------
# 1) Configurable Parameters
# -------------------
SYMBOLS = ["EURUSD", "EURGBP", "EURJPY"]
TIMEFRAME = mt5.TIMEFRAME_M15
MIN_GAP_SIZE = 0.0001
DEBUG_MODE = True  # Toggle this to True/False to show/hide detailed prints

# Define default lot sizes for each symbol
DEFAULT_LOT_SIZES = {
    "EURUSD": 2.8,
    "EURGBP": 2,
    "EURJPY": 2
}

@dataclass
class PriceTracker:
    symbol: str
    mid_point: float
    direction: str  # 'BUY' or 'SELL'
    stop_loss: float
    take_profit: float
    price_history: Deque[float]
    crossed_below: bool
    crossed_above: bool
    
    def __init__(self, symbol: str, mid_point: float, direction: str, stop_loss: float, take_profit: float):
        self.symbol = symbol
        self.mid_point = mid_point
        self.direction = direction
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.price_history = deque(maxlen=10)
        self.crossed_below = direction == 'BUY'  # For BUY, we need price to cross from below
        self.crossed_above = direction == 'SELL'  # For SELL, we need price to cross from above
    
    def update_price(self, price: float) -> bool:
        """Returns True if entry condition is met"""
        self.price_history.append(price)
        
        if self.direction == 'BUY':
            if price < self.mid_point:
                self.crossed_below = True
            elif self.crossed_below and price >= self.mid_point:
                # Verify price is within reasonable range of midpoint
                if abs(price - self.mid_point) < (self.take_profit - self.mid_point) * 0.1:  # Within 10% of gap size
                    return True
                self.crossed_below = False  # Reset if we moved too far
        else:  # SELL
            if price > self.mid_point:
                self.crossed_above = True
            elif self.crossed_above and price <= self.mid_point:
                # Verify price is within reasonable range of midpoint
                if abs(price - self.mid_point) < (self.mid_point - self.take_profit) * 0.1:  # Within 10% of gap size
                    return True
                self.crossed_above = False  # Reset if we moved too far
        
        return False

class TickProcessor:
    def __init__(self):
        self.tick_queue = Queue()
        self.running = True
        self.lock = Lock()
        self.thread = Thread(target=self._process_ticks, daemon=True)
        self.thread.start()
    
    def _process_ticks(self):
        while self.running:
            try:
                symbol, tick = self.tick_queue.get(timeout=1)
                with self.lock:
                    if symbol in active_trackers:
                        tracker = active_trackers[symbol]
                        price = tick.ask if tracker.direction == 'BUY' else tick.bid
                        
                        print(f"\r{symbol} - {tracker.direction}: {price:.5f} vs Mid: {tracker.mid_point:.5f}", end='')
                        
                        if tracker.update_price(price):
                            print(f"\nEntry condition met for {symbol} at {price:.5f}")
                            
                            if tracker.direction == 'BUY':
                                result = place_order(symbol, mt5.ORDER_TYPE_BUY, 
                                                  DEFAULT_LOT_SIZES[symbol],
                                                  tracker.stop_loss, 
                                                  tracker.take_profit, 
                                                  "FVG_BUY")
                            else:
                                result = place_order(symbol, mt5.ORDER_TYPE_SELL, 
                                                  DEFAULT_LOT_SIZES[symbol],
                                                  tracker.stop_loss, 
                                                  tracker.take_profit, 
                                                  "FVG_SELL")
                            
                            if result:
                                del active_trackers[symbol]
            except queue.Empty:
                continue
    
    def add_tick(self, symbol: str, tick):
        self.tick_queue.put((symbol, tick))
    
    def stop(self):
        self.running = False
        self.thread.join()

# Create global tick processor
tick_processor = TickProcessor()

def on_tick(symbol: str):
    """Callback function for tick updates"""
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        tick_processor.add_tick(symbol, tick)

def subscribe_to_ticks(symbols):
    """Subscribe to tick updates for given symbols"""
    for symbol in symbols:
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select symbol {symbol}")
            continue
        print(f"Subscribed to {symbol} ticks")

def check_price_updates():
    """Check current prices against active trackers"""
    with tick_processor.lock:
        for symbol in list(active_trackers.keys()):
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
            
            tracker = active_trackers[symbol]
            print(f"\r{symbol} - Current Ask: {tick.ask:.5f} Bid: {tick.bid:.5f} vs Mid: {tracker.mid_point:.5f}", end='')
            tick_processor.add_tick(symbol, tick)

# Now declare the global variable after the class definition
active_trackers: Dict[str, PriceTracker] = {}

# -------------------
# 2) MT5 Utility Functions
# -------------------
def initialize_mt5(login=None, password=None, server=None):
    """Initialize and log into the MT5 terminal."""
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

def place_order(symbol, order_type, lot_size, stop_loss=None, take_profit=None, comment="FVG"):
    """Places a single order with the specified parameters"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[Error] Symbol {symbol} not found!")
        return None

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"[Error] Failed to select symbol {symbol}")
            return None

    # Validate lot size
    lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
    print(f"Adjusted lot size: {lot_size}")

    # Get the filling mode
    filling_type = (mt5.ORDER_FILLING_FOK if symbol_info.filling_mode & mt5.ORDER_FILLING_FOK else
                   mt5.ORDER_FILLING_IOC if symbol_info.filling_mode & mt5.ORDER_FILLING_IOC else
                   mt5.ORDER_FILLING_RETURN)

    # Get current price
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    
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
    for attempt in range(3):
        request["price"] = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Order successfully placed! Ticket: {result.order}, Lots: {lot_size}")
            return result
        
        print(f"Order attempt {attempt + 1} failed, retrying...")
        time.sleep(1)
    
    print("Failed to place order after maximum attempts")
    return None

# -------------------
# 3) FVG Strategy Logic
# -------------------
def is_gap_filled(gap, current_low, current_high):
    """Check if a gap has been filled by price action"""
    if gap['Type'] == 'Bullish':
        return current_low <= gap['FVG_Start']
    else:  # Bearish
        return current_high >= gap['FVG_Start']

def identify_fair_value_gaps(df, current_idx, open_gaps, min_gap_size=MIN_GAP_SIZE):
    """Identify new FVG gaps at the current index"""
    new_gaps = []
    
    if current_idx < 2:
        return new_gaps
    
    prev_high, prev_low = df.iloc[current_idx - 2]['High'], df.iloc[current_idx - 2]['Low']
    current_high, current_low = df.iloc[current_idx]['High'], df.iloc[current_idx]['Low']
    current_time = df.iloc[current_idx]['Date']
    
    if current_low > prev_high:  # Bullish gap
        gap_size = current_low - prev_high
        if gap_size > min_gap_size:
            new_gap = {
                'index': current_idx,
                'Type': 'Bullish',
                'FVG_Start': prev_high,
                'FVG_End': current_low,
                'Start_Time': current_time,
                'Used': False
            }
            new_gaps.append(new_gap)
            open_gaps.append(new_gap)
    
    elif prev_low > current_high:  # Bearish gap
        gap_size = prev_low - current_high
        if gap_size > min_gap_size:
            new_gap = {
                'index': current_idx,
                'Type': 'Bearish',
                'FVG_Start': prev_low,
                'FVG_End': current_high,
                'Start_Time': current_time,
                'Used': False
            }
            new_gaps.append(new_gap)
            open_gaps.append(new_gap)
    
    return new_gaps

def debug_print(*args, **kwargs):
    """Utility function for debug printing"""
    if DEBUG_MODE:
        print(*args, **kwargs)

def check_fvg_signals(symbol, timeframe):
    """Check for FVG trading signals and create price trackers"""
    timeframe_minutes = {
        mt5.TIMEFRAME_M1: 1,
        mt5.TIMEFRAME_M5: 5,
        mt5.TIMEFRAME_M15: 15,
        mt5.TIMEFRAME_M30: 30,
        mt5.TIMEFRAME_H1: 60,
        mt5.TIMEFRAME_H4: 240,
        mt5.TIMEFRAME_D1: 1440,
    }.get(timeframe, 15)
    
    bars_needed = int((60 * 24 * 60) / timeframe_minutes)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars_needed)
    if rates is None or len(rates) < 3:
        print(f"Failed to get rates for {symbol}")
        return None, None, None

    df = pd.DataFrame(rates)
    df.rename(columns={"time": "Date", "open": "Open", "high": "High",
                      "low": "Low", "close": "Close"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    df['symbol'] = symbol

    # Maintain list of open gaps
    open_gaps = []
    
    # Scan historical data for gaps
    for i in range(2, len(df)):
        identify_fair_value_gaps(df, i, open_gaps)
        current_low, current_high = df.iloc[i]['Low'], df.iloc[i]['High']
        # Remove filled gaps
        open_gaps = [gap for gap in open_gaps if not is_gap_filled(gap, current_low, current_high)]

    # Print open gaps information
    print(f"\nFound {len(open_gaps)} open gaps for {symbol}:")
    for gap in open_gaps:
        mid_point = (gap['FVG_Start'] + gap['FVG_End']) / 2
        gap_size = abs(gap['FVG_Start'] - gap['FVG_End'])
        print(f"- {gap['Type']} gap from {gap['Start_Time'].strftime('%Y-%m-%d %H:%M')}:")
        print(f"  Start: {gap['FVG_Start']:.5f}")
        print(f"  End: {gap['FVG_End']:.5f}")
        print(f"  Mid-point: {mid_point:.5f}")
        print(f"  Size: {gap_size:.5f}")
        print(f"  Used: {gap['Used']}")

    if not open_gaps:
        print(f"No open gaps found for {symbol}")
        return None, None, None

    # Check current price against open gaps
    current_idx = len(df) - 1
    current_price = df.iloc[current_idx]['Close']
    prev_price = df.iloc[current_idx - 1]['Close']

    print(f"\nCurrent price for {symbol}: {current_price:.5f}")
    print(f"Previous price for {symbol}: {prev_price:.5f}")

    # Look for entry signals on any open gap
    for gap in open_gaps:
        if gap['Used']:
            continue

        mid_point = (gap['FVG_Start'] + gap['FVG_End']) / 2
        gap_size = abs(gap['FVG_Start'] - gap['FVG_End'])

        print(f"\nGap Analysis for {symbol}:")
        print(f"Gap Start: {gap['FVG_Start']:.5f}")
        print(f"Gap End: {gap['FVG_End']:.5f}")
        print(f"Mid Point: {mid_point:.5f}")
        print(f"Raw Gap Size: {gap_size:.5f}")
        print(f"1.5x Gap Size: {1.5 * gap_size:.5f}")
        print(f"3x Gap Size: {3 * gap_size:.5f}")

        if gap['Type'] == 'Bullish' and symbol not in active_trackers:
            stop_loss = mid_point - 1 * gap_size 
            take_profit = mid_point + 3 * gap_size     
            
            gap['Used'] = True
            active_trackers[symbol] = PriceTracker(
                symbol=symbol,
                mid_point=mid_point,
                direction='BUY',
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            print(f"\nTracking Bullish signal on {symbol}:")
            print(f"Mid Point: {mid_point:.5f}")
            print(f"Stop Loss: {stop_loss:.5f} ({abs(mid_point - stop_loss):.5f} pips from mid)")
            print(f"Take Profit: {take_profit:.5f} ({abs(take_profit - mid_point):.5f} pips from mid)")
            
        elif gap['Type'] == 'Bearish' and symbol not in active_trackers:
            stop_loss = mid_point + 0.5 * gap_size  # Reduced from 1.5x to 0.5x
            take_profit = mid_point - gap_size      # Reduced from 3x to 1x
            
            gap['Used'] = True
            active_trackers[symbol] = PriceTracker(
                symbol=symbol,
                mid_point=mid_point,
                direction='SELL',
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            print(f"\nTracking Bearish signal on {symbol}:")
            print(f"Mid Point: {mid_point:.5f}")
            print(f"Stop Loss: {stop_loss:.5f} ({abs(mid_point - stop_loss):.5f} pips from mid)")
            print(f"Take Profit: {take_profit:.5f} ({abs(take_profit - mid_point):.5f} pips from mid)")

    return None, None, None

def run_fvg_strategy(timeframe):
    """Main strategy function that handles multiple symbols"""
    for symbol in SYMBOLS:
        print(f"\nChecking FVG signals for {symbol}...")
        signal, stop_loss, take_profit = check_fvg_signals(symbol, timeframe)
        
        if not signal:
            continue

        lot = DEFAULT_LOT_SIZES[symbol]
        print(f"Attempting to open {symbol} with lot size: {lot}")  # Keep important signals visible

        if signal == "BUY":
            place_order(symbol, mt5.ORDER_TYPE_BUY, lot, stop_loss, take_profit, "FVG_BUY")
        elif signal == "SELL":
            place_order(symbol, mt5.ORDER_TYPE_SELL, lot, stop_loss, take_profit, "FVG_SELL")

# -------------------
# 4) Main Loop
# -------------------
def main():
    # Update with your credentials
    MY_LOGIN = 5032225458
    MY_PASSWORD = "@6IzQoQz"
    MY_SERVER = "MetaQuotes-Demo"

    initialize_mt5(login=MY_LOGIN, password=MY_PASSWORD, server=MY_SERVER)
    print(f"Starting FVG strategy on {SYMBOLS}, timeframe={TIMEFRAME}...")

    # Subscribe to tick updates for all symbols
    subscribe_to_ticks(SYMBOLS)

    try:
        while True:
            now = datetime.now()
            
            # Calculate next candle time
            next_candle = now.replace(second=0, microsecond=0)
            minutes_to_add = 15 - (next_candle.minute % 15)
            if minutes_to_add == 15 and now.second == 0:
                minutes_to_add = 0
            next_candle = next_candle + pd.Timedelta(minutes=minutes_to_add)
            
            # If it's time for a new candle check
            if now >= next_candle:
                run_fvg_strategy(TIMEFRAME)
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Checked all symbols. "
                      f"Active trackers: {list(active_trackers.keys())}. "
                      f"Next candle at {next_candle.strftime('%H:%M:%S')}")
            
            # Actively check for new ticks
            if active_trackers:
                check_price_updates()
                
            # Small sleep to prevent CPU overload
            time.sleep(0.1)  # 100ms delay

    except KeyboardInterrupt:
        print("\nManual shutdown signal received.")
    finally:
        tick_processor.stop()
        mt5.shutdown()
        print("MetaTrader5 connection shut down.")

if __name__ == "__main__":
    main()
