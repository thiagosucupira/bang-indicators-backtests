import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import logging
import matplotlib.dates as mdates
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to calculate the daily range between 13:30 and 14:30
def calculate_daily_range(df):
    if df.empty:
        logger.warning("Received an empty DataFrame in calculate_daily_range.")
        return df.copy()

    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # First, set the 'Date' column as the index if it exists
    if 'Date' in df.columns:
        logger.info("Setting 'Date' column as index.")
        try:
            df.set_index('Date', inplace=True)
        except Exception as e:
            logger.error(f"Error setting Date as index: {e}")
            return df

    # Ensure the DataFrame has a DateTime index with timezone awareness
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.info("Converting index to DateTime with UTC timezone.")
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception as e:
            logger.error(f"Error converting index to DateTime: {e}")
            return df

    # Log the index after initial conversion
    logger.info(f"DataFrame index after initial conversion: {df.index.min()} to {df.index.max()}")

    # Create a boolean mask for the time range 13:30 to 14:30
    try:
        start_time = pd.to_datetime('13:30').time()
        end_time = pd.to_datetime('14:30').time()
        mask = (df.index.time >= start_time) & (df.index.time <= end_time)
        range_df = df.loc[mask]
        logger.info(f"Filtered data between {start_time} and {end_time}. Rows before: {len(df)}, after: {len(range_df)}")
    except Exception as e:
        logger.error(f"Error in time range filtering: {e}")
        return df

    # Calculate Range High and Range Low for each day based on range_df
    if not range_df.empty:
        try:
            # Group by date (floor to day to handle timezone properly)
            daily_range = range_df.groupby(range_df.index.floor('D')).agg({
                'High': 'max',
                'Low': 'min'
            }).rename(columns={'High': 'Range_High', 'Low': 'Range_Low'})
            
            logger.info(f"Calculated daily ranges for {len(daily_range)} days.")
        except Exception as e:
            logger.error(f"Error in grouping to calculate daily range: {e}")
            return df

        # Assign Range_High and Range_Low via mapping to preserve the original index
        try:
            df['Range_High'] = df.index.floor('D').map(daily_range['Range_High'])
            df['Range_Low'] = df.index.floor('D').map(daily_range['Range_Low'])
            logger.info("Assigned Range_High and Range_Low via mapping.")
        except Exception as e:
            logger.error(f"Error assigning Range_High and Range_Low: {e}")
            return df
    else:
        logger.warning("No data within the specified time range (13:30-14:30). Setting Range_High and Range_Low to NaN.")
        df['Range_High'] = np.nan
        df['Range_Low'] = np.nan

    # Verify the presence of Range_High and Range_Low
    if 'Range_High' not in df.columns or 'Range_Low' not in df.columns:
        logger.error("'Range_High' or 'Range_Low' columns are missing after assignment.")
    else:
        logger.info(f"'Range_High' and 'Range_Low' columns successfully added.")

    # Final verification of the index
    logger.info(f"Final DataFrame index: {df.index.min()} to {df.index.max()}")

    return df

# Function to implement the range support/resistance strategy
def backtest_range_support_resistance_strategy(df, risk_reward_ratio=1.5):

    if df.empty:
        logger.warning("Empty DataFrame provided to backtest_range_support_resistance_strategy.")
        return pd.DataFrame()
    
    # Ensure index is datetime with proper unit if necessary
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            # Attempt to parse as seconds first
            df.index = pd.to_datetime(df.index, unit='s', errors='coerce')
            if df.index.isnull().any():
                logger.warning("Datetime conversion with unit='s' resulted in NaT. Trying unit='ms'.")
                df.index = pd.to_datetime(df.index, unit='ms', errors='coerce')
            # Filter out rows where datetime conversion failed
            df = df[df.index.notna()]
            if df.empty:
                logger.error("All datetime conversions resulted in NaT. Please check your timestamp units.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error converting index to datetime: {e}")
            return pd.DataFrame()
    
    trades = []
    position = None  # Current open position
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_time = row.name.time()
        current_date = row.name.date()
        
        # Ensure Range_High and Range_Low are available
        if pd.isna(row['Range_High']) or pd.isna(row['Range_Low']):
            continue  # Skip if range is not defined for the day
        
        # Entry condition: After 14:30 and no open position
        if current_time >= pd.to_datetime('14:30').time() and position is None:
            # Long Entry
            if row['High'] >= row['Range_High']:
                entry_price = row['Range_High']
                stop_loss = row['Range_Low']
                take_profit = entry_price + (entry_price - stop_loss) * risk_reward_ratio
                position = {
                    'Entry_Time': row.name,
                    'Entry_Price': entry_price,
                    'Type': 'Long',
                    'Stop_Loss': stop_loss,
                    'Take_Profit': take_profit
                }
                logger.info(f"Entered Long at {entry_price} on {row.name}")
            
            # Short Entry
            elif row['Low'] <= row['Range_Low']:
                entry_price = row['Range_Low']
                stop_loss = row['Range_High']
                take_profit = entry_price - (stop_loss - entry_price) * risk_reward_ratio
                position = {
                    'Entry_Time': row.name,
                    'Entry_Price': entry_price,
                    'Type': 'Short',
                    'Stop_Loss': stop_loss,
                    'Take_Profit': take_profit
                }
                logger.info(f"Entered Short at {entry_price} on {row.name}")
        
        # Exit conditions
        if position is not None:
            exit_trade = False
            exit_reason = None
            exit_price = None
            
            if position['Type'] == 'Long':
                if row['Low'] <= position['Stop_Loss']:
                    exit_price = position['Stop_Loss']
                    exit_reason = 'Stop Loss'
                    exit_trade = True
                elif row['High'] >= position['Take_Profit']:
                    exit_price = position['Take_Profit']
                    exit_reason = 'Take Profit'
                    exit_trade = True
            elif position['Type'] == 'Short':
                if row['High'] >= position['Stop_Loss']:
                    exit_price = position['Stop_Loss']
                    exit_reason = 'Stop Loss'
                    exit_trade = True
                elif row['Low'] <= position['Take_Profit']:
                    exit_price = position['Take_Profit']
                    exit_reason = 'Take Profit'
                    exit_trade = True
            
            if exit_trade:
                position['Exit_Time'] = row.name
                position['Exit_Price'] = exit_price
                position['Exit_Reason'] = exit_reason
                position['Result'] = 'Win' if (
                    (exit_price > position['Entry_Price'] and position['Type'] == 'Long') or
                    (exit_price < position['Entry_Price'] and position['Type'] == 'Short')
                ) else 'Loss'
                trades.append(position)
                logger.info(f"Exited {position['Type']} at {exit_price} on {row.name} due to {exit_reason}")
                position = None  # Reset position
    
    # Close any open positions at the end of the data
    if position is not None:
        last_row = df.iloc[-1]
        position['Exit_Time'] = last_row.name
        position['Exit_Price'] = last_row['Close']
        position['Result'] = 'Win' if (
            (last_row['Close'] > position['Entry_Price'] and position['Type'] == 'Long') or
            (last_row['Close'] < position['Entry_Price'] and position['Type'] == 'Short')
        ) else 'Loss'
        trades.append(position)
        logger.info(f"Exited {position['Type']} at {last_row['Close']} on {last_row.name} due to End of Data")
    
    trades_df = pd.DataFrame(trades)
    
    return trades_df

# Function to calculate performance metrics (adapted for range support/resistance)
def calculate_range_support_resistance_metrics(df, trades_df):

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
        print(f"Error in calculate_range_support_resistance_metrics: {str(e)}")
        return default_metrics

# Function to plot the trades (adapted for range support/resistance)
def plot_trades_range_support_resistance(df, trades_df, title):
    try:
        # Create the plot
        plt.figure(figsize=(15, 7))
        
        # Plot price data
        plt.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.6)
        
        # Plot Range High and Range Low
        plt.plot(df.index, df['Range_High'], label='Range High', color='red', linestyle='--', alpha=0.5)
        plt.plot(df.index, df['Range_Low'], label='Range Low', color='green', linestyle='--', alpha=0.5)
        
        # Plot trade entry and exit points
        for _, trade in trades_df.iterrows():
            if trade['Type'] == 'Long':
                plt.scatter(trade['Entry_Time'], trade['Entry_Price'], color='green', marker='^', s=100)
                plt.scatter(trade['Exit_Time'], trade['Exit_Price'], color='red', marker='v', s=100)
            else:  # Short trade
                plt.scatter(trade['Entry_Time'], trade['Entry_Price'], color='red', marker='v', s=100)
                plt.scatter(trade['Exit_Time'], trade['Exit_Price'], color='green', marker='^', s=100)
        
        # Add shaded areas for different days
        unique_dates = pd.Series(df.index.date).unique()
        for i, date in enumerate(unique_dates):
            if i % 2 == 0:  # Alternate between shaded and non-shaded
                date_df = df[df.index.date == date]
                if not date_df.empty:
                    plt.axvspan(date_df.index[0], date_df.index[-1], 
                              alpha=0.1, color='gray')
                    logger.info(f"Added axvspan for date: {date}")
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()  # Close the figure to free memory
        
        # Encode the image to base64
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return plot_base64
        
    except Exception as e:
        logger.error(f"Error in plot_trades_range_support_resistance: {e}")
        logger.error(traceback.format_exc())
        return None