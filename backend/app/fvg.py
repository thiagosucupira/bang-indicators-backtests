# fvg.py
import pandas as pd


def is_gap_filled(gap, current_low, current_high):
    if gap['Type'] == 'Bullish':
        return current_low <= gap['FVG_Start']
    else:  # Bearish
        return current_high >= gap['FVG_Start']

def identify_fair_value_gaps(df, current_idx, open_gaps, min_gap_size=0.0001):
    new_gaps = []
    
    if current_idx < 2:
        return new_gaps
    
    prev_high, prev_low = df.iloc[current_idx - 2]['High'], df.iloc[current_idx - 2]['Low']
    current_high, current_low = df.iloc[current_idx]['High'], df.iloc[current_idx]['Low']
    current_time = df.iloc[current_idx]['Date']
    
    if current_low > prev_high:
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
    
    elif prev_low > current_high:
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

def identify_all_fair_value_gaps(df, min_gap_size=0.0001):
    fvg_data = []
    for i in range(2, len(df)):
        prev_high, prev_low = df.iloc[i-2]['High'], df.iloc[i-2]['Low']
        current_high, current_low = df.iloc[i]['High'], df.iloc[i]['Low']
        current_time = df.iloc[i]['Date']
        
        if current_low > prev_high:
            gap_size = current_low - prev_high
            if gap_size > min_gap_size:
                new_gap = {
                    'index': i,
                    'Type': 'Bullish',
                    'FVG_Start': prev_high,
                    'FVG_End': current_low,
                    'Start_Time': current_time,
                    'Used': False
                }
                fvg_data.append(new_gap)
        elif prev_low > current_high:
            gap_size = prev_low - current_high
            if gap_size > min_gap_size:
                new_gap = {
                    'index': i,
                    'Type': 'Bearish',
                    'FVG_Start': prev_low,
                    'FVG_End': current_high,
                    'Start_Time': current_time,
                    'Used': False
                }
                fvg_data.append(new_gap)
    
    return pd.DataFrame(fvg_data)

def identify_current_open_fair_value_gaps(df, min_gap_size=0.0001):
    fvg_data = []
    open_gaps = []
    for i in range(2, len(df)):
        prev_high, prev_low = df.iloc[i-2]['High'], df.iloc[i-2]['Low']
        current_high, current_low = df.iloc[i]['High'], df.iloc[i]['Low']
        current_time = df.iloc[i]['Date']
        
        if current_low > prev_high:
            gap_size = current_low - prev_high
            if gap_size > min_gap_size:
                new_gap = {
                    'index': i,
                    'Type': 'Bullish',
                    'FVG_Start': prev_high,
                    'FVG_End': current_low,
                    'Start_Time': current_time,
                    'Used': False
                }
                fvg_data.append(new_gap)
                open_gaps.append(new_gap)
        elif prev_low > current_high:
            gap_size = prev_low - current_high
            if gap_size > min_gap_size:
                new_gap = {
                    'index': i,
                    'Type': 'Bearish',
                    'FVG_Start': prev_low,
                    'FVG_End': current_high,
                    'Start_Time': current_time,
                    'Used': False
                }
                fvg_data.append(new_gap)
                open_gaps.append(new_gap)
        
        open_gaps = [gap for gap in open_gaps if not is_gap_filled(gap, current_low, current_high)]
    
    return pd.DataFrame(open_gaps)

def backtest_fvg_strategy(df, min_gap_size=0.0001):
    open_gaps = []
    trades = []
    used_fvgs = set()
    closed_trades = []
    total_fvgs_identified = 0
    
    for i in range(len(df)):
        current_price = df.iloc[i]['Close']
        current_time = df.iloc[i]['Date']
        current_low, current_high = df.iloc[i]['Low'], df.iloc[i]['High']
        
        new_gaps = identify_fair_value_gaps(df, i, open_gaps, min_gap_size)
        total_fvgs_identified += len(new_gaps)
        
        open_gaps = [gap for gap in open_gaps if not is_gap_filled(gap, current_low, current_high)]
        
        for gap in open_gaps[:]:
            mid_point = (gap['FVG_Start'] + gap['FVG_End']) / 2
            
            if gap['Used']:
                continue
            
            entry_condition = False
            trade_direction = None
            entry_price = None
            
            if gap['Type'] == 'Bullish':
                if i > 0 and df.iloc[i - 1]['Close'] < mid_point <= current_price:
                    entry_condition = True
                    trade_direction = 'Long'
                    entry_price = mid_point
            elif gap['Type'] == 'Bearish':
                if i > 0 and df.iloc[i - 1]['Close'] > mid_point >= current_price:
                    entry_condition = True
                    trade_direction = 'Short'
                    entry_price = mid_point
            
            if entry_condition:
                gap_size = abs(gap['FVG_Start'] - gap['FVG_End'])
                stop_loss = entry_price - 1.5*gap_size if trade_direction == 'Long' else entry_price + 1.5*gap_size
                take_profit = entry_price + 3 * gap_size if trade_direction == 'Long' else entry_price - 3 * gap_size
                
                trades.append({
                    'Entry_Time': current_time,
                    'Entry_Price': entry_price,
                    'Type': gap['Type'],
                    'Stop_Loss': stop_loss,
                    'Take_Profit': take_profit,
                    'FVG_Index': gap['index'],
                    'Trade_Direction': trade_direction
                })
                used_fvgs.add(gap['index'])
                gap['Used'] = True
                open_gaps.remove(gap)
    
    for trade in trades:
        entry_time = trade['Entry_Time']
        entry_price = trade['Entry_Price']
        trade_direction = trade['Trade_Direction']
        fvg_index = trade['FVG_Index']
        
        entry_indices = df.index[df['Date'] == entry_time].tolist()
        if not entry_indices:
            entry_index = (df['Date'] - entry_time).abs().idxmin()
        else:
            entry_index = entry_indices[0]
        
        exit_found = False
        for i in range(entry_index + 1, len(df)):
            current_price = df.iloc[i]['Close']
            current_time = df.iloc[i]['Date']
            
            if trade_direction == 'Long':
                if current_price >= trade['Take_Profit']:
                    result = 'Win'
                elif current_price <= trade['Stop_Loss']:
                    result = 'Loss'
                else:
                    continue
            elif trade_direction == 'Short':
                if current_price <= trade['Take_Profit']:
                    result = 'Win'
                elif current_price >= trade['Stop_Loss']:
                    result = 'Loss'
                else:
                    continue
            
            closed_trades.append({
                'Entry_Time': trade['Entry_Time'],
                'Exit_Time': current_time,
                'Entry_Price': trade['Entry_Price'],
                'Exit_Price': current_price,
                'Type': trade['Type'],
                'Result': result,
                'FVG_Index': fvg_index,
                'Trade_Direction': trade_direction
            })
            exit_found = True
            break
        
        if not exit_found:
            closed_trades.append({
                'Entry_Time': trade['Entry_Time'],
                'Exit_Time': df.iloc[-1]['Date'],
                'Entry_Price': trade['Entry_Price'],
                'Exit_Price': df.iloc[-1]['Close'],
                'Type': trade['Type'],
                'Result': 'Open',
                'FVG_Index': fvg_index,
                'Trade_Direction': trade_direction
            })
    
    return pd.DataFrame(closed_trades), used_fvgs, total_fvgs_identified

def plot_candlesticks_with_fvg_and_trades(df, open_gaps, title):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import io
    import base64
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    width = 0.6
    width2 = 0.05
    up = df[df.Close >= df.Open]
    down = df[df.Close < df.Open]
    
    ax.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='g', edgecolor='black', linewidth=0.5)
    ax.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color='black', edgecolor='black', linewidth=0.5)
    ax.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color='black', edgecolor='black', linewidth=0.5)
    
    ax.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color='r', edgecolor='black', linewidth=0.5)
    ax.bar(down.index, down.High - down.Open, width2, bottom=down.Open, color='black', edgecolor='black', linewidth=0.5)
    ax.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, color='black', edgecolor='black', linewidth=0.5)
    
    for gap in open_gaps:
        gap_index = gap['index']
        mid_point = (gap['FVG_Start'] + gap['FVG_End']) / 2
        rect = Rectangle((gap_index, gap['FVG_End']), 
                         len(df) - gap_index, 
                         gap['FVG_Start'] - gap['FVG_End'], 
                         facecolor="yellow",
                         alpha=0.2,
                         edgecolor='none')
        ax.add_patch(rect)
        ax.axhline(y=mid_point, xmin=gap_index/len(df), xmax=1, color='red', linestyle='--', linewidth=1, alpha=0.7)
   
    ax.set_xlim(-1, len(df))
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(title)
    
    step = max(len(df)//10, 1)
    date_ticks = df['Date'][::step]
    ax.set_xticks(range(0, len(df), step))
    ax.set_xticklabels([dt.strftime('%Y-%m-%d') for dt in date_ticks], rotation=45)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return image_base64

def calculate_strategy_metrics(df, trades_df, start_date, end_date):
    if trades_df.empty:
        return {
            'Buy and Hold Return (%)': 0.0,
            'Strategy Return (%)': 0.0,
            'Max Drawdown Strategy (%)': 0.0,
            'Max Drawdown Buy and Hold (%)': 0.0,
            'Number of Trades': 0,
            'Win Rate (%)': 0.0,
            'Average Win (%)': 0.0,
            'Average Loss (%)': 0.0,
            'Profit Factor': 0.0
        }

    buy_and_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100

    strategy_returns = trades_df.apply(
        lambda x: (x['Exit_Price'] - x['Entry_Price']) / x['Entry_Price']
        if x['Type'] == 'Bullish' and x['Trade_Direction'] == 'Long'
        else (x['Entry_Price'] - x['Exit_Price']) / x['Entry_Price']
        if x['Type'] == 'Bearish' and x['Trade_Direction'] == 'Short'
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
