# main.py

from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
import logging
import traceback
import numpy as np

from auth import auth
from auth.models import Base, User
from dependencies import engine, get_current_user
from data import fetch_yfinance_data
from fvg import (
    identify_current_open_fair_value_gaps,
    backtest_fvg_strategy,
    calculate_strategy_metrics,
    plot_candlesticks_with_fvg_and_trades
)
from williamsR import (
    calculate_williams_r,
    backtest_williams_r_strategy,
    plot_trades_williams_r,
    calculate_williams_r_metrics
)
from crossover import (
    calculate_sma_ema,
    backtest_sma_ema_strategy,
    calculate_sma_ema_metrics,
    plot_trades_sma_ema
)
from breakout import (
    calculate_daily_range,
    backtest_range_support_resistance_strategy,
    calculate_range_support_resistance_metrics,
    plot_trades_range_support_resistance
)
from markov import run_markov_strategy  # Add this import near the other strategy imports

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI(title="BANG Indicators and Backtests API")

# Create database tables
Base.metadata.create_all(bind=engine)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication router
app.include_router(auth.router)

class PlotRequest(BaseModel):
    indicator: str
    symbol: str
    interval: str
    start_date: str
    end_date: str

@app.post("/generate_plot")
def generate_plot(request: PlotRequest, user: User = Depends(get_current_user)):
    try:
        if request.indicator == "FairValueGap":
            df = fetch_yfinance_data(request.symbol, request.interval, request.start_date, request.end_date)
            open_fvg_df = identify_current_open_fair_value_gaps(df)
            open_gaps = open_fvg_df.to_dict(orient='records')
            plot_image = plot_candlesticks_with_fvg_and_trades(df, open_gaps, "Open Fair Value Gaps")
        
        elif request.indicator == "Williams_R":
            df = calculate_williams_r(fetch_yfinance_data(request.symbol, request.interval, request.start_date, request.end_date))
            plot_image = plot_trades_williams_r(df, "Williams %R Indicator")
        
        elif request.indicator == "Crossover":
            df = calculate_sma_ema(fetch_yfinance_data(request.symbol, request.interval, request.start_date, request.end_date))
            trades_df = backtest_sma_ema_strategy(df)
            plot_image = plot_trades_sma_ema(df, trades_df, "SMA and EMA Crossover Strategy")
            
        elif request.indicator == "Breakout":
            # Fetch data
            df = fetch_yfinance_data(request.symbol, request.interval, request.start_date, request.end_date)
            
            # Log the initial state
            logger.info(f"DataFrame before calculate_daily_range:\n{df.head()}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            
            # Calculate daily range
            df = calculate_daily_range(df)
            
            # Log the state after calculate_daily_range
            logger.info(f"DataFrame after calculate_daily_range:\n{df.head()}")
            
            # Only proceed with backtesting if we have valid ranges
            if df['Range_High'].notna().any():
                trades_df = backtest_range_support_resistance_strategy(df)
                logger.info(f"Trades DataFrame after backtest:\n{trades_df.head()}")
                
                if not trades_df.empty:
                    # Generate the plot
                    plot_image = plot_trades_range_support_resistance(df, trades_df, "Range Support/Resistance Strategy")
                    logger.info(f"Plot image generated: {plot_image is not None}")
                    if plot_image:
                        logger.info(f"Plot image length: {len(plot_image)}")
                        logger.info(f"Plot image starts with: {plot_image[:50]}")
                    
                    # Calculate metrics
                    metrics = calculate_strategy_metrics(
                        df=df,
                        trades_df=trades_df,
                        start_date=request.start_date,
                        end_date=request.end_date
                    )
                    
                    # Convert metrics and handle infinite values
                    serializable_metrics = {}
                    for key, value in metrics.items():
                        if hasattr(value, 'item'):
                            value = value.item()
                        if isinstance(value, (int, float)):
                            if np.isinf(value) or np.isnan(value):
                                value = None
                        serializable_metrics[key] = value
                    
                    # Convert trades data
                    trades_list = []
                    for trade in trades_df.to_dict(orient='records'):
                        serializable_trade = {}
                        for key, value in trade.items():
                            if hasattr(value, 'item'):
                                value = value.item()
                            elif isinstance(value, pd.Timestamp):
                                value = value.isoformat()
                            if isinstance(value, (int, float)):
                                if np.isinf(value) or np.isnan(value):
                                    value = None
                            serializable_trade[key] = value
                        trades_list.append(serializable_trade)
                    
                    response = {
                        "plot": plot_image,
                        "metrics": serializable_metrics,
                        "closed_trades": trades_list
                    }
                    logger.info("Response contains plot: %s", "plot" in response)
                    return response
                else:
                    return {"message": "No trades were generated during the backtest."}
            else:
                return {"message": "No valid ranges found in the specified time period."}
        elif request.indicator == "Markov":  # New branch for Markov strategy
            df = fetch_yfinance_data(request.symbol, request.interval, request.start_date, request.end_date)
            result = run_markov_strategy(df)
            plot_image = result.get("plotImage")
        else:
            raise ValueError("Unsupported indicator selected.")
        
        response_content = {
            "plot_image": plot_image,
            "indicator": request.indicator
        }
        return JSONResponse(content=response_content, media_type="application/json; charset=utf-8")
    
    except Exception as e:
        logger.error("An error occurred during plot generation:")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")

class BacktestRequest(BaseModel):
    indicator: str
    symbol: str
    interval: str
    start_date: str
    end_date: str

@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest, user: User = Depends(get_current_user)):
    try:
        if request.indicator == "FairValueGap":
            df = fetch_yfinance_data(request.symbol, request.interval, request.start_date, request.end_date)
            trades_df, used_fvgs, total_fvgs = backtest_fvg_strategy(df)
            metrics = calculate_strategy_metrics(df, trades_df, request.start_date, request.end_date)
        
        elif request.indicator == "Williams_R":
            df = fetch_yfinance_data(request.symbol, request.interval, request.start_date, request.end_date)
            df = calculate_williams_r(df)
            trades_df = backtest_williams_r_strategy(df)
            metrics = calculate_williams_r_metrics(df, trades_df, request.start_date, request.end_date)
        
        elif request.indicator == "Crossover":
            df = fetch_yfinance_data(request.symbol, request.interval, request.start_date, request.end_date)
            df = calculate_sma_ema(df)
            trades_df = backtest_sma_ema_strategy(df)
            metrics = calculate_sma_ema_metrics(df, trades_df)
        
        elif request.indicator == "Breakout":
            df = fetch_yfinance_data(request.symbol, request.interval, request.start_date, request.end_date)
            df = calculate_daily_range(df)
            trades_df = backtest_range_support_resistance_strategy(df)
            metrics = calculate_range_support_resistance_metrics(df, trades_df)
        
        elif request.indicator == "Markov":  # New branch for Markov strategy
            df = fetch_yfinance_data(request.symbol, request.interval, request.start_date, request.end_date)
            result = run_markov_strategy(df)
            metrics = result.get("metrics")
            trades = result.get("trades")
            if trades:
                trades_df = pd.DataFrame(trades)
            else:
                trades_df = pd.DataFrame()
        
        else:
            raise ValueError("Unsupported indicator selected.")

        if trades_df.empty:
            return {
                "metrics": metrics,  # Return default metrics
                "closed_trades": [],
                "message": "No trades were taken during the selected period. Try adjusting the date range or strategy parameters."
            }
        
        response = {
            "metrics": metrics,
            "closed_trades": trades_df.to_dict(orient='records')
        }
        return response

    except Exception as e:
        logger.error("An error occurred during backtesting:")
        logger.error(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tickers")
def get_tickers():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'tickers.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError("Tickers CSV file not found.")

        df = pd.read_csv(csv_path)
        tickers = df.to_dict(orient='records')
        
        return {"tickers": tickers}
    
    except Exception as e:
        logger.error("An error occurred while fetching tickers:")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")
