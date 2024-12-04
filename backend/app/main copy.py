# main.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
from auth import auth
from auth.models import Base, User
from dependencies import engine, get_pro_user, get_current_user
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
import logging
import traceback

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
    allow_origins=["*"],
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
        
        else:
            raise ValueError("Unsupported indicator selected.")
        
        response = {
            "plot_image": plot_image,
            "indicator": request.indicator
        }
        return response

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
async def run_backtest(request: BacktestRequest):
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Ensure the 'Date' column is of datetime type and set it as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.index.name = 'Date'  # Name the index for clarity

        # Calculate SMA and EMA
        df = calculate_sma_ema(df)
        
        # Run backtest strategy
        trades_df = backtest_sma_ema_strategy(df)
        
        # Log trades for debugging
        logger.info(f"Trades DataFrame columns: {trades_df.columns.tolist()}")
        logger.info(f"Trades DataFrame data types:\n{trades_df.dtypes}")
        logger.info(trades_df.head().to_string())
        
        # Calculate metrics
        metrics = calculate_sma_ema_metrics(df, trades_df)
        
        # Plot trades
        plot_base64 = plot_trades_sma_ema(df, trades_df, "SMA and EMA Crossover Strategy")
        
        # Prepare response
        response = {
            'metrics': metrics,
            'trades': trades_df.to_dict(orient='records'),
            'plot': plot_base64
        }
        
        return response

    except Exception as e:
        logger.error(f"Error in run_backtest: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during backtesting.")

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

