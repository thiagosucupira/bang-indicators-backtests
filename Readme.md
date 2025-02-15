# Trading Strategies & Backtesting Suite

## Overview
This project is designed to backtest and evaluate multiple algorithmic trading strategies. The system includes a diverse set of strategies, integrated performance metrics, risk management tools, and visualization features to provide thorough insights into the potential advantages and pitfalls of each strategy.

## Implemented Trading Strategies

### 1. Fair Value Gap (FVG) Strategy
The Fair Value Gap (FVG) strategy identifies price gaps that occur in rapidly moving markets. It capitalizes on the notion that when the price returns to the gap, a potential reversal or continuation may occur.
- **Entry**: Long for bullish gaps, short for bearish gaps.
- **Risk Management**: Stop Loss is set at the gap edge; Take Profit is at twice the stop loss distance (2:1 Reward-to-Risk).

### 2. Williams %R Strategy
The Williams %R strategy uses the Williams %R oscillator to detect overbought/oversold conditions and potential market reversals.
- **Entry**: Buy when Williams %R crosses above -80; Sell when it crosses below -20.
- **Risk Management**: Stop Loss based on a one-day price range from the entry price; Take Profit maintains a 2:1 reward-to-risk ratio.

### 3. ATR-Based Strategy
This strategy employs the Average True Range (ATR) indicator to capture market volatility.
- **Calculation**: Dynamically sets stop-loss and profit target levels using ATR.
- **Metrics**: Computes total return, Sharpe Ratio, and maximum drawdown to evaluate performance.

### 4. SMA/EMA Crossover Strategy
A dual moving average system that uses Simple Moving Average (SMA) and Exponential Moving Average (EMA) crossovers to generate trade signals.
- **Entry**: Triggered when a fast moving average crosses a slow moving average.
- **Analysis**: Backtested with cumulative return and drawdown calculations.

### 5. Range Support/Resistance Strategy
This strategy identifies key support and resistance zones to determine potential breakout or reversal opportunities.
- **Logic**: Trades based on the detection of supply and demand imbalances within price ranges.
- **Risk Management**: Incorporates dynamic stop-loss and take-profit levels based on range volatility.

### 6. Markov Strategy
The Markov Strategy applies probabilistic models to analyze transitions in price behavior, offering a unique approach to trading.
- **Variants**: Includes both standard and detailed implementations.
- **Integration**: Accessible via API endpoints and a dedicated front-end interface for strategy execution and visualization.

## System Features

- **Backtesting Engine**: Execute historical analyses across multiple strategies with performance metrics including:
  - Total Return
  - Sharpe Ratio
  - Maximum Drawdown
  - Win Rate & Profit Factor

- **Risk Management**: Integrated stop-loss and take-profit mechanisms consistently apply a 2:1 reward-to-risk ratio across strategies.

- **Visualization Tools**: Generate charts and plots to visualize:
  - Price action and trade signals
  - Equity curves and performance metrics

- **API Endpoints & Front-End Integration**:
  - RESTful APIs (e.g., /api/markov-strategy) allow programmatic access to strategy backtests.
  - An interactive front-end module enables real-time strategy execution and result visualization.

- **Modular Design**: Separates back-end strategy implementations from front-end interfaces, allowing easy extension and customization.

- **Deployment**:
  - Docker Compose support for containerized execution.
  - Environment configuration via .env for flexible setup.

## Installation & Setup

1. Clone the repository.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure your environment variables in the .env file.
4. Launch the application using Docker Compose:
   ```
   docker-compose up
   ```
5. Run the back-end server and front-end client as specified in their respective directories.

## Usage

- Backtest strategies via command-line interfaces or API calls.
- Use the interactive front-end to execute strategies and explore visual data representations.
- Customize strategy parameters within the codebase for tailored analyses.

## Disclaimer
This project is for research and educational purposes only. Backtesting results do not guarantee future performance. Trade responsibly and at your own risk.

## Contributing & License
Contributions are welcome. Please see CONTRIBUTING.md for guidelines. This project is open-sourced under the MIT License.