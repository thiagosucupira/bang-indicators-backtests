import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

def plot_average_prices():
    # Read the tickers from CSV
    tickers_df = pd.read_csv('backend/app/tickers.csv')
    
    # Get today's date and date from 30 days ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Create a figure with larger size
    plt.figure(figsize=(15, 8))
    
    # Get data for first 5 tickers (to avoid overcrowding)
    for i in range(5):
        ticker = tickers_df['Symbol'].iloc[i]
        name = tickers_df['Name'].iloc[i]
        
        try:
            # Fetch data from Yahoo Finance
            data = yf.download(ticker, start=start_date, end=end_date)
            
            # Plot the closing prices
            plt.plot(data.index, data['Close'], label=name, linewidth=2)
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
    
    # Customize the plot
    plt.title('30-Day Price History for Major Currency Pairs', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('price_history.png')
    plt.close()

if __name__ == "__main__":
    plot_average_prices() 