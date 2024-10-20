import logging
import numpy as np
import pandas as pd

class Backtester:
    def __init__(self, strategy, initial_balance=10000, commission=0.001, verbose=False):
        """
        Initializes the Backtester with the given strategy, starting balance, and commission fee.
        :param strategy: A TradingStrategy object that defines the buy/sell logic.
        :param initial_balance: The starting balance of the portfolio.
        :param commission: The fee applied to each trade as a decimal (e.g., 0.001 for 0.1%).
        :param verbose: If True, print detailed logs.
        """
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.verbose = verbose
        self.position = 0
        self.portfolio_value = 0
        self.trade_log = []
        self.balance_over_time = []
        self.current_date = None

    def log(self, message):
        """Handles logging of messages."""
        if self.verbose:
            logging.info(message)

    def simulate(self, historical_data):
        """
        Simulates the strategy over the historical data.
        :param historical_data: A pandas DataFrame containing the historical price data.
        :return: A pandas DataFrame with portfolio value and balance over time.
        """
        for index, row in historical_data.iterrows():
            self.current_date = row.name
            signal = self.strategy.generate_signal(row)

            self.log(f"Date: {self.current_date}, Signal: {signal}")

            # Buy signal
            if signal == "BUY" and self.balance > 0:
                self.buy(row['close'])
            # Sell signal
            elif signal == "SELL" and self.position > 0:
                self.sell(row['close'])

            # Record balance and portfolio value
            self.balance_over_time.append({
                'date': self.current_date,
                'balance': self.balance,
                'portfolio_value': self.portfolio_value()
            })

        # Return the final portfolio and balance history as DataFrame
        return pd.DataFrame(self.balance_over_time).set_index('date')

    def buy(self, price):
        """Executes a buy order."""
        # Buy as much as we can with the available balance
        shares_to_buy = self.balance / price
        cost = shares_to_buy * price * (1 + self.commission)

        self.position += shares_to_buy
        self.balance -= cost
        self.log(f"Bought {shares_to_buy} shares at {price}, new balance: {self.balance}")

    def sell(self, price):
        """Executes a sell order."""
        # Sell all holdings
        proceeds = self.position * price * (1 - self.commission)
        self.balance += proceeds
        self.log(f"Sold {self.position} shares at {price}, new balance: {self.balance}")
        self.position = 0

    def portfolio_value(self):
        """Calculates the current value of the portfolio (cash + value of holdings)."""
        return self.balance + (self.position * self.strategy.current_price if self.position > 0 else 0)

    def summary(self):
        """Prints a summary of the backtest performance."""
        total_return = (self.balance + self.portfolio_value()) / self.initial_balance - 1
        self.log(f"Initial balance: {self.initial_balance}")
        self.log(f"Final balance: {self.balance}")
        self.log(f"Total return: {total_return * 100:.2f}%")
        return total_return

    def get_trade_log(self):
        """Returns the log of all trades executed during the backtest."""
        return pd.DataFrame(self.trade_log)
