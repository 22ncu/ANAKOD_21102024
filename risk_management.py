import numpy as np
import pandas as pd
import logging

class RiskManager:
    def __init__(self, initial_balance, max_position_size=0.02, stop_loss_pct=0.02, take_profit_pct=0.03):
        self.balance = initial_balance
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.positions = {}
        logging.info(f"RiskManager initialized with balance={initial_balance}, max_position_size={max_position_size}, stop_loss_pct={stop_loss_pct}, take_profit_pct={take_profit_pct}")

    def calculate_position_size(self, current_price):
        max_position_value = self.balance * self.max_position_size
        position_size = max_position_value / current_price
        return np.floor(position_size)

    def check_stop_loss(self, symbol, current_price):
        if symbol in self.positions:
            entry_price = self.positions[symbol]['entry_price']
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                logging.info(f"Stop loss triggered for {symbol} at {current_price}")
                return True
        return False

    def check_take_profit(self, symbol, current_price):
        if symbol in self.positions:
            entry_price = self.positions[symbol]['entry_price']
            if current_price >= entry_price * (1 + self.take_profit_pct):
                logging.info(f"Take profit triggered for {symbol} at {current_price}")
                return True
        return False

    def update_balance(self, pnl):
        self.balance += pnl
        logging.info(f"Balance updated: {self.balance}")

    def add_position(self, symbol, entry_price, size):
        self.positions[symbol] = {
            'entry_price': entry_price,
            'size': size
        }
        logging.info(f"Added position: {symbol}, entry_price={entry_price}, size={size}")

    def remove_position(self, symbol):
        if symbol in self.positions:
            del self.positions[symbol]
            logging.info(f"Removed position: {symbol}")

class PortfolioManager:
    def __init__(self, symbols, initial_weights=None):
        self.symbols = symbols
        if initial_weights is None:
            self.weights = {symbol: 1/len(symbols) for symbol in symbols}
        else:
            self.weights = initial_weights
        logging.info(f"PortfolioManager initialized with symbols={symbols}")

    def rebalance(self, current_prices, risk_scores):
        total_risk_score = sum(risk_scores.values())
        new_weights = {symbol: risk_scores[symbol] / total_risk_score for symbol in self.symbols}
        
        # Calculate the difference between current and target weights
        current_total_value = sum(current_prices[symbol] * self.weights[symbol] for symbol in self.symbols)
        current_weights = {symbol: current_prices[symbol] * self.weights[symbol] / current_total_value for symbol in self.symbols}
        weight_diff = {symbol: new_weights[symbol] - current_weights[symbol] for symbol in self.symbols}
        
        # Sort symbols by absolute weight difference
        rebalance_order = sorted(self.symbols, key=lambda x: abs(weight_diff[x]), reverse=True)
        
        # Rebalance
        for symbol in rebalance_order:
            if weight_diff[symbol] > 0:
                logging.info(f"Buying {symbol} to increase weight by {weight_diff[symbol]:.4f}")
            elif weight_diff[symbol] < 0:
                logging.info(f"Selling {symbol} to decrease weight by {abs(weight_diff[symbol]):.4f}")
        
        self.weights = new_weights
        logging.info(f"Portfolio rebalanced. New weights: {self.weights}")

    def calculate_risk_scores(self, df):
        risk_scores = {}
        for symbol in self.symbols:
            # Calculate volatility
            returns = df[symbol]['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate drawdown
            cum_returns = (1 + returns).cumprod()
            drawdown = (cum_returns.cummax() - cum_returns) / cum_returns.cummax()
            max_drawdown = drawdown.max()
            
            # Calculate risk score (lower is better)
            risk_scores[symbol] = volatility * max_drawdown
        
        return risk_scores

    def get_portfolio_value(self, current_prices):
        return sum(current_prices[symbol] * self.weights[symbol] for symbol in self.symbols)

    def get_portfolio_returns(self, price_data):
        portfolio_values = []
        for date in price_data.index:
            prices = {symbol: price_data.loc[date, (symbol, 'close')] for symbol in self.symbols}
            portfolio_values.append(self.get_portfolio_value(prices))
        
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        return portfolio_returns

    def calculate_portfolio_metrics(self, portfolio_returns):
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility
        
        drawdown = (1 + portfolio_returns).cumprod() / (1 + portfolio_returns).cumprod().cummax() - 1
        max_drawdown = drawdown.min()
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
        logging.info(f"Portfolio metrics calculated: {metrics}")
        return metrics
