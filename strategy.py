import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

class TradingStrategy:
    def __init__(self, model, scaler, lookback=60, threshold=0.5):
        self.model = model
        self.scaler = scaler
        self.lookback = lookback
        self.threshold = threshold
        logging.info(f"TradingStrategy initialized with lookback={lookback} and threshold={threshold}")

    def generate_signals(self, df):
        logging.info("Generating trading signals")
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0  # 0: Hold, 1: Buy, -1: Sell

        # Scale the features
        scaled_features = self.scaler.transform(df)

        # Prepare data for prediction
        X = []
        for i in range(self.lookback, len(scaled_features)):
            X.append(scaled_features[i-self.lookback:i])
        X = np.array(X)

        # Make predictions
        predictions = self.model.predict(X)

        # Generate signals based on predictions
        signals['prediction'] = [0] * self.lookback + list(predictions.flatten())
        signals['signal'] = np.where(signals['prediction'] > self.threshold, 1, 0)
        signals['signal'] = np.where(signals['prediction'] < -self.threshold, -1, signals['signal'])

        # Add additional signal generation logic
        signals = self.add_trend_following_signals(df, signals)
        signals = self.add_mean_reversion_signals(df, signals)
        signals = self.add_volatility_based_signals(df, signals)

        logging.info(f"Generated {len(signals[signals['signal'] != 0])} trading signals")
        return signals

    def add_trend_following_signals(self, df, signals):
        # Simple Moving Average crossover
        signals['SMA_short'] = df['close'].rolling(window=20).mean()
        signals['SMA_long'] = df['close'].rolling(window=50).mean()
        signals['trend_signal'] = np.where(signals['SMA_short'] > signals['SMA_long'], 1, 0)
        signals['trend_signal'] = np.where(signals['SMA_short'] < signals['SMA_long'], -1, signals['trend_signal'])
        
        # Combine with existing signals
        signals['signal'] = np.where((signals['signal'] == 1) & (signals['trend_signal'] == 1), 1, signals['signal'])
        signals['signal'] = np.where((signals['signal'] == -1) & (signals['trend_signal'] == -1), -1, signals['signal'])
        
        return signals

    def add_mean_reversion_signals(self, df, signals):
        # Bollinger Bands
        signals['BB_upper'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
        signals['BB_lower'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()
        
        signals['mean_reversion_signal'] = np.where(df['close'] > signals['BB_upper'], -1, 0)
        signals['mean_reversion_signal'] = np.where(df['close'] < signals['BB_lower'], 1, signals['mean_reversion_signal'])
        
        # Combine with existing signals
        signals['signal'] = np.where((signals['signal'] == 0) & (signals['mean_reversion_signal'] != 0), signals['mean_reversion_signal'], signals['signal'])
        
        return signals

    def add_volatility_based_signals(self, df, signals):
        # Average True Range (ATR)
        signals['ATR'] = df['atr']  # Assuming ATR is already calculated in the input dataframe
        signals['ATR_threshold'] = signals['ATR'].rolling(window=14).mean() * 1.5
        
        # Adjust signal strength based on volatility
        signals['volatility_factor'] = np.minimum(signals['ATR'] / signals['ATR_threshold'], 1)
        signals['signal'] = signals['signal'] * signals['volatility_factor']
        
        return signals

    def update_parameters(self, params):
        if 'lookback' in params:
            self.lookback = params['lookback']
        if 'threshold' in params:
            self.threshold = params['threshold']
        logging.info(f"Updated strategy parameters: lookback={self.lookback}, threshold={self.threshold}")
