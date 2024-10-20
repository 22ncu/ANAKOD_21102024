    # Let's refactor the current code to define a FeatureEngineer class and move the feature engineering logic inside it.

import pandas as pd
import numpy as np
from scipy.stats import linregress

class FeatureEngineer:
    def __init__(self):
        pass
    
    def engineer_features(self, df):
        # Fiyat değişim oranları
        df['price_change'] = df['close'].pct_change()
        df['price_change_1d'] = df['close'].pct_change(periods=1)
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        df['price_change_20d'] = df['close'].pct_change(periods=20)

        # Volatilite özellikleri
        df['volatility_1d'] = df['close'].rolling(window=1).std()
        df['volatility_5d'] = df['close'].rolling(window=5).std()
        df['volatility_20d'] = df['close'].rolling(window=20).std()

        # Trend özellikleri
        df['trend_1d'] = df['close'] - df['close'].shift(1)
        df['trend_5d'] = df['close'] - df['close'].shift(5)
        df['trend_20d'] = df['close'] - df['close'].shift(20)

        # Gösterge çapraz kesişimleri
        df['sma_cross'] = np.where(df['sma_30'] > df['ema_30'], 1, 0)
        df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, 0)

        # RSI aşırı alım/satım bölgeleri
        df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
        df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)

        # Bollinger Bands özellikleri
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mavg']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])

        # Hacim bazlı özellikler
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

        # Fiyat momentumu
        df['momentum'] = df['close'] - df['close'].shift(5)

        # Trend gücü
        def trend_strength(prices, window=14):
            slopes = [linregress(range(window), prices[i:i+window])[0] for i in range(len(prices)-window+1)]
            return pd.Series(slopes, index=prices.index[window-1:])
        
        df['trend_strength'] = trend_strength(df['close'])

        # Fraktal Boyut İndeksi (Fractal Dimension Index)
        def fdi(high, low, close, window=5):
            hc = np.log(high) - np.log(close)
            cl = np.log(close) - np.log(low)
            hlc = np.log(high) - np.log(low)
            
            n1 = (np.log(hc.rolling(window).sum()) + np.log(cl.rolling(window).sum()) 
                  - np.log(hlc.rolling(window).sum())) / np.log(2)
            return (2 - n1) * 100

        df['fdi'] = fdi(df['high'], df['low'], df['close'])

        return df
