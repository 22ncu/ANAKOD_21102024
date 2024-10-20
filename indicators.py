import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator, IchimokuIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
import logging

class Indicators:
    def __init__(self):
        logging.info("Indicators sınıfı başlatıldı.")

    def calculate_indicators(self, df):
        logging.info("Teknik göstergelerin hesaplanması başlatılıyor.")
        try:
            # Sütunları sayısal formata çevirme (high, low, close, volume sütunları)
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

            # Hatalı veri olup olmadığını kontrol etme
            if df[['high', 'low', 'close', 'volume']].isnull().any().any():
                logging.warning("Veri setinde hatalı veya eksik değerler bulundu. Bu değerler işleme alınmadan geçilecek.")
                df.dropna(subset=['high', 'low', 'close', 'volume'], inplace=True)  # NaN içeren satırları kaldır

            # Bollinger Bands hesaplaması
            indicator_bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_mavg'] = indicator_bb.bollinger_mavg()
            df['bb_high'] = indicator_bb.bollinger_hband()
            df['bb_low'] = indicator_bb.bollinger_lband()

            # MACD hesaplaması
            indicator_macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = indicator_macd.macd()
            df['macd_signal'] = indicator_macd.macd_signal()
            df['macd_diff'] = indicator_macd.macd_diff()

            # RSI hesaplaması
            indicator_rsi = RSIIndicator(close=df['close'], window=14)
            df['rsi'] = indicator_rsi.rsi()

            # ADX hesaplaması
            indicator_adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['adx'] = indicator_adx.adx()

            # ATR hesaplaması
            indicator_atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['atr'] = indicator_atr.average_true_range()

            # Stochastic Oscillator hesaplaması
            indicator_stochastic = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
            df['stoch_k'] = indicator_stochastic.stoch()
            df['stoch_d'] = indicator_stochastic.stoch_signal()

            # ROC (Rate of Change) hesaplaması
            indicator_roc = ROCIndicator(close=df['close'], window=10)
            df['roc'] = indicator_roc.roc()

            # SMA (Basit Hareketli Ortalama) hesaplaması
            indicator_sma = SMAIndicator(close=df['close'], window=30)
            df['sma_30'] = indicator_sma.sma_indicator()

            # EMA (Üssel Hareketli Ortalama) hesaplaması
            indicator_ema = EMAIndicator(close=df['close'], window=30)
            df['ema_30'] = indicator_ema.ema_indicator()

            # Ichimoku Göstergesi hesaplaması
            indicator_ichimoku = IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
            df['ichimoku_a'] = indicator_ichimoku.ichimoku_a()
            df['ichimoku_b'] = indicator_ichimoku.ichimoku_b()

            # On-Balance Volume (OBV) hesaplaması
            indicator_obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
            df['obv'] = indicator_obv.on_balance_volume()

            # Chaikin Money Flow (CMF) hesaplaması
            indicator_cmf = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=20)
            df['cmf'] = indicator_cmf.chaikin_money_flow()

            # Basit momentum hesaplaması
            df['momentum'] = df['close'].pct_change(periods=10)

            logging.info("Teknik göstergeler başarıyla hesaplandı.")
        except Exception as e:
            logging.error(f"Göstergelerin hesaplanması sırasında hata: {e}")
            raise

        return df

    def detect_candlestick_patterns(self, df):
        logging.info("Mum formasyonlarının tespiti başlatılıyor.")
        try:
            # Mum formasyonlarının hesaplanması
            df['body'] = df['close'] - df['open']
            df['range'] = df['high'] - df['low']
            df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
            
            # Hammer formasyonu tespiti
            df['hammer'] = np.where((df['lower_shadow'] > 2 * df['body'].abs()) & 
                                    (df['upper_shadow'] < 0.1 * df['body'].abs()), 1, 0)
            
            # Doji formasyonu tespiti
            df['doji'] = np.where(df['body'].abs() <= 0.1 * df['range'], 1, 0)
            
            logging.info("Mum formasyonları başarıyla tespit edildi.")
        except Exception as e:
            logging.error(f"Mum formasyonlarının tespiti sırasında hata: {e}")
            raise
        return df

    def calculate_fibonacci_levels(self, df, period=50):
        logging.info("Fibonacci seviyelerinin hesaplanması başlatılıyor.")
        try:
            recent_high = df['high'].rolling(window=period).max()
            recent_low = df['low'].rolling(window=period).min()
            
            # Fibonacci seviyelerinin hesaplanması
            df['fib_0'] = recent_high
            df['fib_23.6'] = recent_high - 0.236 * (recent_high - recent_low)
            df['fib_38.2'] = recent_high - 0.382 * (recent_high - recent_low)
            df['fib_50'] = recent_high - 0.5 * (recent_high - recent_low)
            df['fib_61.8'] = recent_high - 0.618 * (recent_high - recent_low)
            df['fib_76.4'] = recent_high - 0.764 * (recent_high - recent_low)
            df['fib_100'] = recent_low
            
            logging.info("Fibonacci seviyeleri başarıyla hesaplandı.")
        except Exception as e:
            logging.error(f"Fibonacci seviyelerinin hesaplanması sırasında hata: {e}")
            raise
        return df
