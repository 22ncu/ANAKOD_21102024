import pandas as pd
import numpy as np
from ta.momentum import KAMAIndicator, TSIIndicator
from ta.trend import CCIIndicator, DPOIndicator, VortexIndicator
from ta.volatility import UlcerIndex
from ta.volume import AccDistIndexIndicator, EaseOfMovementIndicator, ForceIndexIndicator
import logging

class AdvancedIndicators:
    def __init__(self):
        logging.info("AdvancedIndicators sınıfı başlatıldı.")

    def calculate_advanced_indicators(self, df):
        logging.info("Gelişmiş indikatörlerin hesaplanması başlatılıyor.")
        try:
            # KAMA (Kaufman's Adaptive Moving Average)
            kama = KAMAIndicator(close=df['close'])
            df['kama'] = kama.kama()

            # TSI (True Strength Index)
            tsi = TSIIndicator(close=df['close'])
            df['tsi'] = tsi.tsi()

            # CCI (Commodity Channel Index)
            cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'])
            df['cci'] = cci.cci()

            # DPO (Detrended Price Oscillator)
            dpo = DPOIndicator(close=df['close'])
            df['dpo'] = dpo.dpo()

            # Vortex Indicator
            vortex = VortexIndicator(high=df['high'], low=df['low'], close=df['close'])
            df['vortex_pos'] = vortex.vortex_indicator_pos()
            df['vortex_neg'] = vortex.vortex_indicator_neg()

            # Ulcer Index
            ulcer = UlcerIndex(close=df['close'])
            df['ulcer_index'] = ulcer.ulcer_index()

            # Accumulation/Distribution Index
            ad = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
            df['ad_index'] = ad.acc_dist_index()

            # Ease of Movement
            eom = EaseOfMovementIndicator(high=df['high'], low=df['low'], volume=df['volume'])
            df['eom'] = eom.ease_of_movement()

            # Force Index
            fi = ForceIndexIndicator(close=df['close'], volume=df['volume'])
            df['force_index'] = fi.force_index()

            logging.info("Gelişmiş indikatörler başarıyla hesaplandı.")
        except Exception as e:
            logging.error(f"Gelişmiş indikatörlerin hesaplanması sırasında hata: {e}")
            raise

        return df

    def calculate_market_regime(self, df, window=20):
        logging.info("Piyasa rejimi hesaplanıyor.")
        try:
            # Trend gücü
            df['trend_strength'] = df['close'].pct_change(window).abs()

            # Volatilite
            df['volatility'] = df['close'].rolling(window=window).std()

            # Momentum
            df['momentum'] = df['close'].pct_change(window)

            # Piyasa rejimi belirleme
            df['market_regime'] = np.where(df['trend_strength'] > df['trend_strength'].median(),
                                           np.where(df['momentum'] > 0, 'Bullish Trend', 'Bearish Trend'),
                                           np.where(df['volatility'] > df['volatility'].median(), 'High Volatility', 'Range Bound'))

            logging.info("Piyasa rejimi başarıyla hesaplandı.")
        except Exception as e:
            logging.error(f"Piyasa rejimi hesaplanması sırasında hata: {e}")
            raise

        return df

    def calculate_support_resistance(self, df, window=20):
        logging.info("Destek ve direnç seviyeleri hesaplanıyor.")
        try:
            df['support'] = df['low'].rolling(window=window).min()
            df['resistance'] = df['high'].rolling(window=window).max()

            logging.info("Destek ve direnç seviyeleri başarıyla hesaplandı.")
        except Exception as e:
            logging.error(f"Destek ve direnç seviyelerinin hesaplanması sırasında hata: {e}")
            raise

        return df
