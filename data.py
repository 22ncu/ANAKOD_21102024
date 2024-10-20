import os
import logging
import time
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

def initialize_binance():
    """
    Binance API istemcisini başlatır.
    :return: Binance Client objesi
    """
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    return Client(api_key, api_secret)

def fetch_data(client, symbol='BTCUSDT', interval='1m', days=1):
    """
    Binance'ten geçmiş veri çeker.
    :param client: Binance Client objesi
    :param symbol: İşlem çifti (ör. 'BTCUSDT')
    :param interval: Zaman dilimi (ör. '1m', '1h')
    :param days: Kaç günlük veri çekileceği
    :return: Pandas DataFrame içinde veri
    """
    logging.info(f"Veri çekme başlatılıyor: {symbol}, Zaman Dilimi: {interval}, Gün: {days}")
    
    since = int(time.time() * 1000) - days * 24 * 60 * 60 * 1000
    klines = []
    
    while since < int(time.time() * 1000):
        try:
            # Binance'ten OHLCV verisi çeker
            new_klines = client.get_klines(symbol=symbol, interval=interval, startTime=since, limit=1000)
            if not new_klines:
                logging.warning("Veri bulunamadı. İşlem sonlandırılıyor.")
                break
            
            klines += new_klines
            since = new_klines[-1][0] + 1  # Sonraki veri başlangıç zamanı
            time.sleep(0.5)  # API limitleri nedeniyle kısa bir gecikme
            
        except BinanceAPIException as e:
            logging.error(f"Veri çekme sırasında Binance API hatası: {e}")
            break
        except Exception as e:
            logging.error(f"Veri çekme sırasında hata: {e}")
            break

    if not klines:
        logging.error("Veri çekilemedi.")
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Sütunları sayısal formata çevirme
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    # NaN değerleri kontrol etme ve temizleme
    if df.isnull().values.any():
        logging.warning("Veri setinde NaN değerler mevcut. Bu değerler işleme alınmadan geçilecek.")
        df.dropna(inplace=True)  # NaN içeren satırları kaldır

    logging.info(f"Veri başarıyla çekildi. Toplam veri noktası: {len(df)}")
    return df
