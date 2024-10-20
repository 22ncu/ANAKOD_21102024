import pandas as pd
import logging
from indicators import Indicators
from feature_engineering import FeatureEngineer
from model_selection import ModelSelector
from risk_management import RiskManager
from strategy import TradingStrategy
from advanced_indicators import AdvancedIndicators
from backtester import Backtester

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Ana program başlatılıyor.")

    # Binance API'den veri çekmek için gerekli fonksiyonları içe aktar
    from data import initialize_binance, fetch_data

    # Binance API istemcisini başlat
    client = initialize_binance()

    # Binance'ten veri çekme
    df = fetch_data(client, symbol='BTCUSDT', interval='1m', days=30)

    # Verileri sayısal formata çevir
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # İndikatörlerin hesaplanması
    indicators = Indicators()
    df = indicators.calculate_indicators(df)  # Teknik göstergelerin hesaplanması
    df = indicators.detect_candlestick_patterns(df)  # Mum formasyonlarının tespiti
    df = indicators.calculate_fibonacci_levels(df)  # Fibonacci seviyelerinin hesaplanması
    logging.info("Tüm indikatörler hesaplandı.")

    # Gelişmiş indikatörlerin hesaplanması
    advanced_ind = AdvancedIndicators()
    df = advanced_ind.calculate_advanced_indicators(df)  # Gelişmiş indikatörlerin hesaplanması
    df = advanced_ind.calculate_market_regime(df)  # Piyasa rejiminin hesaplanması
    df = advanced_ind.calculate_support_resistance(df)  # Destek ve direnç seviyelerinin hesaplanması
    logging.info("Gelişmiş indikatörler hesaplandı.")

    # Özellik mühendisliği
    fe = FeatureEngineer()
    df = fe.engineer_features(df)  # Özellik mühendisliği işlemleri
    logging.info("Özellik mühendisliği tamamlandı.")

    # Model seçimi ve eğitimi
    y = df['close']  # Hedef değişken (tahmin edilmek istenen)
    X = df.drop(columns=['close'])  # Özellikler (hedef değişkeni çıkar)

    ms = ModelSelector(X, y)  # ModelSelector sınıfını başlat
    best_model = ms.select_best_model()  # En iyi modeli seç
    logging.info("Model seçildi ve eğitildi.")

    # Risk yönetimi
    rm = RiskManager(initial_balance=10000)  # İlk bakiye ile RiskManager başlat

    # Strateji oluşturma
    strategy = TradingStrategy(best_model)  # Ticaret stratejisini oluştur

    # Backtesting
    backtester = Backtester(df, best_model, rm)  # Backtester'ı başlat
    results = backtester.run()  # Backtesting işlemini başlat
    metrics = backtester.calculate_metrics()  # Performans metriklerini hesapla

    logging.info("Backtesting tamamlandı.")
    logging.info("Performans metrikleri:")
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")

    # Sonuçların görselleştirilmesi
    backtester.plot_results()  # Sonuçları görselleştir

    # Parametre optimizasyonu
    param_grid = {
        'lookback': [30, 60, 90],  # Bakış aralıkları
        'threshold': [0.5, 0.6, 0.7]  # Eşik değerleri
    }
    best_params = backtester.optimize_parameters(param_grid)  # Parametreleri optimize et
    logging.info(f"En iyi parametreler: {best_params}")

    logging.info("Ana program tamamlandı.")

if __name__ == "__main__":
    main()
