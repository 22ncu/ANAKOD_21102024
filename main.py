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

    # Veri yükleme
    df = pd.read_csv("historical_data.csv", index_col=0, parse_dates=True)
    logging.info(f"Veri yüklendi. Şekil: {df.shape}")

    # Indikatörlerin hesaplanması
    indicators = Indicators()
    df = indicators.calculate_indicators(df)
    df = indicators.detect_candlestick_patterns(df)
    df = indicators.calculate_fibonacci_levels(df)
    logging.info("Tüm indikatörler hesaplandı.")

    # Gelişmiş indikatörlerin hesaplanması
    advanced_ind = AdvancedIndicators()
    df = advanced_ind.calculate_advanced_indicators(df)
    df = advanced_ind.calculate_market_regime(df)
    df = advanced_ind.calculate_support_resistance(df)
    logging.info("Gelişmiş indikatörler hesaplandı.")

    # Feature engineering
    fe = FeatureEngineer()
    df = fe.engineer_features(df)
    logging.info("Özellik mühendisliği tamamlandı.")

    # Model seçimi ve eğitimi
    ms = ModelSelector(df)
    model, scaler = ms.select_and_train_model()
    logging.info("Model seçildi ve eğitildi.")

    # Risk yönetimi
    rm = RiskManager(initial_balance=10000)

    # Strateji oluşturma
    strategy = TradingStrategy(model, scaler)

    # Backtesting
    backtester = Backtester(df, model, scaler, initial_balance=10000)
    results = backtester.run()
    metrics = backtester.calculate_metrics()

    logging.info("Backtesting tamamlandı.")
    logging.info("Performans metrikleri:")
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")

    # Sonuçların görselleştirilmesi
    backtester.plot_results()

    # Parametre optimizasyonu
    param_grid = {
        'lookback': [30, 60, 90],
        'threshold': [0.5, 0.6, 0.7]
    }
    best_params = backtester.optimize_parameters(param_grid)
    logging.info(f"En iyi parametreler: {best_params}")

    logging.info("Ana program tamamlandı.")

if __name__ == "__main__":
    main()
