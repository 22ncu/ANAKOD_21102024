import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# XGBoost ve LightGBM'i isteğe bağlı olarak içe aktarma
try:
    from xgboost import XGBRegressor
except ImportError:
    logging.warning("XGBoost is not installed. Skipping XGBRegressor.")

try:
    from lightgbm import LGBMRegressor
except ImportError:
    logging.warning("LightGBM is not installed. Skipping LGBMRegressor.")

# Keras modüllerini içe aktarma
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelSelector:
    def __init__(self, X, y, n_splits=5):
        """
        ModelSelector sınıfı, verilen özellikler (X) ve hedef değişken (y) ile 
        model seçimi ve eğitimi için gerekli ayarları yapar.
        
        Args:
            X (array-like): Özellik matrisi.
            y (array-like): Hedef değişken.
            n_splits (int): Zaman serisi verisi için çapraz doğrulama sayısı.
        """
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)  # Zaman serisi verisi için çapraz doğrulama
        self.scaler = StandardScaler()  # Veriyi ölçeklendirmek için scaler
        self.models = {  # Kullanılacak modeller
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf')
        }
        
        # XGBoost ve LightGBM modellerini ekleme
        if 'XGBRegressor' in globals():
            self.models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
        if 'LGBMRegressor' in globals():
            self.models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42)

        logging.info("ModelSelector initialized with {} splits".format(n_splits))

    def evaluate_models(self):
        """
        Mevcut modellerin MSE, MAE ve R2 skorlarını değerlendirir.
        
        Returns:
            results (dict): Her modelin performans skorları.
        """
        results = {}
        X_scaled = self.scaler.fit_transform(self.X)  # Özellikleri ölçekle

        for name, model in self.models.items():
            try:
                # Her model için MSE, MAE ve R2 skorlarını hesapla
                mse_scores = cross_val_score(model, X_scaled, self.y, scoring='neg_mean_squared_error', cv=self.tscv)
                mae_scores = cross_val_score(model, X_scaled, self.y, scoring='neg_mean_absolute_error', cv=self.tscv)
                r2_scores = cross_val_score(model, X_scaled, self.y, scoring='r2', cv=self.tscv)

                results[name] = {
                    'MSE': -np.mean(mse_scores),
                    'MAE': -np.mean(mae_scores),
                    'R2': np.mean(r2_scores)
                }
                logging.info(f"Model: {name}, MSE: {results[name]['MSE']:.4f}, MAE: {results[name]['MAE']:.4f}, R2: {results[name]['R2']:.4f}")
            except Exception as e:
                logging.error(f"Error evaluating model {name}: {e}")

        logging.info(f"Evaluation results: {results}")
        return results

    def select_best_model(self, metric='MSE'):
        """
        Belirtilen metriğe göre en iyi modeli seçer.
        
        Args:
            metric (str): Değerlendirme metriği (varsayılan 'MSE').
        
        Returns:
            model: En iyi model.
        """
        results = self.evaluate_models()  # Modelleri değerlendir
        best_model = min(results, key=lambda x: results[x][metric])  # En iyi modeli seç
        logging.info(f"Best model based on {metric}: {best_model}")
        return self.models[best_model]  # En iyi modeli döndür

    def create_lstm_model(self, input_shape):
        """
        LSTM modelini oluşturur.
        
        Args:
            input_shape (tuple): LSTM girişi için şekil.
        
        Returns:
            model: Oluşturulan LSTM modeli.
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  # Modeli derle
        return model

    def evaluate_lstm(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """
        LSTM modelini değerlendirir ve performansını ölçer.
        
        Args:
            X_train (array-like): Eğitim özellikleri.
            y_train (array-like): Eğitim hedefi.
            X_test (array-like): Test özellikleri.
            y_test (array-like): Test hedefi.
            epochs (int): Eğitim dönemi sayısı.
            batch_size (int): Batch boyutu.
        
        Returns:
            model: Eğitilmiş LSTM modeli.
            metrics (dict): Metrikler (MSE, MAE, R2).
        """
        # LSTM modeli için verileri ölçekle
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Verileri uygun şekle dönüştür
        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        model = self.create_lstm_model((1, X_train_scaled.shape[1]))  # LSTM modelini oluştur
        model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)  # Modeli eğit

        y_pred = model.predict(X_test_reshaped)  # Tahminleri yap
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"LSTM Evaluation: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        return model, {'MSE': mse, 'MAE': mae, 'R2': r2}  # Sonuçları döndür

    def hyperparameter_tuning(self):
        """
        Hiperparametre optimizasyonu yapar.
        
        Returns:
            best_model: En iyi hiperparametrelerle eğitilmiş model.
        """
        try:
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid,
                                       cv=self.tscv, scoring='neg_mean_squared_error', n_jobs=-1)
            X_scaled = self.scaler.fit_transform(self.X)
            grid_search.fit(X_scaled, self.y)
            best_model = grid_search.best_estimator_
            logging.info(f"Best hyperparameters: {grid_search.best_params_}")
            return best_model
        except Exception as e:
            logging.error(f"Error during hyperparameter tuning: {e}")
            return None

def prepare_data_for_lstm(df, target_col, lookback=60):
    """
    LSTM için veriyi hazırlar.
    
    Args:
        df (DataFrame): Verilerin bulunduğu DataFrame.
        target_col (str): Hedef değişkenin adı.
        lookback (int): Geriye dönük bakılacak dönem sayısı.
        
    Returns:
        X (array-like): Özellik dizisi.
        y (array-like): Hedef dizisi.
    """
    data = df.filter([target_col]).values
    scaled_data = StandardScaler().fit_transform(data)  # Veriyi ölçeklendir

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])  # Geçmiş verileri ekle
        y.append(scaled_data[i, 0])  # Hedef veriyi ekle

    X, y = np.array(X), np.array(y)  # Listeleri numpy dizisine dönüştür
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Veriyi LSTM için uygun şekle getir

    return X, y  # Özellikler ve hedefi döndür
