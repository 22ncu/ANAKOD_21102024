import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import logging

# Attempt to import xgboost and lightgbm with graceful fallbacks
try:
    from xgboost import XGBRegressor
except ImportError:
    logging.warning("XGBoost is not installed. Skipping XGBRegressor.")

try:
    from lightgbm import LGBMRegressor
except ImportError:
    logging.warning("LightGBM is not installed. Skipping LGBMRegressor.")

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

class ModelSelector:
    def __init__(self, X, y, n_splits=5):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.scaler = StandardScaler()
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf')
        }
        if 'XGBRegressor' in globals():
            self.models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
        if 'LGBMRegressor' in globals():
            self.models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42)

        logging.info("ModelSelector initialized with {} splits".format(n_splits))

    def evaluate_models(self):
        results = {}
        X_scaled = self.scaler.fit_transform(self.X)

        for name, model in self.models.items():
            mse_scores = cross_val_score(model, X_scaled, self.y, scoring='neg_mean_squared_error', cv=self.tscv)
            mae_scores = cross_val_score(model, X_scaled, self.y, scoring='neg_mean_absolute_error', cv=self.tscv)
            r2_scores = cross_val_score(model, X_scaled, self.y, scoring='r2', cv=self.tscv)

            results[name] = {
                'MSE': -np.mean(mse_scores),
                'MAE': -np.mean(mae_scores),
                'R2': np.mean(r2_scores)
            }

        logging.info(f"Evaluation results: {results}")
        return results

    def select_best_model(self, metric='MSE'):
        results = self.evaluate_models()
        best_model = min(results, key=lambda x: results[x][metric])
        logging.info(f"Best model based on {metric}: {best_model}")
        return self.models[best_model]

    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def evaluate_lstm(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        model = self.create_lstm_model((1, X_train_scaled.shape[1]))
        model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

        y_pred = model.predict(X_test_reshaped)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"LSTM Evaluation: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        return model, {'MSE': mse, 'MAE': mae, 'R2': r2}

def prepare_data_for_lstm(df, target_col, lookback=60):
    data = df.filter([target_col]).values
    scaled_data = StandardScaler().fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y
