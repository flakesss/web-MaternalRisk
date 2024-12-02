# model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Fungsi untuk melatih model
def train_model():
    # Membaca data dari file CSV
    data = pd.read_csv('maternalHealthRisk.csv')

    # Pra-pemrosesan data
    # Mengubah kolom 'RiskLevel' menjadi numerik
    label_encoder = LabelEncoder()
    data['RiskLevel_encoded'] = label_encoder.fit_transform(data['RiskLevel'])

    # Memisahkan fitur dan label
    X = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
    y = data['RiskLevel_encoded']

    # Mengatasi ketidakseimbangan kelas dengan SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Standarisasi fitur
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)

    # Hyperparameter tuning dengan GridSearchCV
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5]
    }

    # Menggunakan Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=skf, n_jobs=-1, verbose=0)
    grid_search_rf.fit(X_resampled_scaled, y_resampled)

    # Model terbaik setelah tuning
    best_rf = grid_search_rf.best_estimator_

    # Menyimpan model dan scaler
    joblib.dump(best_rf, 'model/model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(label_encoder, 'model/label_encoder.pkl')

    print("Model telah dilatih dan disimpan.")

if __name__ == "__main__":
    train_model()
