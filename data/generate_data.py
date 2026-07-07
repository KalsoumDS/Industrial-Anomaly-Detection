"""
Chargement du dataset SKAB (Skoltech Anomaly Benchmark).
Dataset réel de capteurs industriels d'une pompe hydraulique.
Source : https://github.com/waico/SKAB (licence GPL-3.0)
Paper : Filonov et al., Skolkovo Institute of Science and Technology, 2020.

8 capteurs réels :
  - Accelerometer1RMS, Accelerometer2RMS : vibrations (g)
  - Current : courant moteur (A)
  - Pressure : pression hydraulique (bar)
  - Temperature : température fluide (°C)
  - Thermocouple : température capteur (°C)
  - Voltage : tension alimentation (V)
  - Volume Flow RateRMS : débit volumique (L/min)
"""
import numpy as np
import pandas as pd
import requests
import io
import os

# URLs des fichiers SKAB sur GitHub (licence GPL-3.0)
SKAB_BASE_URL = "https://raw.githubusercontent.com/waico/SKAB/master/data/other"
SKAB_FILES = [str(i) for i in range(1, 10)]  # fichiers 1.csv à 9.csv

FEATURES = [
    'Accelerometer1RMS', 'Accelerometer2RMS', 'Current',
    'Pressure', 'Temperature', 'Thermocouple',
    'Voltage', 'Volume Flow RateRMS'
]

FEATURE_LABELS = {
    'Accelerometer1RMS':   'Accéléromètre 1 (g)',
    'Accelerometer2RMS':   'Accéléromètre 2 (g)',
    'Current':             'Courant (A)',
    'Pressure':            'Pression (bar)',
    'Temperature':         'Température fluide (°C)',
    'Thermocouple':        'Thermocouple (°C)',
    'Voltage':             'Tension (V)',
    'Volume Flow RateRMS': 'Débit volumique (L/min)',
}

FEATURE_COLORS = {
    'Accelerometer1RMS':   '#ff6b6b',
    'Accelerometer2RMS':   '#ff8e53',
    'Current':             '#4ecdc4',
    'Pressure':            '#45b7d1',
    'Temperature':         '#96ceb4',
    'Thermocouple':        '#feca57',
    'Voltage':             '#a29bfe',
    'Volume Flow RateRMS': '#fd79a8',
}


def load_skab_data(cache_path: str = None, n_files: int = 5) -> pd.DataFrame:
    """
    Télécharge et concatène les fichiers SKAB depuis GitHub.
    Utilise un cache local pour éviter de re-télécharger.

    Args:
        cache_path : chemin de sauvegarde locale (optionnel)
        n_files    : nombre de fichiers à charger (1-9)

    Returns:
        DataFrame avec colonnes capteurs + 'timestamp' + 'is_anomaly' + 'is_changepoint'
    """
    if cache_path and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=['timestamp'])
        return df

    dfs = []
    for fname in SKAB_FILES[:n_files]:
        url = f"{SKAB_BASE_URL}/{fname}.csv"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            df_part = pd.read_csv(io.StringIO(resp.text), sep=';', parse_dates=['datetime'])
            df_part = df_part.rename(columns={
                'datetime': 'timestamp',
                'anomaly': 'is_anomaly',
                'changepoint': 'is_changepoint'
            })
            df_part['is_anomaly'] = df_part['is_anomaly'].astype(bool)
            df_part['is_changepoint'] = df_part['is_changepoint'].astype(bool)
            df_part['file_id'] = int(fname)
            dfs.append(df_part)
        except Exception as e:
            print(f"⚠️  Impossible de charger {fname}.csv : {e}")

    if not dfs:
        raise RuntimeError("Impossible de télécharger les données SKAB. Vérifier la connexion internet.")

    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    df = df.dropna(subset=FEATURES)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


def get_normal_data(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne uniquement les données normales (pour l'entraînement)."""
    return df[~df['is_anomaly']].reset_index(drop=True)


def get_feature_columns() -> list:
    return FEATURES


if __name__ == '__main__':
    print("📥 Téléchargement du dataset SKAB...")
    df = load_skab_data(cache_path='saved_model/skab_data.csv', n_files=5)
    print(f"✅ Dataset chargé : {len(df)} échantillons, {len(FEATURES)} capteurs")
    print(f"   Anomalies : {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
    print(f"   Période   : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(df[FEATURES].describe().round(3))
