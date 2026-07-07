"""
Script d'entraînement — Industrial Anomaly Detection
Dataset : SKAB (Skoltech Anomaly Benchmark) — données réelles de pompe industrielle
Source  : https://github.com/waico/SKAB (GPL-3.0)
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

from data.generate_data import load_skab_data, get_normal_data, get_feature_columns
from model.autoencoder import AnomalyDetector

SAVE_DIR = 'saved_model'
FEATURES = get_feature_columns()


def train(
    n_files: int = 5,
    sequence_length: int = 30,
    epochs: int = 50,
    hidden_size: int = 64,
    latent_size: int = 16,
    batch_size: int = 32,
    lr: float = 1e-3,
    threshold_percentile: float = 95.0,
):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 1. Chargement des données SKAB ──────────────────────────────────────
    print("📥 Chargement du dataset SKAB (données réelles de pompe industrielle)...")
    df = load_skab_data(
        cache_path=os.path.join(SAVE_DIR, 'skab_data.csv'),
        n_files=n_files
    )
    print(f"✅ {len(df)} échantillons chargés — {df['is_anomaly'].sum()} anomalies réelles ({df['is_anomaly'].mean()*100:.1f}%)")

    # ── 2. Séparation train (normal) / test (avec anomalies) ─────────────────
    df_normal = get_normal_data(df)
    print(f"   Données normales (train) : {len(df_normal)} échantillons")
    print(f"   Données complètes (test) : {len(df)} échantillons")

    # ── 3. Normalisation ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_normal[FEATURES].values)
    joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
    print(f"✅ Scaler StandardScaler ajusté et sauvegardé")

    # ── 4. Entraînement ──────────────────────────────────────────────────────
    print(f"\n🔧 Entraînement de l'Autoencoder LSTM (PyTorch)...")
    print(f"   Dataset  : SKAB — pompe hydraulique industrielle")
    print(f"   Capteurs : {len(FEATURES)} ({', '.join(FEATURES[:4])}...)")
    print(f"   Séquence : {sequence_length} pas de temps (1 sec/pas)")
    print(f"   Epochs   : {epochs} | Hidden: {hidden_size} | Latent: {latent_size}")

    detector = AnomalyDetector(
        input_size=len(FEATURES),
        hidden_size=hidden_size,
        latent_size=latent_size,
        sequence_length=sequence_length
    )

    def progress(epoch, total, loss):
        if epoch % 10 == 0 or epoch == total:
            bar = '█' * int(epoch / total * 20) + '░' * (20 - int(epoch / total * 20))
            print(f"   [{bar}] {epoch:3d}/{total} — Loss: {loss:.6f}")

    losses = detector.train_model(
        train_scaled, epochs=epochs,
        batch_size=batch_size, lr=lr,
        progress_callback=progress
    )

    # ── 5. Seuil d'anomalie ──────────────────────────────────────────────────
    threshold = detector.compute_threshold(train_scaled, percentile=threshold_percentile)
    print(f"\n✅ Seuil calculé : {threshold:.6f} (P{threshold_percentile} des erreurs normales)")

    # ── 6. Évaluation sur données complètes (avec vraies anomalies SKAB) ────
    test_scaled = scaler.transform(df[FEATURES].values)
    result = detector.detect(test_scaled)
    true_anomalies = df['is_anomaly'].values[sequence_length - 1:sequence_length - 1 + len(result['errors'])]

    print(f"\n📈 Évaluation sur SKAB (vraies anomalies labelisées) :")
    print(f"   Anomalies détectées  : {result['n_anomalies']} ({result['anomaly_rate']:.1f}%)")
    print(f"   Vraies anomalies     : {true_anomalies.sum()} ({true_anomalies.mean()*100:.1f}%)")

    # ── 7. Sauvegarde ────────────────────────────────────────────────────────
    model_path = os.path.join(SAVE_DIR, 'autoencoder.pt')
    detector.save(model_path)
    print(f"\n✅ Modèle sauvegardé : {model_path}")
    print(f"🚀 Lancer le dashboard : streamlit run app.py")

    return detector, losses


if __name__ == '__main__':
    train()
