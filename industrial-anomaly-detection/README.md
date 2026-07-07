# 🏭 Industrial Anomaly Detection

> Détection d'anomalies en temps réel sur capteurs industriels — Autoencoder LSTM PyTorch + Dashboard Streamlit

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-ff4b4b)

## 🎯 Objectif

Ce projet implémente un système de détection d'anomalies non supervisé sur des séries temporelles industrielles (données de capteurs : température, vibration, pression, courant).

**Principe :** Un Autoencoder LSTM apprend à reconstruire les données **normales**. Toute séquence anormale génère une erreur de reconstruction élevée → **anomalie détectée**.

## 🏗️ Architecture

```
Capteurs → Fenêtres glissantes → LSTM Encodeur → Vecteur latent
                                                       ↓
Erreur MSE ← Reconstruction ← LSTM Décodeur ←────────┘
     ↓
  > Seuil ? → ANOMALIE 🔴
```

## 📊 Types d'anomalies détectées

| Type | Description | Capteur |
|------|-------------|---------|
| Spike | Pic soudain (×2.5) | Température, Vibration |
| Drift | Dérive progressive | Pression, Courant |
| Drop | Chute soudaine (×0.5) | Température, Pression |

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 🏃 Utilisation

### 1. Entraîner le modèle
```bash
python train.py
```

### 2. Lancer le dashboard
```bash
streamlit run app.py
```

Ou directement depuis le dashboard : cliquer sur **"(Ré)entraîner le modèle"**.

## 📁 Structure

```
industrial-anomaly-detection/
├── model/
│   ├── autoencoder.py      ← Architecture Autoencoder LSTM + AnomalyDetector
│   └── __init__.py
├── data/
│   ├── generate_data.py    ← Génération de données synthétiques
│   └── __init__.py
├── saved_model/            ← Modèle entraîné (généré automatiquement)
│   ├── autoencoder.pt
│   ├── scaler.pkl
│   └── demo_data.csv
├── app.py                  ← Dashboard Streamlit
├── train.py                ← Script d'entraînement
├── requirements.txt
└── README.md
```

## 🛠️ Stack technique

- **PyTorch** — Architecture Autoencoder LSTM
- **Streamlit** — Dashboard interactif temps réel
- **Plotly** — Visualisations dynamiques
- **Scikit-learn** — Normalisation (StandardScaler)
- **NumPy / Pandas** — Traitement des données

## 📈 Résultats

- Détection non supervisée (pas de labels nécessaires à l'entraînement)
- Seuil adaptatif basé sur le percentile des erreurs normales
- Dashboard temps réel avec alertes visuelles
- Distribution des erreurs et table des anomalies détectées

## 👩‍💻 Auteur

**Oumou Kaltoum Sall** — Data Scientist & ML Engineer  
[GitHub](https://github.com/KalsoumDS) · [Email](mailto:s.sall@mundiapolis.ma)
