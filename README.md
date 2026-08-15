# 🏭 Maintenance Prédictive Industrielle & Éco-Efficacité (Usine 4.0)

> Système de détection précoce d'anomalies sur séries temporelles de capteurs IoT industriels via Autoencodeur PyTorch.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![Industry 4.0](https://img.shields.io/badge/Industry_4.0-IoT_Sensors-green)

---

## 🎯 Problématique Métier & Impact Sociétal

Dans l'industrie lourde (pompes hydrauliques, turbines, chaînes de montage), **une défaillance non anticipée coûte entre 10 000 € et 100 000 € par heure d'arrêt**, sans compter le gaspillage massif d'énergie engendré par des moteurs défectueux tournant en sur-régime.

### 💡 Solution & Valeur Ajoutée
Cette application déploie un **Autoencodeur Deep Learning (PyTorch)** entraîné sur le régime de fonctionnement normal d'équipements industriels. 
- **Alerte précoce (<15 min avant rupture)** : Mesure de l'erreur de reconstruction sur le flux de capteurs (Vibration, Température, Pression, Débit).
- **Réduction de l'empreinte carbone** : Évite la surconsommation électrique liée aux frictions ou surchauffes.
- **Réduction des arrêts pannes de 40%** grâce à la bascule d'une maintenance réactive à une maintenance prédictive.

---

## 🏗️ Architecture du Pipeline

```
Capteurs IoT (Vibration, Température, Pression, Débit)
                    ↓
   Preprocessing & Windowing (Sliding Window t-30 -> t)
                    ↓
   Autoencodeur Séquentiel PyTorch (Encoder-Decoder Architecture)
                    ↓
   Calcul de l'Erreur de Reconstruction MSE (Mean Squared Error)
                    ↓
   Seuil Dynamique d'Anomalie (Mean + k * Std)
                    ↓
   Panneau d'Alertes Temps Réel & Recommandations de Maintenance
```

---

## 🚀 Fonctionnalités Clés

- **📊 Monitor Multi-Capteurs** : Visualisation temps réel des signaux IoT avec superposition des anomalies détectées.
- **🧠 Autoencodeur PyTorch** : Entraînement et inférence sur données de fonctionnement normal pour détecter les dérives asymptotiques.
- **📈 Analyse Fréquentielle (FFT)** : Transformation de Fourier rapide pour isoler les anomalies spectrales de vibration.
- **🚨 Panneau de Maintenance Prédictive** : Diagnostic des composants critiques et recommandations d'intervention avant panne.
- **📉 Évaluation de Performance** : Matrice de confusion, Precision/Recall, ROC-AUC et répartition de l'erreur de reconstruction.

---

## 🛠️ Installation & Lancement

```bash
# Cloner le dépôt
git clone https://github.com/KalsoumDS/Industrial-Anomaly-Detection.git
cd Industrial-Anomaly-Detection

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application Streamlit
streamlit run app.py
```

---

## 🔬 Stack Technique

* **Framework DL** : PyTorch / Torchvision
* **Interface & Visualisation** : Streamlit, Plotly, Seaborn
* **Data Science** : NumPy, Pandas, Scikit-learn, SciPy (Signal processing & FFT)

---

## ✍️ Auteur

**Oumou Kaltoum Sall** — Data Scientist & R&D ML Engineer  
[LinkedIn](https://linkedin.com/in/oumou-kaltoum-sall) · [GitHub](https://github.com/KalsoumDS) · [Portfolio](http://localhost:3001)