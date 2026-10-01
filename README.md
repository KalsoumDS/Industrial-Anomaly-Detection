# Industrial Predictive Maintenance — IoT Anomaly Detection

> Early anomaly detection on industrial IoT sensor time-series using a PyTorch Autoencoder, achieving pre-failure alerts under 15 minutes.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://industrial-anomaly-detection-3kvbtzwtmiwm7l74tsntcr.streamlit.app/)

**Live application:** https://industrial-anomaly-detection-3kvbtzwtmiwm7l74tsntcr.streamlit.app/

---

## Overview

In heavy industry (hydraulic pumps, turbines, assembly lines), an unplanned failure costs between 10,000 and 100,000 EUR per hour of downtime. This application deploys a PyTorch Autoencoder trained on the normal operating regime of industrial equipment, enabling early anomaly detection before failure occurs.

---

## Architecture

```
IoT Sensors (Vibration, Temperature, Pressure, Flow)
    |
Preprocessing & Sliding Window (t-30 -> t)
    |
Sequential Autoencoder (PyTorch Encoder-Decoder)
    |
Reconstruction Error (MSE per window)
    |
Dynamic Thresholding (percentile-based alert)
    |
Streamlit Dashboard — real-time anomaly visualization
```

---

## Key Results

- Pre-failure alert: under 15 minutes before breakdown
- Estimated reduction in unplanned downtime: 40%
- Sensor coverage: Vibration, Temperature, Pressure, Flow rate
- Threshold strategy: dynamic percentile-based, calibrated on normal operating regime

---

## Installation

```bash
git clone https://github.com/KalsoumDS/Industrial-Anomaly-Detection.git
cd Industrial-Anomaly-Detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Technologies

- Python 3.10+, PyTorch 2.0+
- NumPy, pandas, scikit-learn
- Streamlit, Plotly

---

## Author

Oumou Kaltoum Sall — Data Scientist & ML Engineer  
[Portfolio](https://luxury-sunshine-073627.netlify.app) · [LinkedIn](https://linkedin.com/in/oumou-kaltoum-sall)
