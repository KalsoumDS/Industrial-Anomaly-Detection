"""
Dashboard Streamlit — Industrial Anomaly Detection
Détection d'anomalies en temps réel sur capteurs industriels.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import joblib
import os
import time

from data.generate_data import load_skab_data, get_normal_data, get_feature_columns, FEATURE_LABELS, FEATURE_COLORS
from model.autoencoder import AnomalyDetector

# ── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Industrial Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEATURES = get_feature_columns()
# FEATURE_LABELS et FEATURE_COLORS importés depuis data/generate_data.py
MODEL_DIR = 'saved_model'

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header styling */
h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* Section header */
.section-header {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
    margin: 2rem 0 0.75rem 0 !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid #e5e7eb !important;
}

@media (prefers-color-scheme: dark) {
    .section-header {
        color: #9ca3af !important;
        border-bottom-color: #374151 !important;
    }
}

/* Info box */
.info-box {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #1f2937;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
}

.info-box h3, .info-box h4 {
    color: #1f2937;
}

.info-box p, .info-box li {
    color: inherit;
}

/* Dark mode styles */
@media (prefers-color-scheme: dark) {
    .info-box {
        background-color: #1e293b;
        border-color: #334151;
        color: #f8fafc;
    }

    .info-box h3, .info-box h4 {
        color: #e0e7ff;
    }
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #f8fafc;
}

[data-testid="stSidebar"] h2 {
    color: #e0e7ff !important;
}

[data-testid="stSidebar"] h3 {
    color: #c7d2fe !important;
}

/* Alert boxes */
.alert-box {
    padding: 12px 20px; border-radius: 8px; margin: 8px 0;
    font-weight: 600; font-size: 0.95rem;
}
.alert-critical { background: rgba(255,59,59,0.15); border-left: 4px solid #ff3b3b; color: #ff3b3b; }
.alert-warning  { background: rgba(255,165,0,0.15);  border-left: 4px solid #ffa500; color: #ffa500; }
.alert-normal   { background: rgba(0,200,100,0.15);  border-left: 4px solid #00c864; color: #00c864; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding: 0 1.5rem;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── FONCTIONS UTILITAIRES ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """Charge le modèle sauvegardé ou l'entraîne si absent."""
    model_path = os.path.join(MODEL_DIR, 'autoencoder.pt')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        detector = AnomalyDetector.load(model_path)
        scaler = joblib.load(scaler_path)
        return detector, scaler, False
    else:
        return None, None, True  # Besoin d'entraînement


@st.cache_data(show_spinner=False)
def get_demo_data(n_files: int = 5):
    """Charge les données réelles SKAB."""
    cache_path = os.path.join(MODEL_DIR, 'skab_data.csv')
    return load_skab_data(cache_path=cache_path, n_files=n_files)


def train_model_in_app(n_files, seq_len, epochs, hidden, latent):
    """Entraîne le modèle depuis l'interface Streamlit avec données SKAB réelles."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    st.info("Téléchargement du dataset SKAB (données réelles de pompe industrielle)...")
    df = load_skab_data(
        cache_path=os.path.join(MODEL_DIR, 'skab_data.csv'),
        n_files=n_files
    )
    df_normal = get_normal_data(df)
    st.success(f"{len(df)} échantillons SKAB chargés — {df['is_anomaly'].sum()} anomalies réelles")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_normal[FEATURES].values)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    detector = AnomalyDetector(
        input_size=len(FEATURES), hidden_size=hidden,
        latent_size=latent, sequence_length=seq_len
    )

    progress_bar = st.progress(0)
    loss_placeholder = st.empty()
    losses = []

    def cb(epoch, total, loss):
        losses.append(loss)
        progress_bar.progress(epoch / total)
        loss_placeholder.markdown(f"**Epoch {epoch}/{total}** — Loss: `{loss:.6f}`")

    detector.train_model(train_scaled, epochs=epochs, progress_callback=cb)
    detector.compute_threshold(train_scaled)
    detector.save(os.path.join(MODEL_DIR, 'autoencoder.pt'))

    progress_bar.progress(1.0)
    loss_placeholder.markdown(f"**Entraînement terminé !** Loss finale: `{losses[-1]:.6f}`")

    st.cache_resource.clear()
    st.cache_data.clear()
    return losses


def plot_sensor_data(df: pd.DataFrame, errors: np.ndarray,
                     threshold: float, selected_feature: str):
    """Graphique principal capteur + score d'anomalie."""
    seq_len = 30
    # Aligner les erreurs avec les timestamps
    n_errors = len(errors)
    timestamps = df['timestamp'].values[seq_len - 1:][:n_errors]
    is_anomaly_arr = errors > threshold

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"📡 {FEATURE_LABELS.get(selected_feature, selected_feature)}",
            "🔴 Score de reconstruction (erreur MSE)"
        ],
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )

    # Signal capteur
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df[selected_feature],
        mode='lines', name=FEATURE_LABELS.get(selected_feature),
        line=dict(color=FEATURE_COLORS.get(selected_feature, '#7c6af7'), width=1.5)
    ), row=1, col=1)

    # Points anomalies sur le signal
    anomaly_mask_full = np.zeros(len(df), dtype=bool)
    anomaly_mask_full[seq_len - 1:seq_len - 1 + n_errors] = is_anomaly_arr
    if anomaly_mask_full.any():
        fig.add_trace(go.Scatter(
            x=df['timestamp'][anomaly_mask_full],
            y=df[selected_feature][anomaly_mask_full],
            mode='markers', name='⚠️ Anomalie',
            marker=dict(color='red', size=8, symbol='x')
        ), row=1, col=1)

    # Score d'anomalie
    fig.add_trace(go.Scatter(
        x=timestamps, y=errors,
        mode='lines', name='Score MSE',
        line=dict(color='#a78bfa', width=1.5),
        fill='tozeroy', fillcolor='rgba(167,139,250,0.1)'
    ), row=2, col=1)

    # Seuil
    fig.add_hline(
        y=threshold, line_dash='dash', line_color='red',
        annotation_text=f"Seuil: {threshold:.4f}",
        annotation_position="top right", row=2, col=1
    )

    fig.update_layout(
        template='plotly_dark', height=500,
        showlegend=True, margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_all_sensors(df: pd.DataFrame):
    """Vue d'ensemble des 4 capteurs."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[FEATURE_LABELS[f] for f in FEATURES],
        vertical_spacing=0.15, horizontal_spacing=0.08
    )
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for feat, (r, c) in zip(FEATURES, positions):
        color = FEATURE_COLORS[feat]
        # Normal
        normal_mask = ~df['is_anomaly'] if 'is_anomaly' in df.columns else pd.Series([True]*len(df))
        fig.add_trace(go.Scatter(
            x=df['timestamp'][normal_mask], y=df[feat][normal_mask],
            mode='lines', name=feat, line=dict(color=color, width=1),
            showlegend=False
        ), row=r, col=c)
        # Anomalies
        if 'is_anomaly' in df.columns and df['is_anomaly'].any():
            anom_mask = df['is_anomaly']
            fig.add_trace(go.Scatter(
                x=df['timestamp'][anom_mask], y=df[feat][anom_mask],
                mode='markers', name='Anomalie',
                marker=dict(color='red', size=5), showlegend=False
            ), row=r, col=c)

    fig.update_layout(
        template='plotly_dark', height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_loss_curve(losses: list):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=losses, mode='lines+markers',
        line=dict(color='#c8ff00', width=2),
        marker=dict(size=4), name='Loss'
    ))
    fig.update_layout(
        template='plotly_dark', height=250,
        title='Courbe de loss (entraînement)',
        xaxis_title='Epoch', yaxis_title='MSE Loss',
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_error_distribution(errors: np.ndarray, threshold: float):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors, nbinsx=50,
        marker_color='#7c6af7', opacity=0.8, name='Erreurs'
    ))
    fig.add_vline(
        x=threshold, line_dash='dash', line_color='red',
        annotation_text=f'Seuil ({threshold:.4f})',
        annotation_position='top right'
    )
    fig.update_layout(
        template='plotly_dark', height=280,
        title='Distribution des erreurs de reconstruction',
        xaxis_title='Erreur MSE', yaxis_title='Fréquence',
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    # Industrial Anomaly Detection
    ## Détection d'anomalies sur capteurs industriels — Autoencoder LSTM PyTorch
    """)
    st.divider()

    # ── ONGLETS PRINCIPAUX (nouvel ordre !) ──
    tab_welcome, tab_train, tab_detect, tab_overview, tab_stats, tab_model = st.tabs([
        "Bienvenue", "Entraîner le modèle", "Détecter des anomalies", 
        "Vue d'ensemble", "Statistiques", "Comment ça marche ?"
    ])

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("## Configuration")

        st.markdown("### Données SKAB")
        n_files = st.slider("Nombre de fichiers SKAB", 1, 9, 5,
                            help="Chaque fichier = ~1000 échantillons réels de capteurs industriels")

        st.markdown("### Modèle")
        seq_len   = st.selectbox("Longueur de séquence", [20, 30, 50], index=1)
        epochs    = st.slider("Epochs d'entraînement", 10, 100, 50, 10)
        hidden    = st.selectbox("Hidden size", [32, 64, 128], index=1)
        latent    = st.selectbox("Latent size", [8, 16, 32], index=1)
        percentile = st.slider("Percentile seuil (%)", 80, 99, 95, 1,
                                help="Plus élevé = moins d'alertes (plus strict)")

        st.markdown("### Entraînement")
        train_btn = st.button("(Ré)entraîner le modèle", type="primary", use_container_width=True)

        st.divider()
        st.markdown("### Capteur affiché")
        selected_feature = st.selectbox(
            "Sélectionner un capteur",
            FEATURES,
            format_func=lambda x: FEATURE_LABELS[x]
        )

        st.divider()
        st.markdown("""
        **Dataset : SKAB**
        
        Skoltech Anomaly Benchmark — données réelles d'une pompe hydraulique industrielle.
        8 capteurs, anomalies labelisées.
        
        [Paper SKAB](https://github.com/waico/SKAB) · [GitHub](https://github.com/KalsoumDS)
        """)

    # ── ONGLET 1: BIENVENUE ──
    with tab_welcome:
        st.markdown('<div class="section-header">Bienvenue</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <h3>Présentation du projet</h3>
            <p style="margin: 0.5rem 0;">Cette application permet de détecter automatiquement les anomalies sur une machine industrielle (pompe hydraulique) en analysant les données de ses capteurs.</p>
            <h4 style="margin: 1rem 0 0.5rem 0;">Pourquoi ça importe ?</h4>
            <p style="margin: 0.5rem 0;">En industrie, détecter les défauts tôt permet :</p>
            <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                <li>D'éviter des pannes coûteuses</li>
                <li>De planifier la maintenance (maintenance prédictive)</li>
                <li>D'assurer la sécurité des opérateurs</li>
            </ul>
            <h4 style="margin: 1rem 0 0.5rem 0;">Étapes pour utiliser l'app :</h4>
            <ol style="margin: 0.5rem 0; padding-left: 1.5rem;">
                <li>Entraîner le modèle (onglet "Entraîner le modèle")</li>
                <li>Détecter des anomalies (onglet "Détecter des anomalies")</li>
                <li>Explorer les résultats</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Capteurs utilisés</div>', unsafe_allow_html=True)
        for feat, label in FEATURE_LABELS.items():
            st.markdown(f"- **{label}** ({feat})")

    # ── ONGLET 2: ENTRAÎNEMENT ──
    with tab_train:
        st.markdown('<div class="section-header">Entraîner le modèle</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <h3>Comment fonctionne l'entraînement ?</h3>
            <p style="margin: 0.5rem 0;">Le modèle est un Autoencoder LSTM : il apprend à reconstruire les données normales de la machine.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if train_btn:
            st.markdown("### Entraînement en cours...")
            losses = train_model_in_app(n_files, seq_len, epochs, hidden, latent)
            st.plotly_chart(plot_loss_curve(losses), use_container_width=True)
            st.success("Modèle entraîné et sauvegardé !")
            time.sleep(1)
            st.rerun()
        else:
            st.info("Cliquez sur le bouton \"(Ré)entraîner le modèle\" dans la barre latérale pour commencer.")

    # ── CHARGEMENT MODÈLE ET DONNÉES (pour les onglets suivants) ──
    detector, scaler, needs_training = load_or_train_model()

    if needs_training:
        for tab in [tab_detect, tab_overview, tab_stats, tab_model]:
            with tab:
                st.info("Aucun modèle trouvé. Veuillez d'abord l'entraîner dans l'onglet 'Entraîner le modèle'.")
        st.stop()

    # Charger les données
    df = get_demo_data(n_files)
    data_scaled = scaler.transform(df[FEATURES].values)
    errors = detector.compute_reconstruction_errors(data_scaled)
    threshold = float(np.percentile(errors, percentile))
    is_anomaly = errors > threshold

    # ── KPIs (pour les onglets) ──
    n_anomalies = int(is_anomaly.sum())
    anomaly_rate = float(is_anomaly.mean() * 100)
    max_error = float(errors.max())
    mean_error = float(errors.mean())
    status = "CRITIQUE" if anomaly_rate > 8 else ("ATTENTION" if anomaly_rate > 3 else "NORMAL")
    status_class = "alert-critical" if anomaly_rate > 8 else ("alert-warning" if anomaly_rate > 3 else "alert-normal")

    # ── ONGLET 3: DÉTECTION ──
    with tab_detect:
        st.markdown(f"""
        <div class="alert-box {status_class}" style="margin-bottom:20px">
            {status} — {n_anomalies} anomalies détectées ({anomaly_rate:.1f}% du signal)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">KPIs</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Anomalies", f"{n_anomalies}", f"{anomaly_rate:.1f}%")
        with col2:
            st.metric("Erreur moyenne", f"{mean_error:.4f}")
        with col3:
            st.metric("Erreur max", f"{max_error:.4f}")
        with col4:
            st.metric("Seuil", f"{threshold:.4f}")
        
        st.divider()
        st.markdown('<div class="section-header">Visualisation</div>', unsafe_allow_html=True)
        st.plotly_chart(
            plot_sensor_data(df, errors, threshold, selected_feature),
            use_container_width=True
        )
        # Table des anomalies
        seq_len_used = detector.sequence_length
        anom_indices = np.where(is_anomaly)[0]
        if len(anom_indices) > 0:
            st.markdown('<div class="section-header">Anomalies détectées</div>', unsafe_allow_html=True)
            anom_data = []
            for idx in anom_indices[:20]:
                real_idx = idx + seq_len_used - 1
                if real_idx < len(df):
                    row = df.iloc[real_idx]
                    anom_data.append({
                        'Timestamp': str(row['timestamp'])[:16],
                        'Score MSE': f"{errors[idx]:.6f}",
                        'Seuil': f"{threshold:.6f}",
                        'Ratio': f"{errors[idx]/threshold:.2f}x",
                        **{FEATURE_LABELS[f]: f"{row[f]:.3f}" for f in FEATURES}
                    })
            st.dataframe(pd.DataFrame(anom_data), use_container_width=True)
        else:
            st.success("Aucune anomalie détectée.")

    # ── ONGLET 4: VUE D'ENSEMBLE ──
    with tab_overview:
        st.markdown('<div class="section-header">Vue d\'ensemble</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <p style="margin: 0;">Visualisation de tous les capteurs en même temps pour avoir une vue d'ensemble de l'état de la machine.</p>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(plot_all_sensors(df), use_container_width=True)

    # ── ONGLET 5: STATISTIQUES ──
    with tab_stats:
        st.markdown('<div class="section-header">Distribution des erreurs</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_error_distribution(errors, threshold), use_container_width=True)
        
        st.markdown('<div class="section-header">Statistiques détaillées</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="info-box">
                <h4>Statistiques des erreurs</h4>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            - Min : `{errors.min():.6f}`
            - Max : `{errors.max():.6f}`
            - Moyenne : `{errors.mean():.6f}`
            - Std : `{errors.std():.6f}`
            - P95 (seuil) : `{np.percentile(errors, 95):.6f}`
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="info-box">
                <h4>Résultats de détection</h4>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            - Total séquences : `{len(errors)}`
            - Anomalies : `{n_anomalies}` (`{anomaly_rate:.1f}%`)
            - Normales : `{len(errors) - n_anomalies}` (`{100-anomaly_rate:.1f}%`)
            - Seuil utilisé : `{threshold:.6f}` (P{percentile})
            """)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── ONGLET 6: COMMENT ÇA MARCHE ──
    with tab_model:
        st.markdown('<div class="section-header">Architecture du modèle</div>', unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            <div class="info-box">
                <h4>Caractéristiques de l'Autoencoder LSTM</h4>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            - Input size : `{len(FEATURES)} capteurs`
            - Séquence : `{detector.sequence_length} pas de temps`
            - Hidden size : `{detector.model.hidden_size}`
            - Latent size : `{detector.model.latent_size}`
            - Layers : `{detector.model.num_layers}`
            - Device : `{detector.device}`
            """)
            total_params = sum(p.numel() for p in detector.model.parameters())
            trainable = sum(p.numel() for p in detector.model.parameters() if p.requires_grad)
            st.markdown(f"""
            **Paramètres :**
            - Total : `{total_params:,}`
            - Entraînables : `{trainable:,}`
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_m2:
            st.markdown("""
            <div class="info-box">
                <h4>Principe de fonctionnement</h4>
                <ol style="margin: 0.5rem 0; padding-left: 1.5rem;">
                    <li>Entraînement : on donne seulement des données normales au modèle</li>
                    <li>Encodage : LSTM compresse la séquence de capteurs en un petit vecteur latent</li>
                    <li>Décodage : le modèle essaie de reconstruire la séquence originale</li>
                    <li>Erreur MSE : on mesure la différence entre la reconstruction et l'original</li>
                    <li>Seuil : on fixe un seuil (percentile 95 des erreurs sur données normales)</li>
                    <li>Détection : si l'erreur dépasse le seuil, on détecte une anomalie</li>
                </ol>
                <p style="margin: 0.5rem 0; font-size: 0.9rem;">Le modèle ne voit jamais les anomalies pendant l'entraînement : il apprend seulement ce qui est "normal".</p>
            </div>
            """, unsafe_allow_html=True)

        model_path = os.path.join(MODEL_DIR, 'autoencoder.pt')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                st.download_button(
                    "Télécharger le modèle (.pt)",
                    f, file_name="autoencoder.pt",
                    mime="application/octet-stream"
                )


if __name__ == '__main__':
    main()
