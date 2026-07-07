"""
Autoencoder LSTM PyTorch pour la détection d'anomalies
sur séries temporelles industrielles.
"""
import torch
import torch.nn as nn
import numpy as np


class TimeSeriesAutoencoder(nn.Module):
    """
    Autoencoder LSTM pour séries temporelles multivariées.
    Principe : apprend à reconstruire les données normales.
    Anomalie = erreur de reconstruction élevée.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 latent_size: int = 16, num_layers: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        # Encodeur
        self.encoder_lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        # Décodeur
        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.output_fc = nn.Linear(hidden_size, input_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder_lstm(x)
        return self.encoder_fc(hidden[-1])

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        hidden = self.decoder_fc(latent)
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(hidden_repeated)
        return self.output_fc(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent, x.size(1))


class AnomalyDetector:
    """
    Wrapper haut niveau : entraînement, seuil, détection.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 latent_size: int = 16, num_layers: int = 2,
                 sequence_length: int = 30, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.threshold = None
        self.train_losses = []
        self.model = TimeSeriesAutoencoder(
            input_size, hidden_size, latent_size, num_layers
        ).to(self.device)

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        return np.array([
            data[i:i + self.sequence_length]
            for i in range(len(data) - self.sequence_length + 1)
        ])

    def train_model(self, data: np.ndarray, epochs: int = 50,
                    batch_size: int = 32, lr: float = 1e-3,
                    progress_callback=None) -> list:
        sequences = self._create_sequences(data)
        dataset = torch.FloatTensor(sequences).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.model.train()
        self.train_losses = []

        for epoch in range(epochs):
            perm = torch.randperm(dataset.size(0))
            epoch_loss, n = 0.0, 0
            for i in range(0, dataset.size(0), batch_size):
                batch = dataset[perm[i:i + batch_size]]
                optimizer.zero_grad()
                loss = criterion(self.model(batch), batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n += 1
            avg = epoch_loss / max(n, 1)
            self.train_losses.append(avg)
            if progress_callback:
                progress_callback(epoch + 1, epochs, avg)

        return self.train_losses

    def compute_reconstruction_errors(self, data: np.ndarray) -> np.ndarray:
        seqs = torch.FloatTensor(self._create_sequences(data)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(seqs)
        return torch.mean((seqs - recon) ** 2, dim=(1, 2)).cpu().numpy()

    def compute_threshold(self, data: np.ndarray, percentile: float = 95.0) -> float:
        errors = self.compute_reconstruction_errors(data)
        self.threshold = float(np.percentile(errors, percentile))
        return self.threshold

    def detect(self, data: np.ndarray) -> dict:
        if self.threshold is None:
            raise ValueError("Appeler compute_threshold() d'abord.")
        errors = self.compute_reconstruction_errors(data)
        is_anomaly = errors > self.threshold
        return {
            'errors': errors,
            'is_anomaly': is_anomaly,
            'threshold': self.threshold,
            'anomaly_rate': float(is_anomaly.mean() * 100),
            'n_anomalies': int(is_anomaly.sum()),
        }

    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'threshold': self.threshold,
            'sequence_length': self.sequence_length,
            'input_size': self.model.input_size,
            'hidden_size': self.model.hidden_size,
            'latent_size': self.model.latent_size,
            'num_layers': self.model.num_layers,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = None):
        ckpt = torch.load(path, map_location='cpu')
        det = cls(
            input_size=ckpt['input_size'], hidden_size=ckpt['hidden_size'],
            latent_size=ckpt['latent_size'], num_layers=ckpt['num_layers'],
            sequence_length=ckpt['sequence_length'], device=device
        )
        det.model.load_state_dict(ckpt['model_state'])
        det.threshold = ckpt['threshold']
        return det
