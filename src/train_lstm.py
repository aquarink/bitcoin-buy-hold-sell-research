from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .baselines import CLASS_ORDER


LABEL_TO_INT = {label: idx for idx, label in enumerate(CLASS_ORDER)}
INT_TO_LABEL = {idx: label for label, idx in LABEL_TO_INT.items()}


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, num_classes: int):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(self.dropout(last_hidden))
        return logits


@dataclass
class LSTMArtifacts:
    model: LSTMClassifier
    imputer: SimpleImputer
    scaler: StandardScaler
    sequence_length: int
    validation_macro_f1: float


def encode_labels(labels: pd.Series) -> np.ndarray:
    return labels.map(LABEL_TO_INT).to_numpy()


def decode_labels(values: np.ndarray) -> pd.Series:
    return pd.Series(values).map(INT_TO_LABEL)


def fit_preprocessors(X_train: pd.DataFrame) -> tuple[SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    scaler.fit(X_imp)
    return imputer, scaler


def transform_features(X: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(imputer.transform(X))


def build_sequences(features: np.ndarray, labels: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_seq = []
    y_seq = []
    target_idx = []
    for end_idx in range(seq_len - 1, len(features)):
        if np.isnan(labels[end_idx]):
            continue
        X_seq.append(features[end_idx - seq_len + 1 : end_idx + 1])
        y_seq.append(int(labels[end_idx]))
        target_idx.append(end_idx)
    return np.asarray(X_seq), np.asarray(y_seq), np.asarray(target_idx)


def train_lstm_model(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_valid_seq: np.ndarray,
    y_valid: np.ndarray,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    seed: int = 42,
) -> tuple[LSTMClassifier, float]:
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=len(CLASS_ORDER),
    )
    device = torch.device("cpu")
    model.to(device)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASS_ORDER)), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(SequenceDataset(X_train_seq, y_train), batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(SequenceDataset(X_valid_seq, y_valid), batch_size=batch_size, shuffle=False)

    best_state = None
    best_score = -1.0
    bad_epochs = 0

    for _epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for X_batch, _ in valid_loader:
                logits = model(X_batch.to(device))
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        valid_pred = np.concatenate(preds)
        score = float(f1_score(y_valid, valid_pred, average="macro"))
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def predict_lstm(model: LSTMClassifier, X_seq: np.ndarray, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = torch.device("cpu")
    loader = DataLoader(torch.tensor(X_seq, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    probs = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)
            probs.append(prob)
            preds.append(pred)
    return np.concatenate(preds), np.concatenate(probs)
