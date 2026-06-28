from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ChurnMLP(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@dataclass
class TrainResult:
    model: ChurnMLP
    best_epoch: int
    train_loss: list[float]
    val_loss: list[float]


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    max_epochs: int = 50,
    patience: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    pos_weight: float | None = None,
) -> TrainResult:
    model = ChurnMLP(x_train.shape[1])
    weight = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_state = model.state_dict()
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in _loader(x_train, y_train, batch_size, True):
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
        train_losses.append(float(np.mean(losses)))
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(x_val, dtype=torch.float32)
            yv = torch.tensor(y_val, dtype=torch.float32)
            val_loss = float(loss_fn(model(xv), yv).item())
        val_losses.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return TrainResult(model, best_epoch, train_losses, val_losses)


def predict_proba(model: ChurnMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()
