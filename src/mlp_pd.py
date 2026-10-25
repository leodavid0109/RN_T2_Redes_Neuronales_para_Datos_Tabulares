"""MLP binario de PD y utilidades de entrenamiento para datos ya preprocesados."""
from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ACTIVACIONES = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
OPTIMIZADORES = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


class MLPBinario(nn.Module):
    """Perceptrón multicapa que devuelve un logit; sigmoid(logit) es P(clase 1)."""

    def __init__(self, input_dim: int, config: dict[str, Any]):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim debe ser positivo")
        if config["activation"] not in ACTIVACIONES:
            raise ValueError(f"Activación no soportada: {config['activation']}")
        if config["optimizer"] not in OPTIMIZADORES:
            raise ValueError(f"Optimizador no soportado: {config['optimizer']}")
        if not 0.0 <= float(config["dropout"]) < 1.0:
            raise ValueError("dropout debe estar en [0, 1)")

        modules: list[nn.Module] = []
        width = input_dim
        for hidden in config["hidden_layers"]:
            hidden = int(hidden)
            if hidden < 1:
                raise ValueError("Cada capa oculta debe tener al menos una neurona")
            modules.extend([nn.Linear(width, hidden), ACTIVACIONES[config["activation"]]()])
            if float(config["dropout"]) > 0:
                modules.append(nn.Dropout(float(config["dropout"])))
            width = hidden
        modules.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def set_seed(seed: int) -> None:
    """Controla inicialización, muestreo de lotes y fuentes aleatorias comunes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numero_parametros(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _logits(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            chunks.append(model(xb).detach().cpu())
    return torch.cat(chunks).numpy()


def probabilidades_clase_1(model: nn.Module, x: np.ndarray, batch_size: int = 4096,
                           device: torch.device | None = None) -> np.ndarray:
    """Devuelve sigmoid(logit), es decir, la probabilidad estimada de incumplimiento (y=1)."""
    device = device or torch.device("cpu")
    z = _logits(model, np.asarray(x, dtype=np.float32), batch_size, device)
    p = torch.sigmoid(torch.from_numpy(z)).numpy().astype(float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise FloatingPointError("El MLP produjo probabilidades inválidas")
    return p


def entrenar_una_corrida(x_train: np.ndarray, y_train: np.ndarray,
                         x_valid: np.ndarray, y_valid: np.ndarray,
                         model_config: dict[str, Any], *, seed: int, batch_size: int,
                         max_epochs: int, patience: int, min_delta: float,
                         num_threads: int = 2, pos_weight: float = 1.0) -> dict[str, Any]:
    """Entrena con BCE de logits, peso positivo opcional y parada por log-loss sin ponderar."""
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError("X e y deben tener el mismo número de filas")
    if set(np.unique(y_train)).difference({0, 1}) or set(np.unique(y_valid)).difference({0, 1}):
        raise ValueError("El objetivo debe contener las etiquetas binarias 0 y 1")
    if not len(x_train) or not len(x_valid):
        raise ValueError("Entrenamiento y validación deben tener filas")
    if not np.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError("pos_weight debe ser un número finito positivo")

    torch.set_num_threads(num_threads)
    set_seed(seed)
    device = torch.device("cpu")
    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    x_valid = np.ascontiguousarray(x_valid, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_valid = np.asarray(y_valid, dtype=np.float32)

    model = MLPBinario(x_train.shape[1], model_config).to(device)
    params = numero_parametros(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    optimizer = OPTIMIZADORES[model_config["optimizer"]](
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
        **({"momentum": 0.9} if model_config["optimizer"] == "sgd" else {}))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=device))
    # La parada se evalúa en log-loss ordinaria: las probabilidades siguen representando PD.
    val_loss_fn = nn.BCEWithLogitsLoss()
    yv_tensor = torch.from_numpy(y_valid).to(device)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_logits = torch.from_numpy(_logits(model, x_valid, batch_size * 4, device)).to(device)
        val_loss = float(val_loss_fn(val_logits, yv_tensor).item())
        history.append({"epoch": epoch, "train_loss_last_batch": float(loss.item()), "val_logloss": val_loss})
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    elapsed = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("No se guardó ningún checkpoint de validación")
    model.load_state_dict(best_state)
    return {"model": model, "state_dict": best_state, "best_epoch": best_epoch,
            "best_validation_logloss": best_loss, "epochs_run": len(history),
            "history": history, "parameters": params, "training_seconds": elapsed,
            "seed": seed, "device": str(device)}
