"""MLP de regresión para LGD continua en [0, 1]."""
from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


class MLPRegresorLGD(nn.Module):
    """Devuelve LGD esperada con sigmoid, estrictamente acotada entre cero y uno."""

    def __init__(self, input_dim: int, config: dict[str, Any]):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim debe ser positivo")
        if config["activation"] not in ACTIVATIONS:
            raise ValueError(f"Activación no soportada: {config['activation']}")
        if config["optimizer"] not in OPTIMIZERS:
            raise ValueError(f"Optimizador no soportado: {config['optimizer']}")
        dropout = float(config["dropout"])
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout debe estar en [0, 1)")

        layers: list[nn.Module] = []
        width = input_dim
        for hidden in config["hidden_layers"]:
            hidden = int(hidden)
            if hidden < 1:
                raise ValueError("Cada capa oculta debe tener al menos una neurona")
            layers.extend([nn.Linear(width, hidden), ACTIVATIONS[config["activation"]]()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden
        layers.extend([nn.Linear(width, 1), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numero_parametros(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _predict(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to("cpu")
            chunks.append(model(xb).detach().cpu())
    pred = torch.cat(chunks).numpy().astype(float)
    if not np.isfinite(pred).all() or np.any((pred < 0.0) | (pred > 1.0)):
        raise FloatingPointError("El MLP produjo LGD fuera de [0, 1]")
    return pred


def predecir_lgd(model: nn.Module, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    return _predict(model, np.ascontiguousarray(x, dtype=np.float32), batch_size)


def entrenar_una_corrida(x_train: np.ndarray, y_train: np.ndarray,
                         x_valid: np.ndarray, y_valid: np.ndarray,
                         model_config: dict[str, Any], *, seed: int, batch_size: int,
                         max_epochs: int, patience: int, min_delta: float,
                         num_threads: int = 2,
                         output_initialization: str = "pytorch") -> dict[str, Any]:
    """Entrena con MSE y restaura el checkpoint con menor RMSE de validación."""
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError("X e y deben tener el mismo número de filas")
    if not len(x_train) or not len(x_valid):
        raise ValueError("Entrenamiento y validación deben tener filas")
    if not np.isfinite(y_train).all() or not np.isfinite(y_valid).all():
        raise ValueError("LGD contiene valores no finitos")
    if np.any((y_train < 0) | (y_train > 1)) or np.any((y_valid < 0) | (y_valid > 1)):
        raise ValueError("El objetivo LGD debe pertenecer a [0, 1]")

    torch.set_num_threads(num_threads)
    set_seed(seed)
    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    x_valid = np.ascontiguousarray(x_valid, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_valid = np.asarray(y_valid, dtype=np.float32)
    model = MLPRegresorLGD(x_train.shape[1], model_config).to("cpu")
    if output_initialization not in {"pytorch", "mean_train"}:
        raise ValueError("output_initialization debe ser 'pytorch' o 'mean_train'")
    if output_initialization == "mean_train":
        # Cero en W y logit(media_train) en b hace constante la salida inicial.
        output_layer = next(layer for layer in reversed(model.network) if isinstance(layer, nn.Linear))
        mean_train = float(np.mean(y_train))
        clipped_mean = float(np.clip(mean_train, 1e-6, 1.0 - 1e-6))
        with torch.no_grad():
            output_layer.weight.zero_()
            output_layer.bias.fill_(float(np.log(clipped_mean / (1.0 - clipped_mean))))
    params = numero_parametros(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True, generator=generator, num_workers=0,
    )
    optimizer = OPTIMIZERS[model_config["optimizer"]](
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
        **({"momentum": 0.9} if model_config["optimizer"] == "sgd" else {}),
    )
    loss_fn = nn.MSELoss()
    yv = torch.from_numpy(y_valid)
    best_rmse, best_state, best_epoch = float("inf"), None, 0
    patience_reference = float("inf")
    stale, history = 0, []
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().item()))
        valid_pred = _predict(model, x_valid, batch_size * 4)
        valid_rmse = float(np.sqrt(np.mean((valid_pred - y_valid) ** 2)))
        history.append({"epoch": epoch, "train_mse_mean": float(np.mean(batch_losses)),
                        "valid_rmse": valid_rmse})
        # El checkpoint siempre conserva el RMSE observado mínimo; min_delta solo
        # decide si un avance es bastante grande para reiniciar la paciencia.
        if valid_rmse < best_rmse:
            best_rmse, best_epoch = valid_rmse, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if valid_rmse < patience_reference - min_delta:
            patience_reference = valid_rmse
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No se guardó un checkpoint de validación")
    model.load_state_dict(best_state)
    stop_reason = "early_stopping_patience" if len(history) < max_epochs else "max_epochs_reached"
    return {"model": model, "state_dict": best_state, "best_epoch": best_epoch,
            "best_validation_rmse": best_rmse, "epochs_run": len(history),
            "history": history, "parameters": params, "training_seconds": elapsed,
            "seed": seed, "device": "cpu", "stop_reason": stop_reason,
            "output_initialization": output_initialization}
