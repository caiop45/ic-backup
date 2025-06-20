import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time

# -------------------------------------------
# 1. IMPLEMENTAÇÃO COMPLETA DO DLINEAR
# -------------------------------------------

class _MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        weight = torch.ones(1, 1, kernel_size) / kernel_size
        self.register_buffer("weight", weight, persistent=False)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        x_padded = F.pad(x, (pad, pad), mode="replicate")
        return F.conv1d(x_padded, self.weight.expand(x.size(1), -1, -1),
                        groups=x.size(1))


class _SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 seq_len: int,
                 kernel_size: int = 25):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        self.decomp = _SeriesDecomposition(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, output_dim)
        self.linear_trend = nn.Linear(seq_len, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, input_dim] -> [B, C, L]
        x = x.permute(0, 2, 1).contiguous()
        seasonal, trend = self.decomp(x)

        B, C, L = seasonal.shape
        seasonal = seasonal.reshape(B * C, L)
        trend = trend.reshape(B * C, L)

        out_seasonal = self.linear_seasonal(seasonal).reshape(B, C, self.output_dim)
        out_trend = self.linear_trend(trend).reshape(B, C, self.output_dim)

        out = out_seasonal + out_trend          # [B, C, output_dim]

        # Se multivariado, devolve média sobre canais;
        # ajuste conforme seu alvo (e.g. selecionar um canal específico)
        if C > 1:
            out = out.mean(dim=1, keepdim=True) # [B, 1, output_dim]

        return out                              # shape compatível c/ y


# -------------------------------------------
# 2. FUNÇÃO DE TREINAMENTO MODULARIZADA
#    (EXATAMENTE a que você já tinha)
# -------------------------------------------

def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    learning_rate: float = 1e-3,
    batch_size: int = 1024,
    patience: int | None = None,
    min_delta: float = 0.0,              # ← novo
    restore_best_weights: bool = True,
):
    device = next(model.parameters()).device
    loss_fn  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False)

    history, best_state = {'train_loss': [], 'val_loss': []}, None
    best_val_loss, wait = float('inf'), 0

    print(f"Iniciando treino com {len(train_loader.dataset)} amostras. "
          f"Validando com {len(val_loader.dataset)} amostras.")

    for epoch in range(epochs):
        t0 = time.time()

        # ---------- TREINO ----------
        model.train(); running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); optimizer.step()
            running += loss.item()
        epoch_train_loss = running / len(train_loader)

        # ---------- VALIDAÇÃO ----------
        model.eval(); running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                running += loss_fn(model(xb), yb).item()
        epoch_val_loss = running / len(val_loader)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)

        # ---------- EARLY-STOP ----------
        if patience is not None:
            if epoch_val_loss < best_val_loss - min_delta:     # ← usa min_delta
                best_val_loss = epoch_val_loss
                best_state    = model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping na época {epoch+1}.")
                    break

        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"Epoch [{epoch+1:>3}/{epochs}] "
                  f"| Train: {epoch_train_loss:.6f} "
                  f"| Val: {epoch_val_loss:.6f} "
                  f"| Δt: {time.time() - t0:.2f}s")

    # ---------- RESTAURA MELHOR PESO ----------
    if restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Pesos restaurados (best val_loss = {best_val_loss:.6f}).")

    return history

# ------------------------------------------------------------------
# EXEMPLO DE USO (mantém sua chamada original)
# ------------------------------------------------------------------
# model = DLinearModel(input_dim=X_train.shape[2],
#                      output_dim=y_train.shape[2],
#                      seq_len=X_train.shape[1]).to('cuda:0')
# history = train_model(model, X_train, y_train, X_val, y_val,
#                       epochs=100, learning_rate=1e-3, batch_size=64)
