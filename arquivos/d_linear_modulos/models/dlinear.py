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
    batch_size: int = 64
    ):

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size, shuffle=False)

    history = {'train_loss': [], 'val_loss': []}

    print(f"Iniciando treino com {len(train_loader.dataset)} amostras. "
          f"Validando com {len(val_loader.dataset)} amostras.")

    for epoch in range(epochs):
        start_time = time.time()

        # ---------- TREINO ----------
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to('cuda:0'), yb.to('cuda:0')
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ---------- VALIDAÇÃO ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to('cuda:0'), yb.to('cuda:0')
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))

        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"Epoch [{epoch+1:>3}/{epochs}] "
                  f"| Train: {history['train_loss'][-1]:.6f} "
                  f"| Val: {history['val_loss'][-1]:.6f} "
                  f"| Δt: {time.time() - start_time:.2f}s")

    return history


# ------------------------------------------------------------------
# EXEMPLO DE USO (mantém sua chamada original)
# ------------------------------------------------------------------
# model = DLinearModel(input_dim=X_train.shape[2],
#                      output_dim=y_train.shape[2],
#                      seq_len=X_train.shape[1]).to('cuda:0')
# history = train_model(model, X_train, y_train, X_val, y_val,
#                       epochs=100, learning_rate=1e-3, batch_size=64)
