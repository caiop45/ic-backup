import torch
import torch.nn as nn
import torch.optim as optim

def _vec(t):
    return t.unsqueeze(1) if t.ndim == 1 else t

class DLinear(nn.Module):
    def __init__(self, input_len, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_len, output_dim)

    def forward(self, x):
        return self.linear(x)

def train_model(model, X_train, y_train, X_val, y_val,
                epochs, lr=1e-3, patience=10, min_delta=1e-4):
    crit = nn.MSELoss()
    opt  = optim.Adam(model.parameters(), lr=lr)

    best, wait, state = float("inf"), 0, None
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        crit(model(X_train), _vec(y_train)).backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_val), _vec(y_val)).item()

        if val_loss < best - min_delta:
            best, state, wait = val_loss, model.state_dict(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if state: model.load_state_dict(state)
