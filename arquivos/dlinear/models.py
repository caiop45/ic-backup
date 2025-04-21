import torch
import torch.nn as nn
import torch.optim as optim
import config # Importa as configurações

class DLinear(nn.Module):
    """Modelo DLinear simples."""
    def __init__(self, input_len=config.WINDOW_SIZE, output_dim=1):
        super(DLinear, self).__init__()
        self.linear = nn.Linear(input_len, output_dim)

    def forward(self, x):
        # Garante que a entrada tem a forma correta [batch_size, input_len]
        if x.dim() > 2:
             # Exemplo: Se entrar [batch, seq_len, features] e features=1, remove a última dim
             x = x.squeeze(-1)
        return self.linear(x)

def train_model(model, X_train, y_train, X_val, y_val, epochs, lr=config.LEARNING_RATE, verbose=True):
    """Treina o modelo PyTorch."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10 # Número de épocas para esperar por melhoria antes de parar (early stopping)

    print(f"Iniciando treinamento por {epochs} épocas (lr={lr})...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # Validação
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Opcional: Salvar o melhor modelo
            # torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break

    print("Treinamento concluído.")
    # Opcional: Carregar o melhor modelo
    # model.load_state_dict(torch.load('best_model.pth'))
    return model # Retorna o modelo treinado (o último ou o melhor)