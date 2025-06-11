import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

# -------------------------------------------
# 1. IMPLEMENTAÇÃO DO MODELO DLINEAR
# -------------------------------------------
class DLinearModel(nn.Module):
    """
    Uma implementação simplificada do modelo DLinear.
    Ele usa uma única camada linear para mapear a sequência de entrada para a saída de previsão.
    """
    def __init__(self, input_dim: int, output_dim: int, seq_len: int):
        super(DLinearModel, self).__init__()
        self.seq_len = seq_len
        
        # A camada linear mapeia a entrada achatada (features * tempo) para a saída desejada
        self.linear = nn.Linear(seq_len * input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x entra com shape: [batch_size, seq_len, input_dim]
        
        # 1. Achata a entrada para o formato [batch_size, seq_len * input_dim]
        x = x.view(x.size(0), -1)
        
        # 2. Passa pela camada linear
        x = self.linear(x)
        
        # 3. Adiciona uma dimensão para corresponder ao shape da saída y: [batch_size, 1, output_dim]
        x = x.unsqueeze(1)
        
        return x

# -------------------------------------------
# 2. FUNÇÃO DE TREINAMENTO MODULARIZADA
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
    """
    Função genérica para treinar e validar um modelo PyTorch.

    Args:
        model (nn.Module): A instância do modelo a ser treinado.
        X_train, y_train: Tensores com os dados de treino.
        X_val, y_val: Tensores com os dados de validação.
        epochs (int): Número de épocas para o treino.
        device (torch.device): Dispositivo ('cpu' ou 'cuda') para o treino.
        learning_rate (float): Taxa de aprendizado para o otimizador.
        batch_size (int): Tamanho do lote para o treino.

    Returns:
        dict: Um dicionário com o histórico de perdas de treino e validação.
    """
    
    # Define a função de perda (erro quadrático médio, bom para regressão)
    loss_fn = nn.MSELoss()
    # Define o otimizador (Adam é uma escolha robusta)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Cria DataLoaders para gerenciar os lotes (batches) de forma eficiente
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    print(f"  Iniciando treino com {len(train_dataset)} amostras. Validando com {len(val_dataset)} amostras.")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- TREINO ---
        model.train() # Coloca o modelo em modo de treino
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            # Envia os dados do lote para o dispositivo
            X_batch, y_batch = X_batch.to('cuda:0'), y_batch.to('cuda:0')
            
            optimizer.zero_grad()       # Zera os gradientes
            y_pred = model(X_batch)     # Forward pass: faz a predição
            loss = loss_fn(y_pred, y_batch) # Calcula a perda
            loss.backward()             # Backward pass: calcula os gradientes
            optimizer.step()            # Atualiza os pesos do modelo
            
            epoch_train_loss += loss.item()
        
        # --- VALIDAÇÃO ---
        model.eval() # Coloca o modelo em modo de avaliação
        epoch_val_loss = 0.0
        with torch.no_grad(): # Desabilita o cálculo de gradientes para validação
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to('cuda:0'), y_batch.to('cuda:0')
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                epoch_val_loss += loss.item()

        # Calcula a perda média da época
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        
        # Imprime o progresso a cada 10 épocas ou na última
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(f"    Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_time:.2f}s")
            
    return history