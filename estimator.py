import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        # output.shape → [batch_size, seq_len, hidden_size]
        # h_n.shape & c_n.shape → [num_layers, batch_size, hidden_size]
        last_output = output[:, -1, :] # dernière sortie temporelle
        last_output = self.dropout(last_output) 
        out = self.fc(last_output)
        return out
    
class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, out_channels=32, dropout=0):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.lstm = LSTMClassifier(input_size=out_channels, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Conv1d attends (batch, input_size, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # sous-echantillonnage seq_len
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        out = self.lstm(x)
        return out
    
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Création masque de padding (True où vecteurs = 0)
        padding_mask = (x.abs().sum(dim=-1) == 0)  # [batch_size, seq_len]

        # Embedding linéaire
        x = self.embedding(x)  # [batch_size, seq_len, d_model]

        # Passage dans le transformer avec masque
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # [batch_size, seq_len, d_model]

        # Moyenne temporelle masquée (ignorer les paddings)
        mask = (~padding_mask).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        x = x * mask  # Masquage embeddings
        x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)  # Moyenne pondérée

        # Classification finale
        return self.fc(x)  # [batch_size, num_classes]
    
def define_model(estimator_name, estimator_params, input_size):
    torch.manual_seed(42) # Seed pour initialisation des poids du modèle uniquement

    input_size=input_size
    num_classes = 6
    
    if estimator_name == 'LSTM':
        hidden_size = estimator_params["hidden_size"]
        num_layers = estimator_params["num_layers"]
        dropout = estimator_params["dropout"]
        hidden_size, num_layers, dropout = estimator_params.values()
        model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout)
    elif estimator_name == 'CNN + LSTM':
        out_channels = estimator_params["out_channels"]
        hidden_size = estimator_params["hidden_size"]
        num_layers = estimator_params["num_layers"]
        dropout = estimator_params["dropout"]
        out_channels, hidden_size, num_layers, dropout = estimator_params.values()
        model = CNNLSTMClassifier(input_size, hidden_size, num_layers, num_classes, out_channels, dropout)
    elif estimator_name == 'Transformer':
        d_model = estimator_params["d_model"]
        nhead = estimator_params["nhead"]
        num_layers = estimator_params["num_layers"]
        dropout = estimator_params["dropout"]
        d_model, nhead, num_layers, dropout = estimator_params.values()
        model = TransformerClassifier(input_size, d_model, nhead, num_layers, num_classes, dropout)
    return model
