import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dl_data_prep import prepare_dl_data, F1Dataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

class F1Predictor(nn.Module):
    def __init__(self, num_features, embedding_info, embedding_dim=8):
        super(F1Predictor, self).__init__()
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim)
            for col, num_embeddings in embedding_info.items()
        ])
        
        total_emb_dim = len(embedding_info) * embedding_dim
        input_dim = num_features + total_emb_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_num, x_cat):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        embs = torch.cat(emb_outs, dim=1)
        x = torch.cat([x_num, embs], dim=1)
        return self.network(x)

def train_model(epochs=100, batch_size=32, lr=0.001):
    # Prepare Data
    (X_num_tr, X_cat_tr, y_tr), (X_num_ts, X_cat_ts, y_ts), info = prepare_dl_data()
    
    train_dataset = F1Dataset(X_num_tr, X_cat_tr, y_tr)
    test_dataset = F1Dataset(X_num_ts, X_cat_ts, y_ts)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize Model
    model = F1Predictor(num_features=X_num_tr.shape[1], embedding_info=info)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_num, x_cat, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_num, x_cat)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")
            
    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_num, x_cat, y in test_loader:
            y_pred = model(x_num, x_cat)
            all_preds.extend(y_pred.numpy())
            all_targets.extend(y.numpy())
            
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    print("\n--- Deep Learning Model Performance ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-Squared: {r2:.2f}")
    
    # Save Model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/f1_dl_model.pth')
    print("Model saved as models/f1_dl_model.pth")

if __name__ == "__main__":
    train_model(epochs=50) # Moderate epochs for initial run
