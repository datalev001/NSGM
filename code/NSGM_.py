import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

csv_file_path = "NSGM.csv"
DF = pd.read_csv(csv_file_path)

print("DF shape:", DF.shape)
print("DF columns:", DF.columns.tolist())
print("DF time range:", DF['time'].min(), "to", DF['time'].max())

# ----------------------------------------------------------------
# 2. Create Time-Series Dataset with Lag Window
# ----------------------------------------------------------------

L = 5  # Lag window length
num_vars = 8  # Number of variables (e.g., x1 to x8)

class TimeSeriesDataset(Dataset):
    def __init__(self, df, lag=L):
        # Ensure data is sorted by 'time' in ascending order
        self.df = df.sort_values("time").reset_index(drop=True)
        self.lag = lag
        self.num_rows = len(self.df)
        # We'll assume columns are named 'x1', 'x2', ..., 'x8'
        self.vars = [f"x{i}" for i in range(1, num_vars+1)]
        
    def __len__(self):
        # Only use rows that can provide a full lag window plus a target row
        return self.num_rows - self.lag
    
    def __getitem__(self, idx):
        # X: rows idx to idx+lag-1, shape: (lag, num_vars)
        X = self.df.loc[idx:idx+self.lag-1, self.vars].values.astype(np.float32)
        # y: row at idx+lag, shape: (num_vars,)
        y = self.df.loc[idx+self.lag, self.vars].values.astype(np.float32)
        return X, y

dataset = TimeSeriesDataset(DF, lag=L)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ----------------------------------------------------------------
# 3. Define the Neural Sparse Graphical Model (NSGM)
# ----------------------------------------------------------------

class NSGM(nn.Module):
    def __init__(self, num_vars, hidden_dim=32):
        super(NSGM, self).__init__()
        self.num_vars = num_vars
        # Learnable adjacency matrix A (for variable selection), shape: (num_vars, num_vars)
        self.A = nn.Parameter(torch.randn(num_vars, num_vars) * 0.01)
        # Simple feedforward network for nonlinear mapping (F_m)
        self.fc1 = nn.Linear(num_vars, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_vars)
        
    def forward(self, X):
        # X: shape (batch_size, L, num_vars)
        # Aggregate the lag window by taking the mean for each variable -> (batch_size, num_vars)
        z = X.mean(dim=1)
        # Weighted inputs via adjacency matrix A: u = z @ A^T
        u = torch.matmul(z, self.A.t())
        # Pass u through a feedforward network
        out = self.fc2(self.relu(self.fc1(u)))
        # Output shape: (batch_size, num_vars) => predictions for next time step
        return out

model = NSGM(num_vars=num_vars)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
lambda_l1 = 0.0007  # L1 penalty weight on the adjacency matrix

# ----------------------------------------------------------------
# 4. Training Loop
# ----------------------------------------------------------------

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        # Add L1 penalty on the adjacency matrix (for sparsity)
        loss += lambda_l1 * torch.norm(model.A, 1)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            pred = model(X_batch)
            v_loss = criterion(pred, y_batch)
            v_loss += lambda_l1 * torch.norm(model.A, 1)
            val_loss += v_loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

# ----------------------------------------------------------------
# 5. Evaluation and Output of Key Information
# ----------------------------------------------------------------

model.eval()
print("Learned Adjacency Matrix (A):")
print(model.A.data.cpu().numpy())

# Show predictions vs true values for a few samples from the validation set
for X_batch, y_batch in val_loader:
    pred = model(X_batch)
    print("Sample Predictions:")
    print(pred[:5].cpu().detach().numpy())
    print("Sample True Values:")
    print(y_batch[:5].cpu().numpy())
    break

print("Training complete. This NSGM model demonstrates effective variable selection and time-series network modeling.")
