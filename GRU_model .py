import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Load datasets
train_df = pd.read_csv("training_set_cleaned.csv")
val_df = pd.read_csv("validation_set_cleaned.csv")
test_df = pd.read_csv("test_set_cleaned.csv")

# Features and target
features = ['G1' , 'cycle'  ]
target =   'crack_length' 

# Scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scale full datasets
X_train_all = scaler_x.fit_transform(train_df[features])
y_train_all = scaler_y.fit_transform(train_df[[target]])
X_val = scaler_x.transform(val_df[features])
y_val = scaler_y.transform(val_df[[target]])
X_test = scaler_x.transform(test_df[features])
y_test = scaler_y.transform(test_df[[target]])

# Convert to torch tensors
def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.float32)

X_val_tensor, y_val_tensor = to_tensor(X_val, y_val)
X_test_tensor, y_test_tensor = to_tensor(X_test, y_test)

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Evaluate MAPE
def evaluate_mape(model, X_tensor, y_tensor):
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)
        preds_np = scaler_y.inverse_transform(preds.numpy())
        targets_np = scaler_y.inverse_transform(y_tensor.numpy())
        return mean_absolute_percentage_error(targets_np, preds_np)

# Train and evaluate GRU for different training sizes
train_sizes = [100, 500, 1000]
results = []

for size in train_sizes:
    # Subset train set
    X_train = X_train_all[:size]
    y_train = y_train_all[:size]
    X_train_tensor, y_train_tensor = to_tensor(X_train, y_train)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

    # New model
    model = GRUModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train for 20 epochs
    for epoch in range(20):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on validation and test sets
    val_mape = evaluate_mape(model, X_val_tensor, y_val_tensor)
    test_mape = evaluate_mape(model, X_test_tensor, y_test_tensor)
    results.append((size, val_mape, test_mape))

# Print table
print(f"\nMAPE (%) Table for GRU")
print(f"{'Train Size':<12} {'Val MAPE (%)':<15} {'Test MAPE (%)':<15}")
for size, val_mape, test_mape in results:
    print(f"{size:<12} {val_mape*100:<15.2f} {test_mape*100:<15.2f}")

