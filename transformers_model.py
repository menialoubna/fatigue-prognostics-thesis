import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Hyperparameters
sequence_length = 10
batch_size = 32
epochs = 50
learning_rate = 0.001

train_df = pd.read_excel("train_interp.xlsx")
val_df = pd.read_excel("train_interp.xlsx")
test_df = pd.read_excel("train_interp.xlsx")

features = ['cycle3']
target =   'crack_length3'

# Scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_all = scaler_x.fit_transform(train_df[features])
y_all = scaler_y.fit_transform(train_df[[target]]).ravel()

X_val = scaler_x.transform(val_df[features])
y_val = scaler_y.transform(val_df[[target]]).ravel()
X_test = scaler_x.transform(test_df[features])
y_test = scaler_y.transform(test_df[[target]]).ravel()

# Helper to create sequences
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.output_proj(x[:, -1, :])  # Use last token output

# Train & eval loop
def train_and_evaluate(X_train_raw, y_train_raw, X_val, y_val, X_test, y_test):
    X_train, y_train = create_sequences(X_train_raw, y_train_raw, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), torch.tensor(y_test_seq, dtype=torch.float32))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    model = TransformerModel(input_size=X_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    def evaluate(dl, y_true_raw):
        preds = []
        with torch.no_grad():
            for xb, _ in dl:
                pred = model(xb).squeeze().numpy()
                preds.extend(pred)
        y_pred = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1))
        y_true = scaler_y.inverse_transform(y_true_raw[sequence_length:].reshape(-1, 1))
        return mean_absolute_percentage_error(y_true, y_pred)

    val_mape = evaluate(val_dl, y_val)
    test_mape = evaluate(test_dl, y_test)
    return val_mape, test_mape

# Evaluate across train sizes
train_sizes = [100, 500, 1000]
results = []

for size in train_sizes:
    val_mape, test_mape = train_and_evaluate(X_all[:size], y_all[:size], X_val, y_val, X_test, y_test)
    results.append((size, val_mape * 100, test_mape * 100))

# Print results
print("\nMAPE (%) Table for Transformer")
print(f"{'Train Size':<12} {'Val MAPE (%)':<15} {'Test MAPE (%)':<15}")
for size, val_mape, test_mape in results:
    print(f"{size:<12} {val_mape:<15.2f} {test_mape:<15.2f}")
