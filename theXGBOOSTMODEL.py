import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib

# === Load interpolated datasets ===
train_df = pd.read_excel("train3_adjusted_final.xlsx")
val_df = pd.read_excel("train3_adjusted_final.xlsx")
test_df = pd.read_excel("train3_adjusted_final.xlsx")

features = ["cycle" , "G1"]
target = "crack length"

# === Scale full val/test sets (used across all sizes) ===
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_val = val_df[features].values
X_test = test_df[features].values
y_val = val_df[[target]].values
y_test = test_df[[target]].values

X_val_scaled = scaler_x.fit_transform(X_val)
X_test_scaled = scaler_x.transform(X_test)
y_val_scaled = scaler_y.fit_transform(y_val).ravel()
y_test_scaled = scaler_y.transform(y_test).ravel()

# === Evaluate function ===
def eval_and_log(X, y_true, label):
    y_pred_scaled = model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()
    mape = mean_absolute_percentage_error(y_true_inv, y_pred) * 100
    return mape

# === Store results for table display ===
results = []

# === Try different training sizes ===
for size in [100, 500, 1000]:
    subset_df = train_df.iloc[:size]
    X_train = subset_df[features].values
    y_train = subset_df[[target]].values

    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train).ravel()

    # Train model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=5, learning_rate=0.05)
    model.fit(X_train_scaled, y_train_scaled)

    # Evaluate
    val_mape = eval_and_log(X_val_scaled, y_val_scaled, "Validation")
    test_mape = eval_and_log(X_test_scaled, y_test_scaled, "Test")

    results.append((size, val_mape, test_mape))

# === Final model (with GridSearch) ===
print("\n--- Final Full Model Training (1000 samples with GridSearch) ---")

X_train = train_df[features].values
y_train = train_df[[target]].values
X_train_scaled = scaler_x.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train).ravel()

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.8, 1.0],
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_scaled, y_train_scaled)
best_model = grid_search.best_estimator_

# Evaluate full model
def eval_and_log_final(X, y_true, label):
    y_pred_scaled = best_model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()
    mape = mean_absolute_percentage_error(y_true_inv, y_pred) * 100
    print(f"{label} MAPE: {mape:.2f}%")
    return mape, y_pred

final_val_mape, _ = eval_and_log_final(X_val_scaled, y_val_scaled, "Validation (Full Model)")
final_test_mape, _ = eval_and_log_final(X_test_scaled, y_test_scaled, "Test (Full Model)")

results.append(("1000 + GridSearch", final_val_mape, final_test_mape))

# === Print results as table ===
print("\n=== MAPE Summary Table ===")
print(f"{'Training Size':<20} {'Validation MAPE (%)':<22} {'Test MAPE (%)':<15}")
print("-" * 60)
for size, val, test in results:
    print(f"{str(size):<20} {val:<22.2f} {test:<15.2f}")

# === Save final model and scalers ===
joblib.dump(best_model, "xgb_model.pkl")
joblib.dump(scaler_x, "scaler_x.pickle")
joblib.dump(scaler_y, "scaler_y.pickle")

print("\nFinal model and scalers saved.")
