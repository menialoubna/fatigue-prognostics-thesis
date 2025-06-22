import pandas as pd
import numpy as np

# === Load and preview datasets ===
def safe_load(path):
    df = pd.read_excel(path)
    print(f"{path} loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")
    return df

train_df = safe_load("aug_train3.xlsx")
val_df = safe_load("aug_val3.xlsx")
test_df = safe_load("aug_test3.xlsx")

# === Combine and standardize column names ===
combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
combined_df.columns = [col.strip().lower() for col in combined_df.columns]

# Rename 'g1_2' to 'g1' for consistency
if 'g1_2' not in combined_df.columns:
    raise ValueError("❌ 'g1_2' column not found in the data.")
combined_df = combined_df.rename(columns={'g1_2': 'g1'})

# Force 'cycle3' to numeric and clean it
combined_df['cycle3'] = pd.to_numeric(combined_df['cycle3'], errors='coerce')
combined_df.dropna(subset=['cycle3'], inplace=True)
combined_df['cycle3'] = combined_df['cycle3'].astype(int)

# Keep necessary columns and drop duplicates
combined_df = combined_df[['cycle3', 'crack_length3', 'g1']]
combined_df = combined_df.drop_duplicates(subset='cycle3')

# === Define full cycle range ===
full_cycles = np.arange(100, 72001)
base_df = pd.DataFrame({'cycle3': full_cycles})

# === Merge and interpolate ===
merged_df = pd.merge(base_df, combined_df, on='cycle3', how='left')
merged_df['crack_length3'] = merged_df['crack_length3'].interpolate(method='linear', limit_direction='both')
merged_df['g1'] = merged_df['g1'].interpolate(method='linear', limit_direction='both')
merged_df.dropna(inplace=True)
merged_df.reset_index(drop=True, inplace=True)

# === Create datasets with fixed start points and lengths ===
train_start = 1000
val_start = 1100
test_start = 1300

train_step = (72000 - train_start) // 999
val_step = (72000 - val_start) // 99
test_step = (72000 - test_start) // 99

train_cycles = np.arange(train_start, 72001, train_step)[:1000]
val_cycles = np.arange(val_start, 72001, val_step)[:100]
test_cycles = np.arange(test_start, 72001, test_step)[:100]

train_df_final = merged_df[merged_df['cycle3'].isin(train_cycles)].reset_index(drop=True)
val_df_final = merged_df[merged_df['cycle3'].isin(val_cycles)].reset_index(drop=True)
test_df_final = merged_df[merged_df['cycle3'].isin(test_cycles)].reset_index(drop=True)

# === Rename back to original column name for saving ===
train_df_final = train_df_final.rename(columns={'g1': 'G1_2'})
val_df_final = val_df_final.rename(columns={'g1': 'G1_2'})
test_df_final = test_df_final.rename(columns={'g1': 'G1_2'})

# === Save output ===
train_df_final.to_excel("train_interp3.xlsx", index=False)
val_df_final.to_excel("val_interp3.xlsx", index=False)
test_df_final.to_excel("test_interp3.xlsx", index=False)

print("✅ Interpolation complete! Files saved as train_interp.xlsx, val_interp.xlsx, test_interp.xlsx")
