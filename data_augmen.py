import pandas as pd
import numpy as np

def augment_data(df, target_size):
    df = df.sort_values("cycle3").reset_index(drop=True)
    current_size = len(df)
    
    if current_size >= target_size:
        return df.iloc[:target_size].copy()

    # Interpolate between real rows
    new_rows = []
    needed = target_size - current_size

    for _ in range(needed):
        i = np.random.randint(0, current_size - 1)
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]

        alpha = np.random.rand()
        cycle = int(round(alpha * row1["cycle3"] + (1 - alpha) * row2["cycle3"]))
        crack = alpha * row1["crack_length3"] + (1 - alpha) * row2["crack_length3"]
        g1 = alpha * row1["G1_2"] + (1 - alpha) * row2["G1_2"]

        new_rows.append({"cycle3": cycle, "crack_length3": crack, "G1_2": g1})

    aug_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    aug_df["cycle3"] = aug_df["cycle3"].round().astype(int)  # ensure integer cycles
    aug_df = aug_df.sort_values("cycle3").reset_index(drop=True)

    return aug_df

# Load files
train_df = pd.read_excel("train3.xlsx")
val_df = pd.read_excel("val3.xlsx")
test_df = pd.read_excel("test3.xlsx")

# Apply augmentation
aug_train = augment_data(train_df, 1000)
aug_val = augment_data(val_df, 100)
aug_test = augment_data(test_df, 100)

# Save results
aug_train.to_excel("aug_train3.xlsx", index=False)
aug_val.to_excel("aug_val3.xlsx", index=False)
aug_test.to_excel("aug_test3.xlsx", index=False)

print("✅ Augmented datasets saved: aug_train3.xlsx, aug_val3.xlsx, aug_test3.xlsx")

