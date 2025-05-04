import pandas as pd
import matplotlib.pyplot as plt

# === Load the CSV file ===
df = pd.read_csv("strain_comsol_ALF.csv")
df.columns = df.columns.str.strip()  # Clean column names

# === Convert Time to Cycles ===
df['Cycle'] = df['Time'] / 3600

# === Convert displacements from mm to meters ===
for col in ['u_G1', 'u_G2', 'u_G3', 'u_X0', 'u_X1']:
    df[col] = df[col] / 1000

# === Extract strain (assumed already calculated in COMSOL) ===
# Rename for clarity if needed
df = df.rename(columns={
    'Strain_G1': 'Strain_G1',
    'Strain_G2': 'Strain_G2',
    'Strain_G3': 'Strain_G3'
})

# === Crack Length Estimation ===
initial_crack_length = 0.002  # 2 mm in meters
df['Crack_Strain'] = (df['u_X1'] - df['u_X0']) / initial_crack_length
df['Crack_Length'] = initial_crack_length + df['Crack_Strain'].cumsum() * 1e-5  # Adjust if needed

# === Plot: Crack Length vs Cycles ===
plt.figure(figsize=(10, 5))
plt.plot(df['Cycle'], df['Crack_Length'], color='purple', label='Crack Length')
plt.xlabel('Cycle')
plt.ylabel('Crack Length (m)')
plt.title('Crack Length vs Cycles')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot: Strain at Each Gauge ===
plt.figure(figsize=(10, 5))
plt.plot(df['Cycle'], df['Strain_G1'], label='Strain at G1')
plt.plot(df['Cycle'], df['Strain_G2'], label='Strain at G2')
plt.plot(df['Cycle'], df['Strain_G3'], label='Strain at G3')
plt.xlabel('Cycle')
plt.ylabel('Strain (dimensionless)')
plt.title('Strain at Each Gauge vs Cycles')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Export Processed Data ===
export_df = df[['Cycle', 'Strain_G1', 'Strain_G2', 'Strain_G3', 'Crack_Length']]
export_df.to_csv("comsol_final_output.csv", index=False)
print("✅ Exported: comsol_final_output.csv")

# === Ask User for Cycle and Show Crack Length ===
try:
    cycle_query = int(input("🔍 Enter a cycle number to check crack length: "))
    closest_row = df.iloc[(df['Cycle'] - cycle_query).abs().idxmin()]
    crack_length = closest_row['Crack_Length']
    print(f"🔩 Estimated Crack Length at Cycle {cycle_query}: {crack_length:.6f} m")
except Exception as e:
    print("⚠️ Input Error:", e)
