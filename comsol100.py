import pandas as pd
import matplotlib.pyplot as plt

# === Load and Clean CSV ===
df = pd.read_csv("strain_comsol100.csv")  # Change to your file name
df.columns = df.columns.str.strip()    # Remove whitespace

# === Convert Time to Cycle ===
df['Cycle'] = df['Time'] / 3600
44
# === Convert mm to meters ===
for col in ['u_G1', 'u_G2', 'u_G3', 'u_X0', 'u_X1']:
    df[col] = df[col] / 1000

# === Positions in meters ===
x_G1, x_G2, x_G3 = 0.003, 0.014, 0.025
initial_crack_length = 0.002  # 2 mm

# === Strain at each gauge ===
df['Strain_G1'] = df['u_G1'] / x_G1
df['Strain_G2'] = df['u_G2'] / x_G2
df['Strain_G3'] = df['u_G3'] / x_G3

# === Crack length over time ===
df['Crack_Strain'] = (df['u_X1'] - df['u_X0']) / initial_crack_length
df['Crack_Length'] = initial_crack_length + df['Crack_Strain'].cumsum() * 1e-5

# === Plot Crack Length ===
plt.figure(figsize=(10, 5))
plt.plot(df['Cycle'], df['Crack_Length'], color='purple', label='Crack Length')
plt.xlabel('Cycle')
plt.ylabel('Crack Length (m)')
plt.title('Crack Length vs Cycles')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot Strain at Each Gauge ===
plt.figure(figsize=(10, 5))
plt.plot(df['Cycle'], df['Strain_G1'], label='Strain G1')
plt.plot(df['Cycle'], df['Strain_G2'], label='Strain G2')
plt.plot(df['Cycle'], df['Strain_G3'], label='Strain G3')
plt.xlabel('Cycle')
plt.ylabel('Strain')
plt.title('Strain at Gauges vs Cycles')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Export Final CSV ===
export_df = df[['Cycle', 'Strain_G1', 'Strain_G2', 'Strain_G3', 'Crack_Length']]
export_df.to_csv("comsol_processed_output.csv", index=False)
print(" Exported: comsol_processed_output.csv")

# === Ask for cycle and print crack length ===
try:
    cycle_query = int(input(" Enter a cycle number to check crack length: "))
    closest_row = df.iloc[(df['Cycle'] - cycle_query).abs().idxmin()]
    crack_length = closest_row['Crack_Length']
    print(f" Estimated Crack Length at Cycle {cycle_query}: {crack_length:.6f} m")
except Exception as e:
    print(" Error with input:", e)
