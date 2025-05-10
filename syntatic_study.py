import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import ast

# === Load your training CSV ===
df = pd.read_csv('training_set2.csv')  # your provided file
df.columns = df.columns.str.strip()

# === Extract fields properly ===
a_0 = float(df['a_0'].values[0])              
C = float(df['C'].values[0])
m = float(df['m'].values[0])
strains_raw = df['strains'].values[0]

# === Extract crack lengths list ===
crack_length_list = ast.literal_eval(df['crack_lengths'].values[0])
crack_length_array = np.array(crack_length_list)
nb_samples = len(crack_length_array)  # Real number of samples (122)

# === Clean and extract strains ===
cleaned_str = re.sub(r'\s+', ',', strains_raw.strip())
cleaned_str = cleaned_str.replace('[,', '[')
strains_array = np.array(eval(cleaned_str))

# === Create Real Cycles and Cycles Left ===
total_cycles_real = int(df['nb_cycles'].values[0])  # 60500 from your file
cycles = np.linspace(0, total_cycles_real, nb_samples, endpoint=False).astype(int)
cycles_left = total_cycles_real - cycles

# === Check dimensions match ===
if strains_array.ndim == 1:
    strains_array = strains_array.reshape(-1, 1)  # Force 2D for consistency

assert len(cycles) == strains_array.shape[0] == crack_length_array.shape[0], "Mismatch detected! Check data dimensions!"

# === Build the new DataFrame ===
data_formatted = pd.DataFrame({
    'Cycle': cycles,
    'Strain_G1': strains_array[:, 0],
    'Strain_G2': strains_array[:, 1],
    'Strain_G3': strains_array[:, 2],
    'Crack_Length': crack_length_array,
    'Cycles_Left': cycles_left
})

# === Save the clean CSV ===
data_formatted.to_csv('training_set_clean_ready_realcycles.csv', index=False)
print("✅ Saved: training_set_clean_ready_realcycles.csv")

# === Plot Crack Length Growth ===
plt.figure(figsize=(10,5))
plt.plot(data_formatted['Cycle'], data_formatted['Crack_Length'], label='Crack Length (m)', color='purple')
plt.xlabel('Cycle')
plt.ylabel('Crack Length (m)')
plt.title('Crack Length Growth over Real Cycles')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Plot Strain at Gauges ===
plt.figure(figsize=(10,5))
plt.plot(data_formatted['Cycle'], data_formatted['Strain_G1'], label='Strain G1')
plt.plot(data_formatted['Cycle'], data_formatted['Strain_G2'], label='Strain G2')
plt.plot(data_formatted['Cycle'], data_formatted['Strain_G3'], label='Strain G3')
plt.xlabel('Cycle')
plt.ylabel('Strain')
plt.title('Strain at Gauges vs Real Cycles')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Optional: Ask user a cycle and show crack length ===
try:
    user_cycle = int(input("🔎 Enter a cycle number to check crack length: "))
    
    if user_cycle not in data_formatted['Cycle'].values:
        print("⚠️ Cycle not available in the dataset.")
    else:
        nearest_cycle = data_formatted[data_formatted['Cycle'] == user_cycle]

        crack = nearest_cycle['Crack_Length'].values[0]
        left = nearest_cycle['Cycles_Left'].values[0]
        cycle_found = nearest_cycle['Cycle'].values[0]

        print(f"📈 Closest data point: cycle {cycle_found}")
        print(f"📈 Crack length = {crack:.6f} meters")
        print(f"🕐 Remaining cycles until failure = {left}")
except Exception as e:
    print("⚠️ Error:", e)





