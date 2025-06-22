# === gui_predictor.py ===
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import matplotlib.pyplot as plt

# === Load model and scalers ===
model = joblib.load("xgb_model.pkl")
scaler_x = joblib.load("scaler_x.pickle")
scaler_y = joblib.load("scaler_y.pickle")

CRITICAL_LENGTH = 10.0  # mm
x_min = scaler_x.data_min_
x_max = scaler_x.data_max_

def log_message(msg):
    console_text.config(state=tk.NORMAL)
    console_text.insert(tk.END, msg + "\n")
    console_text.config(state=tk.DISABLED)
    console_text.see(tk.END)

def predict_crack():
    try:
        cycle = float(entry_cycle.get())
        g1 = float(entry_g1.get())
        log_message(f"[DEBUG] Raw input: cycle={cycle}, G1={g1}")

        cycle = np.clip(cycle, x_min[0], x_max[0])
        g1 = np.clip(g1, x_min[1], x_max[1])
        log_message(f"[DEBUG] Clamped input: cycle={cycle}, G1={g1}")

        input_data = np.array([[cycle, g1]])
        scaled_input = scaler_x.transform(input_data)
        log_message(f"[DEBUG] Scaled input: {scaled_input}")

        pred_scaled = model.predict(scaled_input).reshape(-1, 1)
        pred_mm = scaler_y.inverse_transform(pred_scaled)[0][0]
        log_message(f"[DEBUG] Predicted crack length (mm): {pred_mm:.4f}")

        remaining_length = max(0, CRITICAL_LENGTH - pred_mm)

        if pred_mm >= CRITICAL_LENGTH:
            rul_cycles = 0
        else:
            future_cycles = np.arange(cycle, cycle + 30000, 100)
            rul_cycles = None
            for future_cycle in future_cycles:
                sim_input = np.array([[future_cycle, g1]])
                sim_input[:, 0] = np.clip(sim_input[:, 0], x_min[0], x_max[0])
                sim_input[:, 1] = np.clip(sim_input[:, 1], x_min[1], x_max[1])
                sim_scaled = scaler_x.transform(sim_input)
                sim_pred = model.predict(sim_scaled).reshape(-1, 1)
                sim_mm = scaler_y.inverse_transform(sim_pred)[0][0]
                if sim_mm >= CRITICAL_LENGTH:
                    rul_cycles = round(future_cycle - cycle, 2)
                    break
            if rul_cycles is None:
                rul_cycles = ">30000"

        result_label.config(
            text=f"Predicted Crack Length: {pred_mm:.4f} mm\n"
                 f"Remaining to Critical (10 mm): {remaining_length:.4f} mm\n"
                 f"Estimated Remaining Useful Life (Cycles): {rul_cycles}",
            fg="red" if pred_mm >= CRITICAL_LENGTH else "black"
        )

        result_label.pred_length = pred_mm
        result_label.cycle = cycle
        result_label.g1 = g1
        result_label.rul_cycles = rul_cycles

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

def plot_graph():
    if hasattr(result_label, 'pred_length'):
        current_cycle = int(result_label.cycle)
        g1 = result_label.g1
        max_cycles = 30000
        step = 100

        cycles = np.arange(current_cycle, current_cycle + max_cycles, step)
        crack_lengths = []

        for cyc in cycles:
            input_data = np.array([[cyc, g1]])
            input_data[:, 0] = np.clip(input_data[:, 0], x_min[0], x_max[0])
            input_data[:, 1] = np.clip(input_data[:, 1], x_min[1], x_max[1])
            scaled_input = scaler_x.transform(input_data)
            pred_scaled = model.predict(scaled_input).reshape(-1, 1)
            crack_length = scaler_y.inverse_transform(pred_scaled)[0][0]
            crack_lengths.append(crack_length)
            if crack_length >= CRITICAL_LENGTH:
                break

        crack_lengths = np.array(crack_lengths)
        cycles = cycles[:len(crack_lengths)]

        plt.figure(figsize=(10, 6))
        plt.plot(cycles, crack_lengths, label="Predicted Crack Growth", color="blue", linewidth=2)
        plt.axhline(y=CRITICAL_LENGTH, color='red', linestyle='--', label='Critical Length (10 mm)')

        if crack_lengths[-1] >= CRITICAL_LENGTH:
            rul_reached = cycles[-1]
            plt.axvline(x=rul_reached, color='orange', linestyle='--',
                        label=f"RUL = {int(rul_reached - current_cycle)} cycles")
            plt.scatter(rul_reached, CRITICAL_LENGTH, color='orange')

        plt.xlabel("Cycle")
        plt.ylabel("Crack Length (mm)")
        plt.title("Crack Growth Prediction to Failure")
        plt.ylim(min(crack_lengths) - 0.1, CRITICAL_LENGTH + 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        messagebox.showinfo("Notice", "Please perform a prediction first.")

def clear_fields():
    entry_cycle.delete(0, tk.END)
    entry_g1.delete(0, tk.END)
    result_label.config(text="", fg="black")
    console_text.config(state=tk.NORMAL)
    console_text.delete(1.0, tk.END)
    console_text.config(state=tk.DISABLED)

# === GUI Layout ===
root = tk.Tk()
root.title("Aircraft Crack Length Predictor")
root.geometry("700x550")
root.configure(bg="#f0f4f7")

style = ttk.Style()
style.configure("TLabel", font=("Segoe UI", 11))
style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=6)
style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))

header = ttk.Label(root, text="Aircraft Crack Length Predictor", style="Header.TLabel", background="#f0f4f7")
header.pack(pady=15)

input_frame = ttk.Frame(root, padding=10)
input_frame.pack(pady=5)

ttk.Label(input_frame, text="Cycle Number:").grid(row=0, column=0, sticky='e', padx=10, pady=5)
entry_cycle = ttk.Entry(input_frame, width=20)
entry_cycle.grid(row=0, column=1, pady=5)

ttk.Label(input_frame, text="G1 Strain:").grid(row=1, column=0, sticky='e', padx=10, pady=5)
entry_g1 = ttk.Entry(input_frame, width=20)
entry_g1.grid(row=1, column=1, pady=5)

button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

ttk_btn1 = ttk.Button(button_frame, text="Predict", command=predict_crack)
ttk_btn2 = ttk.Button(button_frame, text="Show Crack Graph", command=plot_graph)
ttk_btn3 = ttk.Button(button_frame, text="Clear", command=clear_fields)

for i, btn in enumerate([ttk_btn1, ttk_btn2, ttk_btn3]):
    btn.grid(row=0, column=i, padx=10)

result_label = tk.Label(root, text="", font=("Segoe UI", 11), bg="#f0f4f7", justify="center")
result_label.pack(pady=15)

console_frame = ttk.LabelFrame(root, text="Debug Console", padding=(10, 5))
console_frame.pack(fill="both", expand=True, padx=15, pady=10)

console_text = tk.Text(console_frame, height=8, font=("Courier", 9), bg="#f9f9f9", fg="black", wrap="word")
console_text.pack(fill="both", expand=True)
console_text.config(state=tk.DISABLED)

root.mainloop()
