import pandas as pd
import numpy as np
import pyttsx3
import threading
import datetime
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
import os

safety_instructions = {
    "CRITICAL": "Evacuate immediately. Call emergency services. Do not use elevators. Stay low to avoid smoke.",
    "HIGH": "Stay alert. Prepare to evacuate. Turn off electrical appliances. Keep fire extinguishers ready.",
    "MEDIUM": "Monitor the situation closely. Avoid open flames or sparks. Ensure proper ventilation.",
    "LOW": "No immediate danger. Stay cautious and continue monitoring conditions."
}


def tts_speak(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)   
        engine.setProperty("volume", 5) 
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()


print("🔄 Training model...")

df = pd.read_csv("smoke_detection_iot.csv")  

target_col = "Fire Alarm"
features = [
    "Temperature[C]", "Humidity[%]", "TVOC[ppb]", "eCO2[ppm]",
    "Raw H2", "Raw Ethanol", "Pressure[hPa]", "PM1.0", "PM2.5",
    "NC0.5", "NC1.0", "NC2.5", "CNT"
]

X = df[features].astype(float)
y = df[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_s, y_train)

print("✅ Model trained successfully!")
print("Accuracy:", accuracy_score(y_test, rf.predict(X_test_s)))
print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test_s)[:, 1]))


def predict_and_warn(input_dict):
    x = np.array([input_dict[f] for f in features], dtype=float).reshape(1, -1)
    x_s = scaler.transform(x)

    prob = rf.predict_proba(x_s)[0][1]
    pred = rf.predict(x_s)[0]

    if prob >= 0.75:
        level, emoji = "CRITICAL", "🔥🔥"
    elif prob >= 0.5:
        level, emoji = "HIGH", "🔥"
    elif prob >= 0.25:
        level, emoji = "MEDIUM", "⚠️"
    else:
        level, emoji = "LOW", "✅"

    msg = f"PREDICTION: {'FIRE RISK' if pred == 1 else 'NO FIRE'}\n" \
          f"Probability: {prob*100:.1f}%\n" \
          f"Level: {level} {emoji}"
    
    speech_text = f"The system predicts {level} fire risk. The probability is {prob*100:.1f} percent."
    tts_speak(speech_text)

    if level in safety_instructions:
        instr = safety_instructions[level]
        tts_speak(f"Safety instructions: {instr}")  
        msg += f"\n\nSafety Instructions:\n{instr}" 

    if level in ["HIGH", "CRITICAL"]:
        messagebox.showwarning("🔥 FIRE RISK ALERT 🚨", msg)
    else:
        messagebox.showinfo("Fire Risk Status", msg)

    # Emergency Instructions Window
    if level in ["HIGH", "CRITICAL"]:
        instructions = (
            "🚨 EMERGENCY INSTRUCTIONS 🚨\n\n"
            "- Stay calm but act quickly.\n"
            "- Evacuate immediately to a safe area.\n"
            "- Do not use elevators.\n"
            "- If smoke is present, stay low to the ground.\n"
            "- Call emergency services or alert local authorities.\n"
            "- Use fire extinguishers only if trained and safe to do so."
        )
        safety_win = tk.Toplevel(root)
        safety_win.title("🔥 Emergency Safety Instructions")
        safety_win.geometry("400x300")
        safety_win.config(bg="#FCE5CD")
        tk.Label(
            safety_win,
            text=instructions,
            justify="left",
            wraplength=380,
            font=("Arial", 11, "bold"),
            bg="#FCE5CD",
            fg="#4B0101"
        ).pack(padx=10, pady=10)

    # --- Log to CSV ---
    log_file = "fire_predictions_log.csv"
    log_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["DateTime"] + features + ["Predicted", "Probability", "Level"])
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            *[input_dict[f] for f in features],
            "FIRE" if pred == 1 else "NO FIRE",
            f"{prob*100:.1f}%",
            level
        ])

root = tk.Tk()
root.title("🔥 Pre-Fire Detection System 🚒")
root.geometry("450x750")
root.config(bg="#E3E3B5") 

entries = {}

tk.Label(
    root, 
    text="🌤️ Enter Atmospheric Conditions 🌡️", 
    font=("Times New Roman", 16, "bold"), 
    bg="#F3CBA5", 
    fg="#302929"
).pack(pady=15)

form_frame = tk.Frame(root, bg="#F5F5DC")
form_frame.pack(pady=5)

# Entry fields
for f in features:
    row = tk.Frame(form_frame, bg="#F5F5DC")
    lab = tk.Label(row, text=f"📝 {f}", anchor="w", width=20, bg="#F5F5DC", fg="#11041A", font=("georgia", 11, "bold"))
    ent = tk.Entry(row, width=15, font=("georgia", 11))
    row.pack(pady=4)
    lab.pack(side=tk.LEFT)
    ent.pack(side=tk.LEFT, padx=5)
    entries[f] = ent


def take_voice_input_all():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        for f in features:
            messagebox.showinfo("Voice Input", f"🎙️ Speak the value for {f}")
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                entries[f].delete(0, tk.END)
                entries[f].insert(0, text)
            except sr.WaitTimeoutError:
                messagebox.showerror("Timeout", "⏳ No voice detected. Please try again.")
            except sr.UnknownValueError:
                messagebox.showerror("Error", "❌ Could not understand your speech.")
            except sr.RequestError:
                messagebox.showerror("Error", "🌐 Speech recognition service unavailable.")

def on_submit():
    try:
        user_input = {f: float(entries[f].get()) for f in features}
        predict_and_warn(user_input)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

submit_btn = tk.Button(
    root, 
    text="🔥 Predict Fire Risk 🚨", 
    command=on_submit, 
    font=("Times New Roman", 14, "bold"),  
    bg="#2D200E", 
    fg="white",    
    activebackground="#24190A",  
    padx=10,
    pady=5
)
submit_btn.pack(pady=10)

voice_btn = tk.Button(
    root,
    text="🎙️ Speak All Inputs",
    command=take_voice_input_all,
    font=("georgia", 12, "bold"),
    bg="#A87F5A",
    fg="white",
    padx=8,
    pady=5
)
voice_btn.pack(pady=5)

def view_pie_chart():
    try:
        user_input = {f: float(entries[f].get()) for f in features}
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields to view chart.")
        return

    values = list(user_input.values())
    plt.figure(figsize=(7,7))
    plt.pie(
    values,
    labels=features,
    autopct="%1.1f%%",
    startangle=140,
    colors=plt.cm.tab20.colors,
    textprops={'fontname': 'Garamond', 'fontsize': 11, 'fontweight': 'bold'}
)

    plt.title(
    "Contributing Factors to Fire Risk",
    fontdict={'fontname': 'Garamond', 'fontsize': 16, 'fontweight': 'bold', 'underline': True},
    pad=20
)

    plt.tight_layout()
    plt.show()

pie_btn = tk.Button(
    root,
    text="📊 View Feature Contribution",
    command=view_pie_chart,
    font=("Times New Roman", 12, "bold"),
    bg="#6C757D",
    fg="white",
    padx=8,
    pady=4
)
pie_btn.pack(pady=10)

root.mainloop()