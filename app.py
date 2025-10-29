import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from gtts import gTTS
from io import BytesIO

st.set_page_config(page_title="FireCast", page_icon="🚒", layout="centered")

st.title("🔥 FireCast: Pre-Fire Detection System")
st.markdown("### Predict Fire Risks based on Atmospheric Conditions")

safety_instructions = {
    "CRITICAL": "Evacuate immediately. Call emergency services. Do not use elevators. Stay low to avoid smoke.",
    "HIGH": "Stay alert. Prepare to evacuate. Turn off electrical appliances. Keep fire extinguishers ready.",
    "MEDIUM": "Monitor the situation closely. Avoid open flames or sparks. Ensure proper ventilation.",
    "LOW": "No immediate danger. Stay cautious and continue monitoring conditions."
}

# -------------------- Load and Train Model --------------------
@st.cache_data
def train_model():
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

    acc = accuracy_score(y_test, rf.predict(X_test_s))
    roc = roc_auc_score(y_test, rf.predict_proba(X_test_s)[:, 1])

    return rf, scaler, features, acc, roc

rf, scaler, features, acc, roc = train_model()
st.success(f"✅ Model trained successfully! Accuracy: {acc:.2f}, ROC-AUC: {roc:.2f}")

# -------------------- User Input Form --------------------
st.subheader("🌡️ Enter Atmospheric Conditions")

user_input = {}
cols = st.columns(2)
for i, f in enumerate(features):
    with cols[i % 2]:
        user_input[f] = st.number_input(f, value=0.0, format="%.2f")

# -------------------- Prediction --------------------
if st.button("🚨 Predict Fire Risk"):
    x = np.array([user_input[f] for f in features], dtype=float).reshape(1, -1)
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

    msg = f"### Prediction: {'FIRE RISK' if pred == 1 else 'NO FIRE'} {emoji}\n" \
          f"**Probability:** {prob*100:.1f}%\n\n" \
          f"**Level:** {level}\n\n" \
          f"**Safety Instructions:** {safety_instructions[level]}"
    st.markdown(msg)

    # ---------------- Voice Output using gTTS ----------------
    text_to_speak = (
        f"The system predicts {level} fire risk. "
        f"The probability is {prob*100:.1f} percent. "
        f"{safety_instructions[level]}"
    )

    audio_fp = BytesIO()
    tts = gTTS(text=text_to_speak, lang="en")
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)

    st.audio(audio_fp, format="audio/mp3", start_time=0)

# -------------------- Pie Chart Visualization --------------------
st.subheader("📊 Feature Contribution Overview")

if st.button("Show Pie Chart"):
    values = list(user_input.values())
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=features,
        autopct="%1.1f%%",
        startangle=140,
        colors=plt.cm.Pastel1.colors,
        textprops={'fontsize': 8, 'fontname': 'Garamond'}
    )
    plt.setp(autotexts, size=8, weight="bold", color="black")
    ax.set_title("Contributing Factors to Fire Risk", fontsize=13, weight="bold", pad=10)
    st.pyplot(fig)

st.caption("Developed by Parshvi")




