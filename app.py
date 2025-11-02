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

st.markdown("""
    <style>
    h1, h2, h3, h4, h5, h6 { text-align: center; }
    .stMarkdown { text-align: center; }
    div.stButton > button {
        display: block;
        margin: 0 auto;
    }
    div[data-testid="stHorizontalBlock"] {
        justify-content: center;
    }
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        color: #222;
    }
    .feature-info {
        font-size: 0.8em;
        color: #666;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔥 FireCast: Pre-Fire Detection System")
st.markdown("### Predict Fire Risks based on Atmospheric Conditions")

safety_instructions = {
    "CRITICAL": "Evacuate immediately. Call emergency services. Do not use elevators. Stay low to avoid smoke.",
    "HIGH": "Stay alert. Prepare to evacuate. Turn off electrical appliances. Keep fire extinguishers ready.",
    "MEDIUM": "Monitor the situation closely. Avoid open flames or sparks. Ensure proper ventilation.",
    "LOW": "No immediate danger. Stay cautious and continue monitoring conditions."
}

feature_info = {
    "Temperature[C]": ("Higher temperatures increase fire risk due to heat buildup.", "Normal ambient temperature is around 25°C."),
    "Humidity[%]": ("Low humidity helps fires spread faster.", "Typical humidity ranges from 40% to 60%."),
    "TVOC[ppb]": ("Volatile Organic Compounds may rise during combustion.", "Normal indoor TVOC is below 500 ppb."),
    "eCO2[ppm]": ("Elevated CO2 levels can indicate smoke or poor air.", "Normal CO2 levels are around 400 ppm."),
    "Raw H2": ("Increased hydrogen can suggest incomplete combustion.", "Normally under 1000 ppm."),
    "Raw Ethanol": ("Ethanol vapors suggest burning organic material.", "Normally under 1000 ppm."),
    "Pressure[hPa]": ("Pressure drops can occur due to fire turbulence.", "Normal air pressure is around 1013 hPa."),
    "PM1.0": ("Fine particles increase with smoke presence.", "Normal PM1.0 is below 50 μg/m³."),
    "PM2.5": ("Small particles indicate smoke and ash.", "Normal PM2.5 is below 100 μg/m³."),
    "NC0.5": ("Particle count (0.5μm size) rises with combustion.", "Normal NC0.5 is below 5000."),
    "NC1.0": ("Particle count (1.0μm size) reflects smoke density.", "Normal NC1.0 is below 5000."),
    "NC2.5": ("Larger particles appear during heavy smoke.", "Normal NC2.5 is below 5000."),
    "CNT": ("Total particle count from all sensors combined.", "Normal total particle count is under 10,000.")
}

# -------------------- Load and Train Model --------------------
@st.cache_data
def train_model():
    df = pd.read_csv("smoke_detection_iot.csv")

    target_col = "Fire Alarm"
    features = list(feature_info.keys())

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
        
        user_input[f] = st.number_input(
            f"**{f}**",
            value=0.0,
            format="%.2f",
            help=feature_info[f][1]  
        )
       
        st.markdown(f"<p class='feature-info'>{feature_info[f][0]}</p>", unsafe_allow_html=True)

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

    # --- Voice Output ---
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
