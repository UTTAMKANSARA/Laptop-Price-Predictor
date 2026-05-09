import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="LaptopLens · AI Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load model & data ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    pipe = pickle.load(open("pipe.pkl", "rb"))
    df   = pickle.load(open("df.pkl",   "rb"))
    return pipe, df

pipe, df = load_artifacts()

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── Root palette ── */
:root {
    --bg:        #06080f;
    --surface:   #0d1117;
    --card:      rgba(255,255,255,0.045);
    --border:    rgba(0,210,255,0.18);
    --accent:    #00d2ff;
    --accent2:   #7b5ea7;
    --text:      #e8edf5;
    --muted:     #6b7a99;
    --success:   #00ffaa;
    --danger:    #ff4d6d;
    --radius:    16px;
    --transition: 0.3s cubic-bezier(.4,0,.2,1);
}

/* ── Global resets ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text) !important;
}

/* Animated grid background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,210,255,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,210,255,.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* Glow orbs */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    width: 600px; height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,210,255,.07) 0%, transparent 70%);
    top: -200px; right: -200px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"] { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div,
div[data-testid="stSlider"] > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px);
    transition: border-color var(--transition), box-shadow var(--transition);
}

div[data-testid="stSelectbox"] > div:hover,
div[data-testid="stNumberInput"] > div:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,210,255,.12) !important;
}

/* Selectbox text */
div[data-testid="stSelectbox"] span,
div[data-testid="stNumberInput"] input {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}

/* Labels */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Slider track */
div[data-testid="stSlider"] .stSlider > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    padding: 1rem 2rem !important;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 14px !important;
    cursor: pointer;
    transition: all var(--transition) !important;
    box-shadow: 0 4px 32px rgba(0,210,255,.25) !important;
    text-transform: uppercase;
    position: relative;
    overflow: hidden;
}

div[data-testid="stButton"] > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,.15) 0%, transparent 60%);
    opacity: 0;
    transition: opacity var(--transition);
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 48px rgba(0,210,255,.4) !important;
}

div[data-testid="stButton"] > button:hover::before { opacity: 1; }
div[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ── Dropdown popover ── */
[data-baseweb="popover"] {
    background: #141922 !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 16px 48px rgba(0,0,0,.6) !important;
}

[data-baseweb="menu"] li {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-baseweb="menu"] li:hover {
    background: rgba(0,210,255,.1) !important;
}

/* Number input arrows */
div[data-testid="stNumberInput"] button {
    background: rgba(0,210,255,.08) !important;
    border: none !important;
    color: var(--accent) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(0,210,255,.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Result card animation ── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}

@keyframes pulse-ring {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,210,255,.4); }
    50%       { box-shadow: 0 0 0 16px rgba(0,210,255,0); }
}

.result-card {
    animation: slideUp 0.6s cubic-bezier(.22,1,.36,1) forwards;
    background: linear-gradient(135deg, rgba(0,210,255,.08) 0%, rgba(123,94,167,.08) 100%);
    border: 1px solid rgba(0,210,255,.3);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
    margin-top: 1rem;
}

.result-card::before {
    content: '';
    position: absolute;
    top: -1px; left: -1px; right: -1px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: shimmer 2s infinite;
    background-size: 400px 100%;
}

.price-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

.price-value {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent) 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    letter-spacing: -0.02em;
}

.price-note {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.75rem;
}

.error-card {
    animation: slideUp 0.4s ease forwards;
    background: rgba(255,77,109,.08);
    border: 1px solid rgba(255,77,109,.3);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    color: #ff8099;
    font-size: 0.9rem;
}

/* Section headers */
.section-tag {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    background: rgba(0,210,255,.08);
    border: 1px solid rgba(0,210,255,.2);
    padding: 4px 10px;
    border-radius: 99px;
    margin-bottom: 0.5rem;
}

.section-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    margin: 0 0 1rem 0 !important;
}

/* Glass card wrapper */
.glass-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 0.5rem 0 1.5rem;
}

/* Tooltip badge */
.badge {
    display: inline-block;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 99px;
    background: rgba(123,94,167,.2);
    color: #a78bfa;
    font-weight: 500;
    margin-left: 6px;
    vertical-align: middle;
}

/* Stagger animation for columns */
[data-testid="column"]:nth-child(1) { animation: slideUp 0.5s .1s both; }
[data-testid="column"]:nth-child(2) { animation: slideUp 0.5s .2s both; }
[data-testid="column"]:nth-child(3) { animation: slideUp 0.5s .3s both; }

/* Remove default padding */
[data-testid="stMainBlockContainer"] { padding-top: 0 !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 3rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3.5rem 1rem 2rem; position: relative; z-index: 1;">
    <div style="
        display: inline-flex; align-items: center; gap: 8px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.16em;
        text-transform: uppercase; color: #00d2ff;
        background: rgba(0,210,255,.08); border: 1px solid rgba(0,210,255,.25);
        padding: 5px 14px; border-radius: 99px; margin-bottom: 1.2rem;
    ">
        <span style="width:6px;height:6px;background:#00d2ff;border-radius:50%;display:inline-block;
              box-shadow:0 0 8px #00d2ff; animation: pulse 2s infinite;"></span>
        AI-Powered · Instant Results
    </div>
    <h1 style="
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.4rem, 5vw, 4rem);
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #00d2ff 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.6rem;
        letter-spacing: -0.03em;
        line-height: 1.1;
    ">LaptopLens</h1>
    <p style="
        font-family: 'DM Sans', sans-serif;
        color: #6b7a99;
        font-size: 1.05rem;
        font-weight: 300;
        max-width: 480px;
        margin: 0 auto 0.5rem;
        line-height: 1.6;
    ">Enter your laptop specifications below and our AI model will predict the fair market price — instantly.</p>
</div>

<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(0,210,255,.2), transparent); margin: 0 2rem 2.5rem;"></div>
""", unsafe_allow_html=True)


# ── Layout ───────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    # ── Section 1: Identity ─────────────────────────────────────────────────
    st.markdown('<div class="section-tag">01 · Identity</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Brand & Category</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        company = st.selectbox("Brand", df["Company"].unique(), key="brand")
    with c2:
        type_name = st.selectbox("Type", df["TypeName"].unique(), key="type")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Section 2: Performance ──────────────────────────────────────────────
    st.markdown('<div class="section-tag">02 · Performance</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Processor & Memory</p>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        cpu = st.selectbox("CPU Brand", df["Cpu Brand"].unique(), key="cpu")
    with c4:
        ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3, key="ram")

    c5, c6 = st.columns(2)
    with c5:
        hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2048], key="hdd")
    with c6:
        ssd = st.selectbox("SSD (GB)", [0, 8, 128, 256, 512, 1024], index=2, key="ssd")

    gpu = st.selectbox("GPU Brand", df["Gpu brand"].unique(), key="gpu")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Section 3: Display ──────────────────────────────────────────────────
    st.markdown('<div class="section-tag">03 · Display</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Screen & Visuals</p>', unsafe_allow_html=True)

    screen_size = st.slider(
        "Screen Size (inches)", 10.0, 18.0, 13.0, 0.1, key="screen"
    )

    resolution = st.selectbox(
        "Screen Resolution",
        ["1920x1080", "1366x768", "1600x900", "3840x2160",
         "3200x1800", "2880x1800", "2560x1600", "2560x1440", "2304x1440"],
        key="res",
    )

    c7, c8 = st.columns(2)
    with c7:
        touchscreen = st.selectbox("Touchscreen", ["No", "Yes"], key="ts")
    with c8:
        ips = st.selectbox("IPS Display", ["No", "Yes"], key="ips")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Section 4: Build ────────────────────────────────────────────────────
    st.markdown('<div class="section-tag">04 · Build</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Physical & OS</p>', unsafe_allow_html=True)

    c9, c10 = st.columns(2)
    with c9:
        weight = st.number_input(
            "Weight (kg)", min_value=0.5, max_value=6.0,
            value=1.8, step=0.1, key="weight",
        )
    with c10:
        os = st.selectbox("Operating System", df["OS"].unique(), key="os")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict button ──────────────────────────────────────────────────────
    predict_clicked = st.button("⚡ Predict Market Price", key="predict")


# ── Right panel ─────────────────────────────────────────────────────────────
with right:
    # Summary card
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(0,210,255,0.12);
        border-radius: 18px;
        padding: 1.6rem;
        margin-bottom: 1.4rem;
        backdrop-filter: blur(12px);
    ">
        <p style="
            font-family:'Syne',sans-serif;
            font-size:0.85rem;font-weight:700;
            text-transform:uppercase;letter-spacing:0.1em;
            color:rgba(255,255,255,0.4);margin:0 0 1rem;
        ">How It Works</p>
        <div style="display:flex;flex-direction:column;gap:1rem;">
            <div style="display:flex;align-items:flex-start;gap:12px;">
                <div style="
                    width:34px;height:34px;border-radius:10px;flex-shrink:0;
                    background:rgba(0,210,255,.1);border:1px solid rgba(0,210,255,.2);
                    display:flex;align-items:center;justify-content:center;
                    font-size:1rem;
                ">🎛️</div>
                <div>
                    <p style="margin:0;font-size:0.88rem;font-weight:500;color:#e8edf5;font-family:'Syne',sans-serif;">Configure Specs</p>
                    <p style="margin:0;font-size:0.78rem;color:#6b7a99;line-height:1.5;">Choose your laptop's exact hardware & display specs from the options.</p>
                </div>
            </div>
            <div style="display:flex;align-items:flex-start;gap:12px;">
                <div style="
                    width:34px;height:34px;border-radius:10px;flex-shrink:0;
                    background:rgba(123,94,167,.1);border:1px solid rgba(123,94,167,.2);
                    display:flex;align-items:center;justify-content:center;
                    font-size:1rem;
                ">🧠</div>
                <div>
                    <p style="margin:0;font-size:0.88rem;font-weight:500;color:#e8edf5;font-family:'Syne',sans-serif;">AI Model Inference</p>
                    <p style="margin:0;font-size:0.78rem;color:#6b7a99;line-height:1.5;">Our trained ML pipeline processes 12+ features to estimate fair value.</p>
                </div>
            </div>
            <div style="display:flex;align-items:flex-start;gap:12px;">
                <div style="
                    width:34px;height:34px;border-radius:10px;flex-shrink:0;
                    background:rgba(0,255,170,.1);border:1px solid rgba(0,255,170,.2);
                    display:flex;align-items:center;justify-content:center;
                    font-size:1rem;
                ">💰</div>
                <div>
                    <p style="margin:0;font-size:0.88rem;font-weight:500;color:#e8edf5;font-family:'Syne',sans-serif;">Instant Price</p>
                    <p style="margin:0;font-size:0.78rem;color:#6b7a99;line-height:1.5;">Get the predicted market price in Indian Rupees within seconds.</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Prediction result ───────────────────────────────────────────────────
    if predict_clicked:
        ts_val  = 1 if touchscreen == "Yes" else 0
        ips_val = 1 if ips == "Yes" else 0

        X_res = int(resolution.split("x")[0])
        Y_res = int(resolution.split("x")[1])
        ppi   = ((X_res**2) + (Y_res**2)) ** 0.5 / screen_size

        query = pd.DataFrame([[
            company, type_name, ram, weight,
            ts_val, ips_val, ppi, cpu, hdd, ssd, gpu, os,
        ]], columns=[
            "Company", "TypeName", "Ram", "Weight",
            "Touchscreen", "Ips", "ppi",
            "Cpu Brand", "HDD", "SSD", "Gpu brand", "OS",
        ])

        try:
            with st.spinner("Analysing specifications…"):
                prediction  = pipe.predict(query)
                final_price = int(np.exp(prediction[0]))

            # Format price nicely
            price_fmt = f"₹{final_price:,}"

            # Low / mid / high bands for context
            lo = int(final_price * 0.92)
            hi = int(final_price * 1.08)

            st.markdown(f"""
            <div class="result-card">
                <p class="price-label">Estimated Market Price</p>
                <p class="price-value">{price_fmt}</p>
                <p class="price-note">
                    Typical range &nbsp;
                    <span style="color:#e8edf5;font-weight:500;">
                        ₹{lo:,} – ₹{hi:,}
                    </span>
                </p>
                <div style="
                    display:flex;justify-content:center;gap:1.5rem;
                    margin-top:1.4rem;padding-top:1.2rem;
                    border-top:1px solid rgba(255,255,255,0.06);
                ">
                    <div style="text-align:center;">
                        <p style="margin:0;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#6b7a99;">RAM</p>
                        <p style="margin:0;font-size:0.9rem;font-weight:600;color:#e8edf5;font-family:'Syne',sans-serif;">{ram} GB</p>
                    </div>
                    <div style="width:1px;background:rgba(255,255,255,0.07);"></div>
                    <div style="text-align:center;">
                        <p style="margin:0;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#6b7a99;">Storage</p>
                        <p style="margin:0;font-size:0.9rem;font-weight:600;color:#e8edf5;font-family:'Syne',sans-serif;">SSD {ssd}GB</p>
                    </div>
                    <div style="width:1px;background:rgba(255,255,255,0.07);"></div>
                    <div style="text-align:center;">
                        <p style="margin:0;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#6b7a99;">Screen</p>
                        <p style="margin:0;font-size:0.9rem;font-weight:600;color:#e8edf5;font-family:'Syne',sans-serif;">{screen_size}"</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="error-card">
                <strong>⚠ Prediction Error</strong><br>
                <span style="font-size:0.82rem;">{e}</span><br>
                <span style="font-size:0.78rem;opacity:0.7;">
                    Ensure pipe.pkl and df.pkl are in the same directory.
                </span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            border: 1px dashed rgba(0,210,255,0.2);
            border-radius: 16px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            color: #3a4a66;
        ">
            <div style="font-size:2.8rem;margin-bottom:0.8rem;opacity:0.5;">💡</div>
            <p style="font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:600;
                      color:#4a5a7a;margin:0 0 0.4rem;">Price will appear here</p>
            <p style="font-size:0.78rem;margin:0;line-height:1.6;color:#3a4a66;">
                Fill in the specs on the left<br>and hit <strong style="color:#4a6a8a;">Predict Market Price</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    text-align:center;
    padding: 2.5rem 1rem 1rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 2rem;
">
    <p style="
        font-family:'DM Sans',sans-serif;
        font-size:0.75rem;
        color:#3a4a66;
        letter-spacing:0.06em;
        margin:0;
    ">
        LaptopLens · Powered by Machine Learning &nbsp;·&nbsp; Prices are estimates based on training data
    </p>
</div>
""", unsafe_allow_html=True)
