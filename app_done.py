import base64
import html
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rental Property Recommender", page_icon="🏠", layout="wide",
                   initial_sidebar_state="collapsed")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "final_final.xlsx"
RENT_MODEL_PATH = BASE_DIR / "rent_model.joblib"        # Gradient Boosting pipeline
KMEANS_PATH = BASE_DIR / "kmeans_model_final.pkl"
SCALER_PATH = BASE_DIR / "scaler_final.pkl"
# Optional: put a CLEAN photo (no UI in it) next to app.py named house_background.png (or bg_photo.jpg/.png).
# Do NOT use the mockup screenshot - its UI would show through behind the glass.
BACKGROUND_CANDIDATES = [f"{n}.{e}" for n in ("house_background", "bg_photo") for e in ("jpg", "jpeg", "png", "webp")]

# Input columns expected by the rent model (it does its own one-hot encoding and log transform)
REG_FEATURES = ["city", "bathroom", "bhk", "furnished_status", "area"]
CLUSTER_FEATURES = ["Rent", "Size", "BHK", "Bathroom"]
CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
FURNISHING = ["Furnished", "Semi-Furnished", "Unfurnished"]
AREA_TYPE_LABELS = {1: "Carpet Area", 2: "Built Area", 3: "Super Area",
                    "1": "Carpet Area", "2": "Built Area", "3": "Super Area"}
BUDGET_MIN = 5000
MAX_RESULTS = 50
PAGE_SIZE = 10


# ----------------------------------------------------------------------------
# Liquid-glass styling
# ----------------------------------------------------------------------------
DEFAULT_BG_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1672 941" preserveAspectRatio="xMidYMid slice">
<defs>
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#171c38"/><stop offset=".25" stop-color="#3b3358"/><stop offset=".42" stop-color="#a5546a"/><stop offset=".53" stop-color="#e9855a"/><stop offset=".6" stop-color="#f3a469"/></linearGradient>
<linearGradient id="pool" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2a7f9c"/><stop offset=".5" stop-color="#14425f"/><stop offset="1" stop-color="#0a1c33"/></linearGradient>
<linearGradient id="glass" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffcf8a"/><stop offset="1" stop-color="#ff8c3f"/></linearGradient>
<linearGradient id="wall" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2b2f45"/><stop offset="1" stop-color="#171a2a"/></linearGradient>
<linearGradient id="ground" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2b2c3e"/><stop offset="1" stop-color="#1a1b2b"/></linearGradient>
<radialGradient id="halo" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#ffb066" stop-opacity=".55"/><stop offset="1" stop-color="#ffb066" stop-opacity="0"/></radialGradient>
<filter id="b6"><feGaussianBlur stdDeviation="6"/></filter>
<filter id="b14"><feGaussianBlur stdDeviation="14"/></filter>
<filter id="b3"><feGaussianBlur stdDeviation="2.5"/></filter>
</defs>
<rect width="1672" height="941" fill="url(#sky)"/>
<g fill="#fff" opacity=".7"><circle cx="180" cy="70" r="1.4"/><circle cx="420" cy="40" r="1.2"/><circle cx="760" cy="90" r="1.3"/><circle cx="1100" cy="50" r="1.4"/><circle cx="1380" cy="110" r="1.2"/><circle cx="1560" cy="60" r="1.5"/><circle cx="1250" cy="30" r="1"/><circle cx="560" cy="140" r="1"/></g>
<ellipse cx="900" cy="560" rx="620" ry="120" fill="#ff9a5a" opacity=".35" filter="url(#b14)"/>
<path d="M0 520 Q120 470 230 520 T470 510 L470 700 L0 700Z" fill="#101a1c"/>
<path d="M1230 520 Q1330 460 1440 510 T1672 500 L1672 700 L1230 700Z" fill="#101a1c"/>
<g fill="#0d1517"><circle cx="70" cy="470" r="70"/><circle cx="190" cy="440" r="85"/><circle cx="330" cy="470" r="75"/><circle cx="1480" cy="450" r="85"/><circle cx="1610" cy="470" r="80"/><circle cx="1360" cy="480" r="62"/></g>
<g fill="#ffcf8a" opacity=".85" filter="url(#b3)"><circle cx="90" cy="560" r="5"/><circle cx="170" cy="600" r="4"/><circle cx="260" cy="580" r="5"/><circle cx="1500" cy="570" r="5"/><circle cx="1590" cy="600" r="4"/><circle cx="1430" cy="610" r="5"/></g>
<rect x="470" y="330" width="520" height="95" fill="url(#wall)"/>
<rect x="450" y="318" width="570" height="16" fill="#3a3f58"/>
<rect x="500" y="352" width="200" height="56" fill="url(#glass)" opacity=".9"/>
<rect x="716" y="352" width="120" height="56" fill="url(#glass)" opacity=".75"/>
<rect x="850" y="352" width="110" height="56" fill="#222742"/>
<rect x="760" y="425" width="560" height="14" fill="#3a3f58"/>
<rect x="780" y="439" width="520" height="195" fill="url(#wall)"/>
<rect x="800" y="458" width="150" height="150" fill="url(#glass)"/>
<rect x="962" y="458" width="150" height="150" fill="url(#glass)" opacity=".95"/>
<rect x="1124" y="458" width="160" height="150" fill="url(#glass)" opacity=".85"/>
<g stroke="#2b2f45" stroke-width="5"><line x1="875" y1="458" x2="875" y2="608"/><line x1="1037" y1="458" x2="1037" y2="608"/><line x1="1204" y1="458" x2="1204" y2="608"/></g>
<rect x="470" y="425" width="300" height="209" fill="#1b1f33"/>
<rect x="500" y="455" width="110" height="150" fill="url(#glass)" opacity=".8"/>
<rect x="625" y="455" width="120" height="150" fill="url(#glass)" opacity=".6"/>
<ellipse cx="1040" cy="540" rx="320" ry="130" fill="url(#halo)"/>
<path d="M440 634 L1340 634 L1340 650 L440 650Z" fill="#3a3f58"/>
<rect x="0" y="650" width="1672" height="291" fill="url(#ground)"/>
<path d="M120 940 L330 705 L1500 705 L1672 760 L1672 941Z" fill="url(#pool)"/>
<g filter="url(#b6)" opacity=".7"><rect x="830" y="715" width="120" height="200" fill="#ffb066"/><rect x="990" y="715" width="120" height="180" fill="#ff9a4f"/><rect x="1150" y="715" width="140" height="170" fill="#ff8c3f"/><rect x="520" y="715" width="100" height="150" fill="#ffb066" opacity=".6"/></g>
<g stroke="#bfe6f5" stroke-width="1.5" opacity=".25" fill="none"><path d="M360 740 Q700 725 1060 745 T1500 740"/><path d="M300 790 Q700 770 1100 795 T1600 785"/><path d="M240 850 Q700 825 1150 855 T1672 840"/><path d="M180 905 Q700 880 1200 910 T1672 895"/></g>
<path d="M330 705 L1500 705" stroke="#4b4f68" stroke-width="6"/>
<g fill="#10121f" opacity=".9"><rect x="1400" y="680" width="150" height="14" rx="4"/><rect x="1410" y="694" width="6" height="22"/><rect x="1535" y="694" width="6" height="22"/><rect x="1330" y="685" width="40" height="12" rx="4"/></g>
<g fill="#0d1517"><rect x="0" y="650" width="260" height="40"/><circle cx="60" cy="660" r="45"/><circle cx="170" cy="670" r="38"/><circle cx="1560" cy="690" r="50"/></g>
</svg>
"""


@st.cache_resource
def background_css():
    for name in BACKGROUND_CANDIDATES:
        p = BASE_DIR / name
        if p.exists():
            mime = {".png": "image/png", ".webp": "image/webp"}.get(p.suffix, "image/jpeg")
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'url("data:{mime};base64,{b64}")'
    b64 = base64.b64encode(DEFAULT_BG_SVG.encode("utf-8")).decode()
    return f'url("data:image/svg+xml;base64,{b64}")'


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
    --red: #F94144;
    --glass: radial-gradient(120% 70% at 0% 0%, rgba(255,255,255,.16), transparent 55%),
             linear-gradient(145deg, rgba(255,255,255,.09), rgba(255,255,255,.012) 45%, rgba(255,255,255,.05));
    --glass-tint: rgba(14,18,32,.06);
    --rim: inset 0 1px 0 rgba(255,255,255,.55), inset 1px 0 0 rgba(255,255,255,.22),
           inset 0 -1px 0 rgba(255,255,255,.07), inset -1px 0 0 rgba(255,255,255,.05);
}
html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", sans-serif;
    letter-spacing: -0.011em;
}
.stApp { background: #0b0e18; isolation: isolate; -webkit-font-smoothing: antialiased; }
/* Background lives on its own fixed, GPU-composited layer so it is painted once and never
   re-rasterised while scrolling (background-attachment: fixed forced a repaint every frame). */
.stApp::before {
    content:""; position:fixed; inset:0; z-index:-1; pointer-events:none;
    background-image:
        radial-gradient(60% 50% at 12% 18%, rgba(249,65,68,.10), transparent 70%),
        radial-gradient(55% 45% at 88% 72%, rgba(90,120,255,.08), transparent 70%),
        radial-gradient(40% 35% at 70% 8%, rgba(255,170,110,.06), transparent 70%),
        linear-gradient(rgba(8,12,22,.05), rgba(8,12,22,.25)),
        __BG__;
    background-size: cover; background-position: center;
    transform: translateZ(0); will-change: transform; backface-visibility: hidden;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], footer, #MainMenu { visibility: hidden; }
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container { max-width: 1500px; padding-top: 1.2rem; padding-bottom: 3rem; }

/* ---------- Brand + nav ---------- */
.brand { display:flex; align-items:center; gap:14px; }
.brand .ic {
    width:54px; height:54px; border-radius:18px; flex:none; display:grid; place-items:center;
    background: var(--glass), rgba(20,24,40,.45); box-shadow: var(--rim), 0 10px 30px rgba(0,0,0,.35);
    border:1px solid rgba(255,255,255,.16);
}
.brand .ic svg { width:30px; height:30px; }
.brand .t { font-size:1.75rem; font-weight:700; line-height:1.1; color:#fff; letter-spacing:-0.03em; }
.brand .t span { color: var(--red); }
.brand .s { font-size:.95rem; color:#cfd4e0; margin-top:3px; }

/* Nav pill: three ordinary buttons (works on every Streamlit version, unlike restyling st.radio) */
.st-key-navbar {
    width:27rem; max-width:100%; margin-left:auto; padding:5px; border-radius:999px;
    background: var(--glass), rgba(20,24,40,.35); box-shadow: var(--rim), 0 8px 28px rgba(0,0,0,.30);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,.16);
}
.st-key-navbar [data-testid="stHorizontalBlock"] { gap:0 !important; flex-wrap:nowrap !important; }
.st-key-navbar [data-testid="stColumn"], .st-key-navbar [data-testid="column"] { min-width:0 !important; flex:1 1 0 !important; }
.st-key-navbar button {
    border-radius:999px !important; border:1px solid transparent !important; background:transparent !important;
    color:#dfe3ee !important; box-shadow:none !important; padding:.5rem 0 !important; filter:none !important; transform:none !important;
    transition: background .25s, color .25s;
}
.st-key-navbar button:hover { background: rgba(255,255,255,.14) !important; color:#fff !important; }
.st-key-navbar button p { font-size:.95rem; white-space:nowrap; }
.st-key-navbar button[kind="primary"] {
    background: linear-gradient(180deg, rgba(255,255,255,.38), rgba(255,255,255,.13)) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.65), inset 0 -1px 0 rgba(255,255,255,.12), 0 6px 18px rgba(0,0,0,.35) !important;
    color:#fff !important; font-weight:600;
}

/* ---------- Glass cards ---------- */
.st-key-prefs_card, .st-key-rent_card, .st-key-results_card, .st-key-page_card {
    background: var(--glass), rgba(14,18,32,.22);
    backdrop-filter: blur(14px) saturate(130%); -webkit-backdrop-filter: blur(14px) saturate(130%);
    border:1px solid rgba(255,255,255,.28); border-radius:32px; padding:1.8rem 2rem;
    box-shadow: var(--rim), 0 24px 70px rgba(0,0,0,.45);
    isolation: isolate;
}
div.card-title { text-shadow: 0 1px 12px rgba(0,0,0,.35); font-size:1.5rem !important; font-weight:700; color:#fff; margin:0; letter-spacing:-0.025em; }
div.card-sub { font-size:.93rem !important; color:#cfd4e0; margin:.2rem 0 .9rem; }
.big-rent {
    font-size:3.9rem; font-weight:700; color:var(--red); line-height:1.05; margin:.3rem 0 1.1rem;
    letter-spacing:-0.04em; text-shadow: 0 0 40px rgba(249,65,68,.45);
}
.info-strip {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:1rem;
    background: linear-gradient(145deg, rgba(255,255,255,.07), rgba(255,255,255,.015));
    border:1px solid rgba(255,255,255,.14); box-shadow: var(--rim);
    border-radius:22px; padding:1.1rem 1.4rem; margin-bottom:1.5rem;
}
.info-strip .k, .info-strip .v { white-space:nowrap; }
.info-strip .k { font-size:.88rem; color:#b9c0cf; }
.info-strip .v { font-size:1.1rem; font-weight:600; color:#fff; margin-top:.35rem; }
div.empty-note { color:#cfd4e0; padding:2rem 0 1rem; font-size:1rem; }

/* ---------- Widgets ---------- */
div[data-baseweb="select"] > div, div[data-baseweb="input"] {
    background: rgba(255,255,255,.06) !important; border:1px solid rgba(255,255,255,.24) !important;
    border-radius:16px !important; box-shadow: inset 0 1px 0 rgba(255,255,255,.14); overflow:hidden;
}
/* number input: ONE glass shell holds the field and the - / + buttons */
div[data-baseweb="base-input"], [data-testid="stNumberInputContainer"] {
    background: transparent !important; border:none !important; box-shadow:none !important; border-radius:0 !important;
}
div[data-baseweb="input"] input { color:#fff; background: transparent !important; }
/* each - / + gets its own circular liquid-glass button */
[data-testid="stNumberInput"] button {
    width:34px !important; height:34px !important; min-width:34px; padding:0 !important;
    margin:0 0 0 8px !important; align-self:center; border-radius:50% !important; color:#fff !important;
    background: radial-gradient(120% 90% at 20% 0%, rgba(255,255,255,.34), rgba(255,255,255,.07) 60%),
                rgba(255,255,255,.08) !important;
    border:1px solid rgba(255,255,255,.30) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.55), inset 0 -1px 0 rgba(255,255,255,.10), 0 4px 12px rgba(0,0,0,.30) !important;
    display:flex; align-items:center; justify-content:center;
    transition: transform .2s cubic-bezier(.34,1.45,.5,1), background .2s;
}
[data-testid="stNumberInput"] button:hover { transform: scale(1.12); color:#fff !important; border-color: rgba(255,255,255,.5) !important; }
[data-testid="stNumberInput"] button:active { transform: scale(.94); }
[data-testid="stNumberInput"] div[data-baseweb="input"] { padding-right:8px; align-items:center; }
div[role="slider"] {
    background:#fff !important;
    box-shadow: 0 2px 10px rgba(0,0,0,.45), 0 0 0 5px rgba(249,65,68,.30) !important;
}
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] p { color:#ff6a6d !important; font-weight:600; }
[data-testid="stWidgetLabel"] p { font-size:.96rem; font-weight:500; color:#f4f6fb; }

.stButton > button[kind="primary"], .stDownloadButton > button {
    background: linear-gradient(180deg, #ff6467 0%, #F94144 55%, #e22b2f 100%);
    border:1px solid rgba(255,255,255,.28); color:#fff; font-weight:600; border-radius:999px; padding:.8rem 1.2rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.5), inset 0 -1px 0 rgba(0,0,0,.12), 0 10px 28px rgba(249,65,68,.42);
    transition: transform .15s, filter .15s;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover { filter:brightness(1.08); transform: translateY(-1px); color:#fff; }
.stButton > button[kind="primary"]:active, .stDownloadButton > button:active { transform: scale(.98); }
.stButton > button[kind="secondary"] {
    border-radius:999px; color:#fff; border:1px solid rgba(255,255,255,.22);
    background: linear-gradient(180deg, rgba(255,255,255,.18), rgba(255,255,255,.06));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.35);
}
.stButton > button[kind="secondary"]:hover { border-color: rgba(255,255,255,.4); color:#fff; }
[data-testid="stDataFrame"] { border-radius:20px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:20px; }

/* ---------- Dropdown (popover) ---------- */
div[data-baseweb="popover"] > div { background: transparent !important; box-shadow:none !important; border-radius:24px !important; }
div[data-baseweb="popover"] ul[role="listbox"], div[data-baseweb="popover"] [data-baseweb="menu"] {
    background: var(--glass), rgba(18,22,38,.55) !important;
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border:1px solid rgba(255,255,255,.20); border-radius:24px !important; padding:8px !important;
    box-shadow: var(--rim), 0 24px 60px rgba(0,0,0,.5) !important;
}
div[data-baseweb="popover"] li[role="option"] {
    background: transparent !important; color:#f4f6fb; border-radius:16px; margin:2px 0; transition: background .2s;
}
div[data-baseweb="popover"] li[role="option"]:hover { background: rgba(255,255,255,.14) !important; }
div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
    background: linear-gradient(180deg, rgba(255,255,255,.24), rgba(255,255,255,.09)) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.4);
}

/* ---------- Glass table ---------- */
.gtable-wrap {
    overflow-x:auto; margin-top:1rem; border-radius:24px; padding:.4rem .6rem;
    background: linear-gradient(145deg, rgba(255,255,255,.07), rgba(255,255,255,.015));
    border:1px solid rgba(255,255,255,.16); box-shadow: var(--rim);
}
table.gtable { width:100%; border-collapse:separate; border-spacing:0 4px; font-variant-numeric: tabular-nums; }
.gtable th { text-align:right; font-size:.8rem; font-weight:500; color:#b9c0cf; padding:.7rem 1rem .4rem; white-space:nowrap; }
.gtable th:first-child, .gtable td:first-child { text-align:left; }
.gtable td { text-align:right; padding:.75rem 1rem; color:#f4f6fb; font-size:.95rem; background: rgba(255,255,255,.05); white-space:nowrap; transition: background .2s; }
.gtable td:first-child { border-radius:16px 0 0 16px; }
.gtable td:last-child { border-radius:0 16px 16px 0; }
.gtable tr:hover td { background: rgba(255,255,255,.13); }
.gtable .pill { display:inline-block; min-width:2rem; text-align:center; padding:.15rem .7rem; border-radius:999px;
    background: rgba(249,65,68,.22); color:#ff8588; border:1px solid rgba(249,65,68,.4); font-weight:600; }

/* ---------- Results ---------- */
.chips { display:flex; gap:8px; flex-wrap:wrap; margin:.9rem 0 1rem; }
.chip {
    background: linear-gradient(145deg, rgba(255,255,255,.14), rgba(255,255,255,.04));
    border:1px solid rgba(255,255,255,.16); box-shadow: var(--rim);
    border-radius:999px; padding:.35rem .9rem; font-size:.85rem; color:#d6dae5;
}
.chip b { color:#fff; font-weight:600; }
.listing {
    display:grid; grid-template-columns:minmax(0,1fr) auto; gap:1rem; align-items:center;
    background: linear-gradient(145deg, rgba(255,255,255,.07), rgba(255,255,255,.015));
    border:1px solid rgba(255,255,255,.18); box-shadow: var(--rim);
    border-radius:24px; padding:1rem 1.3rem; margin-bottom:.7rem;
    transition: transform .2s, border-color .2s;
    content-visibility: auto; contain-intrinsic-size: auto 118px;
}
.listing:hover { transform: translateY(-2px); border-color: rgba(249,65,68,.6); }
.listing .loc { font-size:1.05rem; font-weight:600; color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.listing .sub { font-size:.85rem; color:#b9c0cf; margin:.1rem 0 .55rem; }
.tags { display:flex; gap:6px; flex-wrap:wrap; }
.tag { background: rgba(255,255,255,.10); border-radius:999px; padding:.18rem .7rem; font-size:.8rem; color:#e3e6ee; border:1px solid rgba(255,255,255,.08); }
.tag.ok { background: rgba(52,199,120,.18); color:#7fe0a3; border-color: rgba(52,199,120,.3); }
.listing .rent { font-size:1.55rem; font-weight:700; color:var(--red); text-align:right; line-height:1.1; letter-spacing:-0.03em; }
.listing .rent small { display:block; font-size:.75rem; font-weight:400; color:#b9c0cf; margin-top:.2rem; letter-spacing:0; }
.foot { text-align:center; color:#b9c0cf; font-size:.85rem; margin:.4rem 0 .6rem; }

@supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
    .st-key-prefs_card, .st-key-rent_card, .st-key-results_card, .st-key-page_card { background: rgba(20,24,40,.82); }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { transition-duration: .01ms !important; animation-duration: .01ms !important; }
}
@media (max-width: 900px) {
    .big-rent { font-size:2.7rem; }
    .listing { grid-template-columns:1fr; }
    .listing .rent { text-align:left; }
}
</style>
"""

st.markdown(CSS.replace("__BG__", background_css()), unsafe_allow_html=True)

HOUSE_SVG = """<svg viewBox="0 0 24 24" fill="none" stroke="#F94144" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v10h13V10"/><path d="M10 20v-6h4v6"/></svg>"""


# ----------------------------------------------------------------------------
# Data / models
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    return joblib.load(RENT_MODEL_PATH), joblib.load(KMEANS_PATH), joblib.load(SCALER_PATH)


@st.cache_data
def load_data(_kmeans, _scaler):
    data = pd.read_excel(DATA_PATH)
    required = set(CLUSTER_FEATURES + ["City", "Furnishing Status"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    # Cluster labels are computed once and cached, not on every widget interaction.
    data["Cluster"] = _kmeans.predict(_scaler.transform(data[CLUSTER_FEATURES]))
    return data


try:
    lr, kmeans, scaler = load_models()
    df = load_data(kmeans, scaler)
except Exception as exc:
    st.error("Could not load the models or dataset. Put app.py, rent_model.joblib, kmeans_model_final.pkl, scaler_final.pkl and final_final.xlsx in the same folder.")
    st.exception(exc)
    st.stop()


def inr(n):
    """Format a number with Indian digit grouping, e.g. 1234567 -> 12,34,567."""
    s = str(int(round(n)))
    if len(s) <= 3:
        return s
    last3, rest, groups = s[-3:], s[:-3], []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return ",".join(groups + [last3])


def predict_rent(size, bhk, bathrooms, furnishing, city):
    row = pd.DataFrame([{
        "city": city, "bathroom": bathrooms, "bhk": bhk,
        "furnished_status": furnishing, "area": size,
    }])[REG_FEATURES]
    # The pipeline predicts log(1 + rent) internally and returns rent in rupees.
    return max(0.0, float(lr.predict(row)[0]))


def get_cluster(rent, size, bhk, bathrooms):
    x = pd.DataFrame([[rent, size, bhk, bathrooms]], columns=CLUSTER_FEATURES)
    return int(kmeans.predict(scaler.transform(x))[0])


@st.cache_data(show_spinner=False)
def search(city, cluster_id, budget, bhk, bathrooms, size, furnishing_filter, all_clusters=False):
    matches = df[
        (df["City"].astype(str).str.casefold() == city.casefold()) &
        (pd.to_numeric(df["Rent"], errors="coerce") <= budget)
    ].copy()
    if not all_clusters:
        matches = matches[matches["Cluster"] == cluster_id].copy()
    if furnishing_filter != "Any":
        matches = matches[
            matches["Furnishing Status"].astype(str).str.casefold() == furnishing_filter.casefold()
        ].copy()
    if matches.empty:
        return matches
    matches["BHK_Diff"] = (matches["BHK"] - bhk).abs()
    matches["Bathroom_Diff"] = (matches["Bathroom"] - bathrooms).abs()
    matches["Size_Diff"] = (pd.to_numeric(matches["Size"], errors="coerce") - size).abs()
    # Closest BHK, then bathrooms, then area (sq ft), and only then rent (highest to lowest).
    return matches.sort_values(["BHK_Diff", "Bathroom_Diff", "Size_Diff", "Rent"],
                               ascending=[True, True, True, False])


def to_export_frame(frame):
    cols = ["City", "Area Locality", "Rent", "Size", "BHK", "Bathroom",
            "Furnishing Status", "Area Type", "Apartment_Floor", "Total_Floors", "Cluster"]
    out = frame[[c for c in cols if c in frame.columns]].head(MAX_RESULTS).reset_index(drop=True)
    if "Area Type" in out.columns:
        out["Area Type"] = out["Area Type"].map(lambda v: AREA_TYPE_LABELS.get(v, v))
    for c in ("Apartment_Floor", "Total_Floors"):      # show 5 instead of 5.0
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


@st.cache_data(show_spinner=False)
def to_excel_bytes(frame):
    """Build a formatted .xlsx file (bold header, frozen header row, auto column widths)."""
    from openpyxl.styles import Font
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Recommended Properties")
        ws = writer.sheets["Recommended Properties"]
        ws.freeze_panes = "A2"
        for col_cells in ws.columns:
            letter = col_cells[0].column_letter
            header = str(col_cells[0].value)
            longest = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            ws.column_dimensions[letter].width = min(max(longest, len(header)) + 3, 45)
            col_cells[0].font = Font(bold=True)
            if header in ("Rent", "Size"):
                for c in col_cells[1:]:
                    c.number_format = "#,##0"
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def cluster_profiles(_data):
    return _data.groupby("Cluster").agg(
        Property_Count=("Rent", "count"),
        Median_Rent=("Rent", "median"),
        Q1_Rent=("Rent", lambda x: x.quantile(.25)),
        Q3_Rent=("Rent", lambda x: x.quantile(.75)),
        Median_Size=("Size", "median"),
        Median_BHK=("BHK", "median"),
        Median_Bathroom=("Bathroom", "median"),
    ).round(0)


def render_cluster_table(profiles):
    money = {"Median_Rent", "Q1_Rent", "Q3_Rent"}
    labels = {"Property_Count": "Properties", "Median_Rent": "Median rent", "Q1_Rent": "Q1 rent",
              "Q3_Rent": "Q3 rent", "Median_Size": "Median size (sq ft)",
              "Median_BHK": "Median BHK", "Median_Bathroom": "Median bathrooms"}
    head = "<th>Cluster</th>" + "".join(f"<th>{labels.get(c, c)}</th>" for c in profiles.columns)
    body = ""
    for idx, row in profiles.iterrows():
        cells = "".join(
            f"<td>{'₹' if c in money else ''}{inr(row[c])}</td>" for c in profiles.columns)
        body += f'<tr><td><span class="pill">{idx}</span></td>{cells}</tr>'
    st.markdown(f'<div class="gtable-wrap"><table class="gtable"><thead><tr>{head}</tr></thead>'
                f'<tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


def render_listings(frame, n):
    """Draw listing rows as liquid-glass cards."""
    cards = []
    for _, r in frame.head(n).iterrows():
        e = lambda v: html.escape(str(v))
        locality = e(r.get("Area Locality", "Unknown locality"))
        tags = []
        if r.get("BHK_Diff", 1) == 0 and r.get("Bathroom_Diff", 1) == 0:
            tags.append('<span class="tag ok">Exact match</span>')
        tags.append(f'<span class="tag">{int(r["BHK"])} BHK</span>')
        tags.append(f'<span class="tag">{int(r["Bathroom"])} bath</span>')
        tags.append(f'<span class="tag">{int(r["Size"]):,} sq ft</span>')
        tags.append(f'<span class="tag">{e(r["Furnishing Status"])}</span>')
        if "Area Type" in r.index:
            tags.append(f'<span class="tag">{e(AREA_TYPE_LABELS.get(r["Area Type"], r["Area Type"]))}</span>')
        if "Apartment_Floor" in r.index and pd.notna(r["Apartment_Floor"]):
            floor = f'{int(r["Apartment_Floor"])}'
            if "Total_Floors" in r.index and pd.notna(r["Total_Floors"]):
                floor += f' of {int(r["Total_Floors"])}'
            tags.append(f'<span class="tag">Floor: {floor}</span>')
        cards.append(
            f'<div class="listing"><div style="min-width:0"><div class="loc">{locality}</div>'
            f'<div class="sub">{e(r["City"])}</div><div class="tags">{"".join(tags)}</div></div>'
            f'<div class="rent">₹{inr(r["Rent"])}<small>per month</small></div></div>'
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Session state defaults + callbacks
# ----------------------------------------------------------------------------
# Streamlit deletes the state of any widget that is not drawn in a run, so leaving the Home page
# (to About / Data Insights) used to wipe every input. We keep a plain copy of those values in
# session state and put them back at the start of each run, before the widgets are created.
PERSISTED_KEYS = ["city", "size_slider", "size_number", "bhk", "bathrooms", "furnishing",
                  "budget_slider", "budget_number", "enforce_furnishing"]
_saved_inputs = st.session_state.setdefault("_saved_inputs", {})
for _k, _v in _saved_inputs.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

defaults = {"city": "Chennai", "size_slider": 800, "size_number": 800,
            "bhk": 2, "bathrooms": 2, "furnishing": "Furnished", "nav": "Home",
            "show_n": PAGE_SIZE, "expand_clusters": False}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


def sync_size_slider():
    st.session_state["size_number"] = st.session_state["size_slider"]


def sync_size_number():
    st.session_state["size_slider"] = st.session_state["size_number"]


def sync_budget_slider():
    st.session_state["budget_number"] = st.session_state["budget_slider"]


def sync_budget_number():
    st.session_state["budget_slider"] = st.session_state["budget_number"]


PAGES = ["Home", "About", "Data Insights"]


def go_to(page_name):
    st.session_state["nav"] = page_name


def show_more():
    st.session_state["show_n"] += PAGE_SIZE


def expand_clusters():
    st.session_state["expand_clusters"] = True
    st.session_state["show_n"] = PAGE_SIZE


def on_predict():
    s = st.session_state
    try:
        predicted = predict_rent(s["size_number"], s["bhk"], s["bathrooms"], s["furnishing"], s["city"])
        cluster = get_cluster(predicted, s["size_number"], s["bhk"], s["bathrooms"])
        s["request"] = {
            "city": s["city"], "size": int(s["size_number"]), "bhk": int(s["bhk"]),
            "bathrooms": int(s["bathrooms"]), "furnishing": s["furnishing"],
            "predicted": predicted, "cluster": cluster,
        }
        suggested = max(0, int(round(predicted)))
        budget_max = max(100000, int(np.ceil(suggested * 3 / 1000) * 1000))
        start = min(max(suggested, BUDGET_MIN), budget_max)
        s["budget_slider"] = start
        s["budget_number"] = start
        s.pop("search_result", None)
        s.pop("predict_error", None)
        s["expand_clusters"] = False
        s["show_n"] = PAGE_SIZE
    except Exception as exc:
        s["predict_error"] = exc


def on_find():
    s = st.session_state
    req = s["request"]
    budget = float(s["budget_number"])
    cluster_id = get_cluster(budget, req["size"], req["bhk"], req["bathrooms"])
    s["search_result"] = {
        **req, "budget": budget, "cluster": cluster_id,
        "furnishing_filter": req["furnishing"] if s.get("enforce_furnishing") else "Any",
    }
    s["expand_clusters"] = False
    s["show_n"] = PAGE_SIZE


# ----------------------------------------------------------------------------
# Header + nav
# ----------------------------------------------------------------------------
h1, h2 = st.columns([3, 2], vertical_alignment="center")
with h1:
    st.markdown(
        f'<div class="brand"><div class="ic">{HOUSE_SVG}</div><div><div class="t">Rental Property <span>Recommender</span></div>'
        f'<div class="s">Find the right home at the right price</div></div></div>',
        unsafe_allow_html=True,
    )
with h2:
    with st.container(key="navbar"):
        for _col, _name in zip(st.columns(3), PAGES):
            with _col:
                st.button(_name, key=f"nav_{_name}", on_click=go_to, args=(_name,),
                          type="primary" if st.session_state["nav"] == _name else "secondary",
                          use_container_width=True)

st.write("")
page = st.session_state["nav"]

# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
if page == "About":
    with st.container(key="page_card"):
        st.markdown('<div class="card-title">How it works</div>', unsafe_allow_html=True)
        st.markdown(
            "1. A Gradient Boosting model predicts your expected rent.\n"
            "2. You adjust your budget.\n"
            "3. K-Means assigns a cluster using Rent, Size, BHK, and Bathroom.\n"
            "4. Listings are filtered by city and budget within that cluster.\n"
            "5. If there are no results, you can explicitly allow searching other clusters."
        )
        st.caption("City is a search filter, not a K-Means input feature.")

elif page == "Data Insights":
    with st.container(key="page_card"):
        st.markdown('<div class="card-title">Cluster profiles</div>', unsafe_allow_html=True)
        render_cluster_table(cluster_profiles(df))

else:
    left, right = st.columns([1, 1.45], gap="large")

    # ---------------- Left: preferences ----------------
    with left:
        with st.container(key="prefs_card"):
            st.markdown('<div class="card-title">Property Preferences</div>'
                        '<div class="card-sub">Tell us your requirements and get personalized recommendations</div>',
                        unsafe_allow_html=True)
            st.selectbox(":material/location_on: City", CITIES, key="city")
            st.slider("Property size (sq ft)", min_value=100, max_value=10000, step=50,
                      key="size_slider", on_change=sync_size_slider)
            st.number_input("Exact size (sq ft)", min_value=100, max_value=10000, step=50,
                            key="size_number", on_change=sync_size_number, label_visibility="collapsed")
            b1, b2 = st.columns(2)
            with b1:
                st.select_slider(":material/bed: BHK", options=[1, 2, 3, 4, 5, 6], key="bhk")
            with b2:
                st.select_slider(":material/bathtub: Bathrooms", options=[1, 2, 3, 4, 5, 6], key="bathrooms")
            st.selectbox(":material/chair: Furnishing status", FURNISHING, key="furnishing")
            st.button(":material/search: Predict expected rent", type="primary",
                      use_container_width=True, on_click=on_predict)
            if "predict_error" in st.session_state:
                st.error("Rent prediction failed. Verify that the regression feature columns match training.")
                st.exception(st.session_state["predict_error"])

    # ---------------- Right: predicted rent + budget ----------------
    with right:
        with st.container(key="rent_card"):
            st.markdown('<div class="card-title">Predicted Monthly Rent</div>', unsafe_allow_html=True)
            request = st.session_state.get("request")
            if not request:
                st.markdown('<div class="empty-note">Enter your requirements and select '
                            '<b>Predict expected rent</b> to see an estimate and set your budget.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="big-rent">₹{inr(request["predicted"])}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="info-strip">'
                    f'<div><div class="k">Cluster</div><div class="v">{request["cluster"]}</div></div>'
                    f'<div><div class="k">City</div><div class="v">{request["city"]}</div></div>'
                    f'<div><div class="k">Property Size</div><div class="v">{request["size"]:,} sq ft</div></div>'
                    f'<div><div class="k">BHK / Bathroom</div><div class="v">{request["bhk"]} / {request["bathrooms"]}</div></div>'
                    f'</div>', unsafe_allow_html=True)

                predicted_default = max(0, int(round(request["predicted"])))
                budget_max = max(100000, int(np.ceil(predicted_default * 3 / 1000) * 1000))
                start = min(max(predicted_default, BUDGET_MIN), budget_max)
                st.session_state.setdefault("budget_slider", start)
                st.session_state.setdefault("budget_number", start)

                st.markdown('<div class="card-title" style="font-size:1.2rem !important">Adjust Budget (₹)</div>',
                            unsafe_allow_html=True)
                st.slider("Budget slider (₹/month)", min_value=BUDGET_MIN, max_value=budget_max, step=1000,
                          key="budget_slider", on_change=sync_budget_slider, label_visibility="collapsed")
                st.number_input("Maximum monthly budget (₹)", min_value=BUDGET_MIN, max_value=budget_max,
                                step=1000, key="budget_number", on_change=sync_budget_number,
                                label_visibility="collapsed")
                st.checkbox("Only show the selected furnishing status", value=False, key="enforce_furnishing")
                st.button(":material/search: Find Properties", type="primary",
                          use_container_width=True, on_click=on_find)
                st.caption("The predicted rent is a starting estimate, not a guarantee of market price.")

    # ---------------- Results ----------------
    result = st.session_state.get("search_result")
    if result:
        st.write("")
        with st.container(key="results_card"):
            st.markdown('<div class="card-title">Recommended Properties</div>', unsafe_allow_html=True)

            all_clusters = st.session_state["expand_clusters"]
            matches = search(result["city"], result["cluster"], result["budget"], result["bhk"],
                             result["bathrooms"], result["size"], result["furnishing_filter"],
                             all_clusters=all_clusters)
            total = min(len(matches), MAX_RESULTS)
            scope = "All clusters" if all_clusters else f"Cluster {result['cluster']}"
            st.markdown(
                f'<div class="chips"><span class="chip">{scope}</span>'
                f'<span class="chip"><b>{html.escape(result["city"])}</b></span>'
                f'<span class="chip">Budget <b>₹{inr(result["budget"])}</b>/month</span>'
                f'<span class="chip"><b>{len(matches)}</b> listings</span></div>',
                unsafe_allow_html=True)

            if not matches.empty:
                exact = matches[(matches["BHK_Diff"] == 0) & (matches["Bathroom_Diff"] == 0)]
                if exact.empty:
                    st.caption("No exact BHK-and-bathroom match. Showing the closest alternatives, "
                               "sorted by BHK, bathroom and area difference, then highest to lowest rent.")
                else:
                    st.caption("Sorted by closest BHK, bathroom and area match, then highest to lowest rent.")

                shown = min(st.session_state["show_n"], total)
                render_listings(matches, shown)
                st.markdown(f'<div class="foot">Showing {shown} of {total} listings</div>', unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                with m1:
                    if shown < total:
                        st.button("Show more", use_container_width=True, on_click=show_more)
                with m2:
                    st.download_button(":material/download: Download Excel",
                                       to_excel_bytes(to_export_frame(matches)),
                                       file_name="recommended_properties.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       use_container_width=True)
            else:
                st.markdown('<div class="empty-note">No properties match this cluster, city, and budget. '
                            'Raise your budget, change your preferences, or search the other clusters.</div>',
                            unsafe_allow_html=True)
                if not all_clusters:
                    st.button("Search other clusters in this city and budget", on_click=expand_clusters)
                else:
                    st.error("No listings found in any cluster for this city and budget.")


# ----------------------------------------------------------------------------
# Remember the current inputs so they survive a trip to another page
# ----------------------------------------------------------------------------
for _k in PERSISTED_KEYS:
    if _k in st.session_state:
        _saved_inputs[_k] = st.session_state[_k]
st.session_state["_saved_inputs"] = _saved_inputs
