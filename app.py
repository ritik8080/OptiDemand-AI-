"""
OptiDemand AI — E-Commerce Demand Forecasting & Inventory Optimization Platform
Built for Amazon India & Flipkart product catalogue with real-world SKUs.
Author  : Principal Full-Stack Python Engineer
Version : 2.0
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OptiDemand AI — E-Commerce Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS INJECTION
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Smooth scrolling on the whole page ─────────────────── */
    html {
        scroll-behavior: smooth !important;
    }
    /* Streamlit's main scroll container */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main .block-container {
        scroll-behavior: smooth !important;
    }

    /* ── Base ─────────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif !important;
    }

    /* ── Header & toolbar ────────────────────────────────────── */
    [data-testid="stHeader"] {
        background-color: rgba(8,15,26,0.97) !important;
        border-bottom: 1px solid #1E3A5F !important;
        backdrop-filter: blur(12px) !important;
    }
    /* Hide only the deploy button — NOT the whole toolbar */
    .stDeployButton, [data-testid="stToolbarActionButtonTooltip"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer    { visibility: hidden !important; }

    /* ── Sidebar ─────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080F1A 0%, #0F172A 100%) !important;
        border-right: 1px solid #1E3A5F !important;
    }
    /* Sidebar close (←) arrow button — visible & styled */
    [data-testid="stSidebarCollapseButton"] > button {
        background: rgba(30,58,95,0.6) !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 8px !important;
        opacity: 1 !important;
        visibility: visible !important;
        transition: all 0.2s !important;
    }
    [data-testid="stSidebarCollapseButton"] > button:hover {
        border-color: #38BDF8 !important;
        background: rgba(56,189,248,0.1) !important;
        box-shadow: 0 0 12px rgba(56,189,248,0.25) !important;
    }
    [data-testid="stSidebarCollapseButton"] > button svg {
        fill: #38BDF8 !important;
    }
    /* Sidebar open (►) button when sidebar is collapsed — visible & styled */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: linear-gradient(135deg, #1E293B, #0F2040) !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 0 10px 10px 0 !important;
        margin-top: 2px !important;
        box-shadow: 4px 0 16px rgba(56,189,248,0.2) !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="collapsedControl"]:hover {
        background: rgba(56,189,248,0.12) !important;
        box-shadow: 6px 0 24px rgba(56,189,248,0.35) !important;
    }
    [data-testid="collapsedControl"] svg,
    [data-testid="collapsedControl"] button svg {
        fill: #38BDF8 !important;
        color: #38BDF8 !important;
    }

    /* ── Typography ──────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 { font-family:'Inter','Segoe UI',Roboto,sans-serif; color:#F8FAFC !important; }
    p, span, div, label     { font-family:'Inter','Segoe UI',Roboto,sans-serif; }

    /* ── Metric Cards ─────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F2040 100%);
        border: 1px solid #1E3A5F; border-radius: 16px;
        padding: 1.4rem 1.6rem; height: 100%;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35), 0 2px 8px rgba(56,189,248,0.05);
        transition: all 0.25s cubic-bezier(.4,0,.2,1);
        position: relative; overflow: hidden;
    }
    .metric-card::before {
        content:''; position:absolute; top:0; left:0; right:0;
        height:3px; border-radius:16px 16px 0 0;
    }
    .metric-card.blue::before   { background:linear-gradient(90deg,#38BDF8,#0EA5E9); }
    .metric-card.green::before  { background:linear-gradient(90deg,#10B981,#059669); }
    .metric-card.amber::before  { background:linear-gradient(90deg,#F59E0B,#D97706); }
    .metric-card.red::before    { background:linear-gradient(90deg,#EF4444,#DC2626); }
    .metric-card.violet::before { background:linear-gradient(90deg,#8B5CF6,#7C3AED); }
    .metric-card:hover { transform:translateY(-4px); border-color:#38BDF8;
        box-shadow:0 16px 40px rgba(0,0,0,0.4),0 4px 12px rgba(56,189,248,0.15); }
    .metric-label { color:#64748B!important; font-size:0.78rem; font-weight:600;
        text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem; }
    .metric-value { font-size:2.1rem; font-weight:800; color:#F8FAFC!important;
        line-height:1.1; margin-bottom:0.45rem; }
    .metric-sub   { font-size:0.82rem; font-weight:500; color:#64748B!important; }
    .delta-up   { color:#10B981!important; font-weight:600; }
    .delta-down { color:#EF4444!important; font-weight:600; }
    .delta-neu  { color:#94A3B8!important; font-weight:600; }

    /* ── Status Badges ────────────────────────────────────────── */
    .badge { display:inline-block; padding:0.3rem 0.9rem; border-radius:9999px;
        font-size:0.72rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; }
    .badge-critical { background:rgba(239,68,68,.15);  color:#EF4444!important; border:1px solid rgba(239,68,68,.35); }
    .badge-warning  { background:rgba(245,158,11,.15); color:#F59E0B!important; border:1px solid rgba(245,158,11,.35); }
    .badge-optimal  { background:rgba(16,185,129,.15); color:#10B981!important; border:1px solid rgba(16,185,129,.35); }

    /* ── Platform Pills ───────────────────────────────────────── */
    .platform-pill { display:inline-block; padding:0.2rem 0.65rem; border-radius:9999px;
        font-size:0.68rem; font-weight:700; letter-spacing:0.04em; margin:0 0.2rem 0 0; }
    .pill-amazon   { background:rgba(255,153,0,.15);  color:#FF9900!important; border:1px solid rgba(255,153,0,.4); }
    .pill-flipkart { background:rgba(47,116,212,.15); color:#2F74D4!important; border:1px solid rgba(47,116,212,.4); }
    .pill-meesho   { background:rgba(167,70,220,.15); color:#A746DC!important; border:1px solid rgba(167,70,220,.4); }

    /* ── Product Info Card ────────────────────────────────────── */
    .product-card { background:linear-gradient(145deg,#1E293B,#0F2040);
        border:1px solid #1E3A5F; border-radius:12px;
        padding:1rem 1.2rem; margin-bottom:1rem; }
    .product-brand { color:#38BDF8!important; font-size:0.72rem; font-weight:700;
        text-transform:uppercase; letter-spacing:0.06em; }
    .product-name  { color:#F8FAFC!important; font-size:0.93rem; font-weight:700; margin:0.25rem 0; line-height:1.3; }
    .product-price { color:#10B981!important; font-size:1.25rem; font-weight:800; }
    .product-meta  { color:#64748B!important; font-size:0.73rem; margin-top:0.4rem; line-height:1.6; }

    /* ── Section Headers ─────────────────────────────────────── */
    .section-header { display:flex; align-items:center; gap:0.6rem;
        border-bottom:1px solid #1E3A5F; padding-bottom:0.6rem; margin-bottom:1.2rem; }
    .section-title    { font-size:1.1rem; font-weight:700; color:#F8FAFC!important; }
    .section-subtitle { font-size:0.82rem; color:#64748B!important; }

    /* ── Sticky info bar ──────────────────────────────────────── */
    .sticky-bar {
        position: sticky; top: 0; z-index: 900;
        background: linear-gradient(90deg, rgba(8,15,26,0.97), rgba(15,32,64,0.97));
        border-bottom: 1px solid #1E3A5F;
        padding: 0.5rem 1rem;
        display: flex; align-items: center; gap: 1rem;
        backdrop-filter: blur(12px);
        margin: -1rem -1rem 1rem -1rem;
        flex-wrap: wrap;
    }
    .sticky-chip {
        background: rgba(30,58,95,0.6);
        border: 1px solid #1E3A5F;
        border-radius: 8px;
        padding: 0.25rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #94A3B8 !important;
        display: inline-flex; align-items: center; gap: 0.3rem;
    }
    .sticky-chip b { color: #38BDF8 !important; }

    /* ── Tab Overrides ───────────────────────────────────────── */
    [data-testid="stTabs"] [data-baseweb="tab"] {
        color: #64748B;
        font-weight: 600;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom-color: #38BDF8 !important;
    }

    /* ── Dataframe ───────────────────────────────────────────── */
    .stDataFrame { border-radius:12px; overflow:hidden; border:1px solid #1E3A5F; }

    /* ── Sidebar sliders / widgets ───────────────────────────── */
    .stSlider [data-baseweb="slider"] div[role="slider"] { background:#38BDF8; }
    .stSelectbox label, .stSlider label, .stToggle label { color:#94A3B8!important; font-size:0.82rem; }

    /* ── Reset button ────────────────────────────────────────── */
    .reset-btn button {
        background: rgba(239,68,68,0.1) !important;
        border: 1px solid rgba(239,68,68,0.4) !important;
        color: #EF4444 !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        transition: all 0.2s !important;
    }
    .reset-btn button:hover {
        background: rgba(239,68,68,0.2) !important;
        box-shadow: 0 0 12px rgba(239,68,68,0.3) !important;
    }

    /* ── Search result cards ────────────────────────────────────── */
    .s-card {
        background: linear-gradient(145deg,#1E293B,#0F2040);
        border: 1px solid #1E3A5F; border-radius: 14px;
        padding: 1.1rem 1.2rem; height: 100%;
        transition: all 0.22s cubic-bezier(.4,0,.2,1);
    }
    .s-card:hover { transform:translateY(-3px); border-color:#38BDF8;
        box-shadow:0 12px 32px rgba(0,0,0,0.35),0 2px 8px rgba(56,189,248,0.12); }
    .s-card-brand { color:#38BDF8!important; font-size:0.7rem; font-weight:700;
        text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.25rem; }
    .s-card-name  { color:#F8FAFC!important; font-size:0.9rem; font-weight:700;
        line-height:1.35; margin-bottom:0.2rem; }
    .s-card-cat   { color:#64748B!important; font-size:0.72rem; margin-bottom:0.6rem; }
    .s-divider    { border:none; border-top:1px solid #1E3A5F; margin:0.55rem 0; }
    .s-price-row  { display:flex; justify-content:space-between;
        align-items:center; padding:0.18rem 0; }
    .s-platform   { font-size:0.72rem; font-weight:600; color:#64748B!important; }
    .s-price      { font-size:0.85rem; font-weight:700; color:#F8FAFC!important; }
    .s-best-price { color:#10B981!important; }
    .s-amazon     { color:#FF9900!important; }
    .s-flipkart   { color:#2F74D4!important; }
    .s-meesho     { color:#A746DC!important; }
    .s-rating     { color:#F59E0B!important; font-size:0.78rem; font-weight:600; }
    .s-demand     { color:#64748B!important; font-size:0.7rem; margin-top:0.15rem; }
</style>
""", unsafe_allow_html=True)

# ── Floating sidebar button injected via components.v1.html (scripts actually run) ──
components.html("""
<!DOCTYPE html>
<html>
<head><style>
  body { margin:0; background:transparent; overflow:hidden; }
  #floatBtn {
    position: fixed;
    bottom: 28px;
    left: 14px;
    z-index: 2147483647;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1E293B, #0F172A);
    border: 2px solid #38BDF8;
    color: #38BDF8;
    font-size: 1.4rem;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(56,189,248,0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
    font-family: sans-serif;
    line-height: 1;
  }
  #floatBtn:hover {
    box-shadow: 0 6px 28px rgba(56,189,248,0.65);
    transform: scale(1.12);
    background: rgba(56,189,248,0.15);
  }
  #floatBtn.hidden { opacity:0; pointer-events:none; }
  #tooltip {
    position: fixed;
    bottom: 28px;
    left: 72px;
    background: #1E293B;
    border: 1px solid #38BDF8;
    border-radius: 8px;
    padding: 6px 12px;
    color: #38BDF8;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: sans-serif;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
    z-index: 2147483646;
  }
</style></head>
<body>
  <div id="floatBtn" title="Open Controls Panel">&#9776;</div>
  <div id="tooltip">Open Controls Panel</div>
  <script>
    var btn = document.getElementById('floatBtn');
    var tip = document.getElementById('tooltip');

    // Hover tooltip
    btn.addEventListener('mouseenter', function() { tip.style.opacity = '1'; });
    btn.addEventListener('mouseleave', function() { tip.style.opacity = '0'; });

    btn.addEventListener('click', function() {
      // Target the parent Streamlit window
      var parentDoc = window.parent.document;
      // Try collapsedControl (the expand arrow when sidebar is hidden)
      var trigger = parentDoc.querySelector('[data-testid="collapsedControl"] button');
      if (!trigger) trigger = parentDoc.querySelector('[data-testid="collapsedControl"]');
      // Also try the sidebar toggle in the header
      if (!trigger) trigger = parentDoc.querySelector('button[kind="header"]');
      if (trigger) {
        trigger.click();
      }
    });

    // Sync visibility: show button only when sidebar is collapsed
    function syncVisibility() {
      var parentDoc = window.parent.document;
      var collapsed = parentDoc.querySelector('[data-testid="collapsedControl"]');
      if (collapsed) {
        var display = window.parent.getComputedStyle(collapsed).display;
        if (display === 'none') {
          btn.classList.add('hidden');
        } else {
          btn.classList.remove('hidden');
        }
      }
    }
    setInterval(syncVisibility, 600);
    syncVisibility();
  </script>
</body>
</html>
""", height=0, scrolling=False)


# ─────────────────────────────────────────────────────────────────
# MODULE A ─ REAL PRODUCT CATALOGUE  (Amazon India / Flipkart)
# ─────────────────────────────────────────────────────────────────
PRODUCT_CATALOGUE: list[dict] = [
    # ── Electronics ───────────────────────────────────────────────
    {
        "id": "EL-SAM-S24U",
        "name": "Samsung Galaxy S24 Ultra 5G",
        "brand": "Samsung",
        "cat": "Electronics",
        "asin": "B0CRN4GQVB",
        "base_price": 129999,          # INR
        "rating": 4.4,
        "lead_time": 3,
        "platform": "Amazon & Flipkart",
        "base_demand": 45,
    },
    {
        "id": "EL-OPL-12R",
        "name": "OnePlus 12R 5G (Cool Blue, 256GB)",
        "brand": "OnePlus",
        "cat": "Electronics",
        "asin": "B0CTMQHV8Y",
        "base_price": 39999,
        "rating": 4.3,
        "lead_time": 3,
        "platform": "Amazon",
        "base_demand": 120,
    },
    {
        "id": "EL-BOAT-RL550",
        "name": "boAt Rockerz 550 BT Headphones",
        "brand": "boAt",
        "cat": "Electronics",
        "asin": "B08JM7FZDY",
        "base_price": 1299,
        "rating": 4.1,
        "lead_time": 5,
        "platform": "Amazon & Flipkart",
        "base_demand": 380,
    },
    {
        "id": "EL-SONY-XM5",
        "name": "Sony WH-1000XM5 ANC Headphones",
        "brand": "Sony",
        "cat": "Electronics",
        "asin": "B09XS7JWHH",
        "base_price": 26990,
        "rating": 4.6,
        "lead_time": 7,
        "platform": "Amazon",
        "base_demand": 60,
    },
    # ── Apparel ────────────────────────────────────────────────────
    {
        "id": "AP-LEV-501",
        "name": "Levi's Men's 501 Original Fit Jeans",
        "brand": "Levi's",
        "cat": "Apparel",
        "asin": "B07CMBVHBM",
        "base_price": 4499,
        "rating": 4.2,
        "lead_time": 12,
        "platform": "Amazon & Flipkart",
        "base_demand": 95,
    },
    {
        "id": "AP-PUM-RS",
        "name": "Puma Men's Softride Enzo Running Shoes",
        "brand": "Puma",
        "cat": "Apparel",
        "asin": "B09NFRXBGZ",
        "base_price": 4999,
        "rating": 4.3,
        "lead_time": 10,
        "platform": "Flipkart",
        "base_demand": 85,
    },
    {
        "id": "AP-PE-SHIRT",
        "name": "Peter England Men's Slim Fit Formal Shirt",
        "brand": "Peter England",
        "cat": "Apparel",
        "asin": "B07ZDNRWM9",
        "base_price": 1199,
        "rating": 4.0,
        "lead_time": 15,
        "platform": "Amazon & Flipkart",
        "base_demand": 150,
    },
    {
        "id": "AP-WC-WCJKT",
        "name": "Wildcraft Unisex Nylon Packable Jacket",
        "brand": "Wildcraft",
        "cat": "Apparel",
        "asin": "B07RQSVXK1",
        "base_price": 2499,
        "rating": 3.9,
        "lead_time": 20,
        "platform": "Amazon",
        "base_demand": 65,
    },
    # ── Home & Kitchen ─────────────────────────────────────────────
    {
        "id": "HK-PHIL-AF",
        "name": "Philips HD9252 Air Fryer 1400W",
        "brand": "Philips",
        "cat": "Home & Kitchen",
        "asin": "B07VMHBXFS",
        "base_price": 6995,
        "rating": 4.4,
        "lead_time": 7,
        "platform": "Amazon & Flipkart",
        "base_demand": 75,
    },
    {
        "id": "HK-PRES-MG",
        "name": "Prestige Iris 750W Mixer Grinder (4 Jars)",
        "brand": "Prestige",
        "cat": "Home & Kitchen",
        "asin": "B009RS9SAQ",
        "base_price": 2295,
        "rating": 4.2,
        "lead_time": 5,
        "platform": "Amazon & Flipkart",
        "base_demand": 110,
    },
    # ── Beauty & Personal Care ─────────────────────────────────────
    {
        "id": "BP-MRE-VCS",
        "name": "Mamaearth Vitamin C Face Serum 30ml",
        "brand": "Mamaearth",
        "cat": "Beauty & Personal Care",
        "asin": "B08CBZC85Z",
        "base_price": 699,
        "rating": 4.3,
        "lead_time": 4,
        "platform": "Amazon & Flipkart",
        "base_demand": 260,
    },
    {
        "id": "BP-LKM-MATT",
        "name": "Lakmé Absolute Matte Melt Lipstick",
        "brand": "Lakmé",
        "cat": "Beauty & Personal Care",
        "asin": "B07VCMLYGY",
        "base_price": 575,
        "rating": 4.1,
        "lead_time": 4,
        "platform": "Amazon & Flipkart",
        "base_demand": 310,
    },
    {
        "id": "BP-DP-BIOF",
        "name": "Dot & Key Barrier Repair Moisturizer",
        "brand": "Dot & Key",
        "cat": "Beauty & Personal Care",
        "asin": "B09P8GHHWJ",
        "base_price": 645,
        "rating": 4.5,
        "lead_time": 4,
        "platform": "Flipkart",
        "base_demand": 195,
    },
    {
        "id": "EL-FIR-43TV",
        "name": "AmazonBasics 43\" Full HD Smart Fire TV",
        "brand": "AmazonBasics",
        "cat": "Electronics",
        "asin": "B09G9HD1C8",
        "base_price": 27999,
        "rating": 4.0,
        "lead_time": 10,
        "platform": "Amazon",
        "base_demand": 52,
    },
]

SKU_MAP: dict = {p["id"]: p for p in PRODUCT_CATALOGUE}


# ─────────────────────────────────────────────────────────────────
# MODULE A ─ REALISTIC DATA GENERATION WITH INDIAN FESTIVE EVENTS
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚙️  Generating 2-year synthetic catalogue data…")
def generate_data() -> tuple[pd.DataFrame, list[dict]]:
    """
    Generates 2 years of daily sales data per SKU with:
    - Indian festive season boosts (Diwali, Big Billion Days, Republic Day Sale)
    - Category-specific seasonality patterns
    - Weekend / weekday splits
    - Promotional discount events
    - Price elasticity model
    - Random marketing spend
    """
    np.random.seed(42)
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=730)
    dates      = pd.date_range(start=start_date, end=end_date, freq='D')

    # ── Indian festive event windows (month, day_start, day_end, multiplier)
    FESTIVE_EVENTS: list[tuple] = [
        # Big Billion Days (Oct 7-12 approx)
        (10,  7, 12, 4.5),
        # Great Indian Festival - Amazon (Oct 8-15 approx)
        (10,  8, 15, 4.0),
        # Diwali week (late Oct/early Nov — approx Oct 28 - Nov 4)
        (10, 28, 31, 3.8), (11,  1,  4, 3.8),
        # Republic Day Sale (Jan 21-25)
        (1,  21, 25, 2.2),
        # End of Season Sales (Jan 1-10)
        (1,   1, 10, 2.5),
        # Holi Sale (March – approx March 20-26)
        (3,  20, 26, 1.8),
        # End of Season Sale June (Jun 15-25)
        (6,  15, 25, 2.0),
    ]

    def festive_multiplier(d: pd.Timestamp) -> float:
        for month, d1, d2, mult in FESTIVE_EVENTS:
            if d.month == month and d1 <= d.day <= d2:
                return mult
        return 1.0

    # Category-specific weekend sensitivity
    WEEKEND_FACTOR: dict = {
        "Electronics":          1.50,
        "Apparel":              1.40,
        "Home & Kitchen":       1.25,
        "Beauty & Personal Care": 1.35,
    }
    # Category-specific off-peak weekday factor
    WEEKDAY_FACTOR: dict = {
        "Electronics":          0.80,
        "Apparel":              0.85,
        "Home & Kitchen":       0.88,
        "Beauty & Personal Care": 0.90,
    }

    records: list[dict] = []
    for sku in PRODUCT_CATALOGUE:
        base_d = sku["base_demand"]
        cat    = sku["cat"]
        for d in dates:
            dow   = d.weekday()                                # 0=Mon … 6=Sun
            month = d.month

            # Weekend effect
            weekend_eff = WEEKEND_FACTOR[cat] if dow >= 5 else WEEKDAY_FACTOR[cat]

            # Festive multiplier
            fest_eff = festive_multiplier(d)

            # Mild Q4 e-commerce boost (Sep-Oct outside festive)
            general_q4 = 1.25 if month in [9, 10, 11, 12] else 1.0
            # January winter / clothing boost for Apparel
            winter_eff = 1.3 if cat == "Apparel" and month in [12, 1, 2] else 1.0
            # Summer kitchen / AC boost for Home
            summer_eff = 1.2 if cat == "Home & Kitchen" and month in [4, 5, 6] else 1.0

            # Promotions: 15% chance on any day; 30% during festive
            promo_prob    = 0.30 if fest_eff > 1 else 0.15
            is_promo      = int(np.random.choice([0, 1], p=[1 - promo_prob, promo_prob]))
            promo_disc    = np.random.uniform(0.10, 0.40) if is_promo else 0.0
            current_price = sku["base_price"] * (1 - promo_disc)

            # Price elasticity: −2 for electronics, −1.5 for others
            elasticity    = -2.0 if cat == "Electronics" else -1.5
            price_eff     = 1 + elasticity * (-(promo_disc))   # discount → demand up

            # Marketing spend (INR/day)
            mktg_spend    = np.random.uniform(5000, 40000) if fest_eff > 1 else np.random.uniform(500, 8000)
            mktg_eff      = 1 + (mktg_spend / 50000) * 0.30

            noise         = np.random.normal(1.0, 0.10)

            demand = max(0, int(
                base_d * weekend_eff * fest_eff * general_q4 *
                winter_eff * summer_eff * price_eff * mktg_eff * noise
            ))

            records.append({
                "Date":          d,
                "SKU_ID":        sku["id"],
                "SKU_Name":      sku["name"],
                "Brand":         sku["brand"],
                "Category":      cat,
                "Platform":      sku["platform"],
                "Base_Price":    sku["base_price"],
                "Current_Price": current_price,
                "Is_Promo":      is_promo,
                "Promo_Discount":promo_disc,
                "Marketing_Spend": mktg_spend,
                "Lead_Time":     sku["lead_time"],
                "Rating":        sku["rating"],
                "Demand":        demand,
            })

    df = pd.DataFrame(records)

    # ── Feature Engineering
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"]     = df["Date"].dt.month
    df["Is_Weekend"]= df["DayOfWeek"].isin([5, 6]).astype(int)

    df = df.sort_values(["SKU_ID", "Date"]).reset_index(drop=True)
    df["Demand_Lag_1"]     = df.groupby("SKU_ID")["Demand"].shift(1)
    df["Demand_Lag_7"]     = df.groupby("SKU_ID")["Demand"].shift(7)
    df["Rolling_Mean_7"]   = df.groupby("SKU_ID")["Demand"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["Rolling_Mean_30"]  = df.groupby("SKU_ID")["Demand"].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df["Rolling_Std_7"]    = df.groupby("SKU_ID")["Demand"].transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))

    df = df.dropna().reset_index(drop=True)
    return df, PRODUCT_CATALOGUE


# ─────────────────────────────────────────────────────────────────
# MODULE A ─ QUANTILE GRADIENT BOOSTING FORECAST ENGINE
# ─────────────────────────────────────────────────────────────────
FEATURES = [
    "Current_Price", "Promo_Discount", "Marketing_Spend",
    "DayOfWeek", "Month", "Is_Weekend",
    "Demand_Lag_1", "Demand_Lag_7", "Rolling_Mean_7", "Rolling_Mean_30",
]

FEATURE_DISPLAY = {
    "Current_Price":   "Unit Price (INR)",
    "Promo_Discount":  "Promotional Discount",
    "Marketing_Spend": "Marketing Ad Spend",
    "DayOfWeek":       "Day of Week",
    "Month":           "Festive / Seasonal Month",
    "Is_Weekend":      "Weekend Effect",
    "Demand_Lag_1":    "Yesterday Demand (T-1)",
    "Demand_Lag_7":    "Last Week Demand (T-7)",
    "Rolling_Mean_7":  "7-Day Rolling Average",
    "Rolling_Mean_30": "30-Day Rolling Average",
}

@st.cache_resource(show_spinner="🤖  Training quantile regression models…")
def train_models(df: pd.DataFrame) -> dict:
    """
    Per-SKU Gradient Boosting quantile regression models.
    Produces P10 (lower), P50 (median/point), P90 (upper) bands.
    """
    models: dict = {}
    for sku_id in df["SKU_ID"].unique():
        sdf = df[df["SKU_ID"] == sku_id]
        X, y = sdf[FEATURES], sdf["Demand"]

        m_median = GradientBoostingRegressor(loss="quantile", alpha=0.50, n_estimators=120, max_depth=4, random_state=42)
        m_lower  = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=120, max_depth=4, random_state=42)
        m_upper  = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=120, max_depth=4, random_state=42)
        m_median.fit(X, y)
        m_lower.fit(X, y)
        m_upper.fit(X, y)
        models[sku_id] = {
            "median": m_median,
            "lower":  m_lower,
            "upper":  m_upper,
            "importance": m_median.feature_importances_,
        }
    return models


# ─────────────────────────────────────────────────────────────────
# HELPER: AUTO-REGRESSIVE FORWARD FORECAST
# ─────────────────────────────────────────────────────────────────
def build_forecast(
    sku_df: pd.DataFrame,
    model_set: dict,
    horizon: int,
    sim_price: float,
    sim_promo_disc: float,
    sim_mktg: float,
) -> pd.DataFrame:
    last_date  = sku_df["Date"].max()
    fut_dates  = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq="D")

    lag1  = float(sku_df["Demand"].iloc[-1])
    lag7  = float(sku_df["Demand"].iloc[-7])
    roll7 = float(sku_df["Demand"].iloc[-7:].mean())
    roll30= float(sku_df["Demand"].iloc[-30:].mean())

    preds, lowers, uppers = [], [], []
    for d in fut_dates:
        row = {
            "Current_Price":   sim_price,
            "Promo_Discount":  sim_promo_disc,
            "Marketing_Spend": sim_mktg,
            "DayOfWeek":       d.weekday(),
            "Month":           d.month,
            "Is_Weekend":      int(d.weekday() >= 5),
            "Demand_Lag_1":    lag1,
            "Demand_Lag_7":    lag7,
            "Rolling_Mean_7":  roll7,
            "Rolling_Mean_30": roll30,
        }
        Xp    = pd.DataFrame([row])
        pred  = max(0.0, model_set["median"].predict(Xp)[0])
        lower = max(0.0, model_set["lower"].predict(Xp)[0])
        upper = max(0.0, model_set["upper"].predict(Xp)[0])
        preds.append(pred);  lowers.append(lower);  uppers.append(upper)

        lag7  = lag1
        lag1  = pred
        roll7 = (roll7 * 6 + pred) / 7
        roll30= (roll30 * 29 + pred) / 30

    return pd.DataFrame({"Date": fut_dates, "Forecast": preds, "Lower": lowers, "Upper": uppers})


# ─────────────────────────────────────────────────────────────────
# HELPER: INR FORMATTER
# ─────────────────────────────────────────────────────────────────
def inr(value: float, compact: bool = False) -> str:
    """Returns a formatted INR string. e.g. ₹1,29,999 or ₹12.4L"""
    if compact:
        if value >= 1e7:  return f"₹{value/1e7:.1f}Cr"
        if value >= 1e5:  return f"₹{value/1e5:.1f}L"
        if value >= 1e3:  return f"₹{value/1e3:.0f}K"
        return f"₹{value:.0f}"
    # Indian comma-sep formatting
    s = f"{int(value):,}"
    return f"₹{s}"


# ─────────────────────────────────────────────────────────────────
# PLATFORM PRICING SIMULATION
# Deterministic per-platform price variation from base price
# ─────────────────────────────────────────────────────────────────
def _platform_prices(p: dict) -> dict:
    """
    Returns per-platform prices for Amazon, Flipkart, Meesho.
    Seeded from product ID so prices are stable on every rerun.
    """
    rng = np.random.RandomState(abs(hash(p["id"])) % (2**31))
    bp  = p["base_price"]
    prices: dict = {}
    if "Amazon"   in p["platform"]: prices["Amazon"]   = int(bp * rng.uniform(0.97, 1.03))
    if "Flipkart" in p["platform"]: prices["Flipkart"] = int(bp * rng.uniform(0.95, 1.05))
    if bp < 3000:                   prices["Meesho"]   = int(bp * rng.uniform(0.88, 0.98))
    return prices


CAT_EMOJI: dict = {
    "Electronics":            "📱",
    "Apparel":                "👔",
    "Home & Kitchen":         "🍳",
    "Beauty & Personal Care": "💄",
}

PLATFORM_COLOR: dict = {
    "Amazon":   ("s-amazon",   "🟠"),
    "Flipkart": ("s-flipkart", "🔵"),
    "Meesho":   ("s-meesho",   "🟣"),
}


# ─────────────────────────────────────────────────────────────────
# PRODUCT SEARCH SECTION
# ─────────────────────────────────────────────────────────────────
def render_search_section(df: pd.DataFrame) -> None:
    """
    Full-text multi-platform product search across the 14-SKU catalogue.
    Compares prices on Amazon India, Flipkart and Meesho with
    stock risk badges and 30-day demand metrics.
    """
    st.markdown("""
    <div class="section-header">
        <span style="font-size:1.6rem;">🔎</span>
        <div>
            <div class="section-title">Multi-Platform Product Search</div>
            <div class="section-subtitle">
                Compare prices across Amazon India · Flipkart · Meesho —
                with live demand trends, stock risk &amp; platform best-price highlights
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Search input row ──────────────────────────────────────────
    col_q, col_cat = st.columns([3, 1])
    with col_q:
        query = st.text_input(
            label="__search__",
            placeholder="🔎  Search by product, brand or category  (e.g. 'Samsung', 'boAt', 'Air Fryer')",
            label_visibility="collapsed",
            key="global_search_query",
        )
    with col_cat:
        cat_filter = st.selectbox(
            "Category Filter",
            ["All Categories"] + sorted({p["cat"] for p in PRODUCT_CATALOGUE}),
            label_visibility="collapsed",
            key="search_category_filter",
        )

    # ── Filter ────────────────────────────────────────────────────
    q = query.strip().lower()
    results = [
        p for p in PRODUCT_CATALOGUE
        if (not q or any(q in str(v).lower() for v in [p["name"], p["brand"], p["cat"], p["id"]]))
        and (cat_filter == "All Categories" or p["cat"] == cat_filter)
    ]

    if not results:
        st.markdown(f"""
        <div class="no-results">
            <div style="font-size:3.5rem; margin-bottom:0.5rem;">🔍</div>
            <div style="color:#64748B; font-size:1rem;">
                No products found for <b style="color:#38BDF8">"{query}"</b>
            </div>
            <div style="color:#334155; font-size:0.8rem; margin-top:0.3rem;">
                Try: 'Samsung', 'boAt', 'Mamaearth', 'Air Fryer', 'Electronics'
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#1E3A5F;'>", unsafe_allow_html=True)
        return

    # ── Result count ──────────────────────────────────────────────
    result_label = f"<b style='color:#38BDF8'>{len(results)}</b> products"
    if q:
        result_label += f" matching <b style='color:#F8FAFC'>'{query}'</b>"
    st.markdown(f"<p style='color:#64748B;font-size:0.82rem;margin-bottom:1rem;'>Found {result_label}</p>",
                unsafe_allow_html=True)

    # ── 30-day demand snapshot ────────────────────────────────────
    last_30 = df[df["Date"] >= (df["Date"].max() - timedelta(days=30))]
    COLS_PER_ROW = 3

    for row_start in range(0, len(results), COLS_PER_ROW):
        row_prods = results[row_start: row_start + COLS_PER_ROW]
        cols = st.columns(COLS_PER_ROW)

        for col, prod in zip(cols, row_prods):
            plat_prices = _platform_prices(prod)
            best_plat   = min(plat_prices, key=plat_prices.get)

            # Demand stats
            psku = last_30[last_30["SKU_ID"] == prod["id"]]
            avg_d    = int(psku["Demand"].mean())   if not psku.empty else 0
            promo_ct = int(psku["Is_Promo"].sum())  if not psku.empty else 0

            # Stock status
            sdf   = df[df["SKU_ID"] == prod["id"]]
            s_avg = sdf["Demand"].iloc[-30:].mean() if not sdf.empty else 1
            s_std = sdf["Demand"].iloc[-30:].std()  if not sdf.empty else 1
            s_ss  = int(1.65 * s_std * np.sqrt(prod["lead_time"]))
            s_oh  = int(s_avg * prod["lead_time"] * 1.5)
            if s_oh <= s_ss:
                stock_html = '<span class="badge badge-critical">Critical Stock</span>'
            elif s_oh <= s_ss * 2.5:
                stock_html = '<span class="badge badge-warning">Low Stock</span>'
            else:
                stock_html = '<span class="badge badge-optimal">In Stock</span>'

            # Rating
            full  = int(prod["rating"])
            stars = "★" * full + "☆" * (5 - full)

            # Build platform price rows
            price_rows_html = ""
            for plat, price in sorted(plat_prices.items()):
                cls_name, emoji = PLATFORM_COLOR.get(plat, ("", ""))
                is_best    = plat == best_plat
                price_cls  = "s-best-price" if is_best else cls_name
                best_badge = " &nbsp;<span style='font-size:0.62rem;background:rgba(16,185,129,.15);color:#10B981;padding:0.1rem 0.4rem;border-radius:4px;font-weight:700;'>BEST</span>" if is_best else ""
                price_rows_html += f"""
                <div class="s-price-row">
                    <span class="s-platform {cls_name}">{emoji} {plat}{best_badge}</span>
                    <span class="s-price {price_cls}">₹{price:,}</span>
                </div>"""

            with col:
                cat_em = CAT_EMOJI.get(prod["cat"], "📦")
                st.markdown(f"""
                <div class="s-card">
                    <div class="s-card-brand">{cat_em} {prod['brand']}</div>
                    <div class="s-card-name">{prod['name']}</div>
                    <div class="s-card-cat">
                        {prod['cat']} &nbsp;·&nbsp;
                        <code style="color:#38BDF8;font-size:0.65rem;">{prod['asin']}</code>
                    </div>
                    <hr class="s-divider">
                    {price_rows_html}
                    <hr class="s-divider">
                    <div style="display:flex;justify-content:space-between;align-items:flex-end;">
                        <div>
                            <div class="s-rating">{stars} {prod['rating']}</div>
                            <div class="s-demand" style="margin-top:0.2rem;">
                                📦 {avg_d} u/day &nbsp;·&nbsp; 🏷️ {promo_ct} promo days
                            </div>
                            <div class="s-demand">⏱ Lead time: {prod['lead_time']}d</div>
                        </div>
                        <div style="text-align:right;">
                            {stock_html}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E3A5F;margin:1.5rem 0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    df, skus = generate_data()
    models   = train_models(df)

    # ── PAGE HEADER — rich navbar with project KPIs ──────────────
    now_str   = datetime.now().strftime("%d %b %Y, %I:%M %p")
    date_min  = df["Date"].min().strftime("%d %b %Y")
    date_max  = df["Date"].max().strftime("%d %b %Y")
    n_records = f"{len(df):,}"
    n_skus    = df["SKU_ID"].nunique()
    n_cats    = df["Category"].nunique()
    total_rev_all = (df["Current_Price"] * df["Demand"]).sum()
    rev_label_all = f"₹{total_rev_all/1e7:.1f}Cr" if total_rev_all >= 1e7 else f"₹{total_rev_all/1e5:.1f}L"

    st.markdown(f"""
    <!-- ── NAVBAR ──────────────────────────────────────── -->
    <div style="display:flex;align-items:center;justify-content:space-between;
                gap:1rem;margin-bottom:0;flex-wrap:wrap;">

      <!-- Brand -->
      <div style="display:flex;align-items:center;gap:0.85rem;">
        <div style="background:linear-gradient(135deg,#1E3A5F,#0F2040);
                    border:1px solid #38BDF8;border-radius:12px;
                    width:44px;height:44px;display:flex;
                    align-items:center;justify-content:center;
                    box-shadow:0 0 18px rgba(56,189,248,0.25);flex-shrink:0;">
          <span style="font-size:1.4rem;">&#128640;</span>
        </div>
        <div>
          <h1 style="margin:0;font-size:1.65rem;font-weight:800;
                     background:linear-gradient(90deg,#38BDF8,#10B981);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            OptiDemand AI
          </h1>
          <p style="margin:0;color:#64748B;font-size:0.78rem;font-weight:500;">
            E-Commerce Demand Forecasting &amp; Inventory Intelligence
            &nbsp;&middot;&nbsp; Amazon India &amp; Flipkart
          </p>
        </div>
      </div>

      <!-- Nav KPI pills -->
      <div style="display:flex;align-items:center;gap:0.45rem;flex-wrap:wrap;">
        <span style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.35);
                     border-radius:8px;padding:0.3rem 0.75rem;font-size:0.7rem;
                     font-weight:700;color:#10B981;letter-spacing:0.04em;">
          &#128994; LIVE
        </span>
        <span style="background:rgba(56,189,248,0.08);border:1px solid #1E3A5F;
                     border-radius:8px;padding:0.3rem 0.75rem;font-size:0.7rem;
                     color:#38BDF8;font-weight:600;">
          &#128202; {n_skus} SKUs &nbsp;|&nbsp; {n_cats} Categories
        </span>
        <span style="background:rgba(56,189,248,0.08);border:1px solid #1E3A5F;
                     border-radius:8px;padding:0.3rem 0.75rem;font-size:0.7rem;
                     color:#64748B;">
          &#128197; {date_min} &rarr; {date_max}
        </span>
        <span style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.35);
                     border-radius:8px;padding:0.3rem 0.75rem;font-size:0.7rem;
                     color:#8B5CF6;font-weight:600;">
          &#128202; {n_records} Records
        </span>
        <span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);
                     border-radius:8px;padding:0.3rem 0.75rem;font-size:0.7rem;
                     color:#F59E0B;font-weight:600;">
          &#128176; {rev_label_all} Synthetic GMV
        </span>
        <span style="background:rgba(30,58,95,0.5);border:1px solid #1E3A5F;
                     border-radius:8px;padding:0.3rem 0.75rem;font-size:0.7rem;
                     color:#64748B;">
          &#128336; {now_str}
        </span>
      </div>
    </div>

    <!-- ── Sub-nav: Model & Data pipeline info ──────────────── -->
    <div style="display:flex;align-items:center;gap:0.6rem;margin-top:0.6rem;
                flex-wrap:wrap;padding:0.45rem 0.75rem;
                background:rgba(15,32,64,0.4);border-radius:10px;
                border:1px solid #1E3A5F;">
      <span style="color:#334155;font-size:0.68rem;font-weight:700;
                   text-transform:uppercase;letter-spacing:0.06em;">&#128736; Model Pipeline:</span>
      <span style="font-size:0.7rem;color:#94A3B8;">&#129302; Gradient Boosting Quantile Regression (P10 / P50 / P90)</span>
      <span style="color:#1E3A5F;">|</span>
      <span style="font-size:0.7rem;color:#94A3B8;">&#128200; Features: Price &middot; Promo &middot; Ad Spend &middot; Lag-1 &middot; Lag-7 &middot; Rolling Avg</span>
      <span style="color:#1E3A5F;">|</span>
      <span style="font-size:0.7rem;color:#94A3B8;">&#127987; Service Level: 95% &nbsp;(Z&thinsp;=&thinsp;1.65)</span>
      <span style="color:#1E3A5F;">|</span>
      <span style="font-size:0.7rem;color:#10B981;font-weight:600;">&#9679; Per-SKU models trained</span>
    </div>

    <hr style="border-color:#1E3A5F;margin:0.75rem 0 1rem 0;">
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────
    with st.sidebar:
        # ── Brand header bar ───────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(180deg,#0F1F35,#080F1A);
                    padding:1.1rem 1.2rem 0.9rem;border-bottom:1px solid #1E3A5F;
                    margin-bottom:0.25rem;">
            <div style="display:flex;align-items:center;gap:0.6rem;">
                <span style="font-size:1.5rem;">🛒</span>
                <div>
                    <div style="font-size:1rem;font-weight:800;
                                background:linear-gradient(90deg,#38BDF8,#10B981);
                                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                        OptiDemand AI
                    </div>
                    <div style="color:#334155;font-size:0.65rem;font-weight:500;
                                letter-spacing:0.05em;text-transform:uppercase;margin-top:0.1rem;">
                        Enterprise Intelligence
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Product Filters ────────────────────────────────────
        st.markdown("""<p style="color:#94A3B8;font-size:0.7rem;font-weight:700;
            text-transform:uppercase;letter-spacing:0.08em;
            margin:0.9rem 0 0.35rem;padding:0 0.2rem;">🔍 Product Filters</p>""",
            unsafe_allow_html=True)
        categories   = ["All"] + sorted(df["Category"].unique().tolist())
        selected_cat = st.selectbox("Category", categories, key="sb_cat_v2")

        if selected_cat == "All":
            avail_skus = df["SKU_ID"].unique().tolist()
        else:
            avail_skus = df[df["Category"] == selected_cat]["SKU_ID"].unique().tolist()

        sku_labels = {
            sid: f"{SKU_MAP[sid]['brand']} — {SKU_MAP[sid]['name'][:22]}…"
            for sid in avail_skus
        }
        selected_sku = st.selectbox(
            "Target SKU", avail_skus,
            format_func=lambda x: sku_labels.get(x, x),
            key="sb_sku_v2"
        )

        # ── Forecast Settings ──────────────────────────────────
        st.markdown("""<p style="color:#94A3B8;font-size:0.7rem;font-weight:700;
            text-transform:uppercase;letter-spacing:0.08em;
            margin:0.9rem 0 0.35rem;padding:0 0.2rem;">📅 Forecast Settings</p>""",
            unsafe_allow_html=True)
        horizon    = st.select_slider("Horizon (Days)", options=[7, 14, 30, 60], value=30, key="sb_horizon_v2")
        show_bands = st.toggle("Show 80% Confidence Band", value=True, key="sb_bands_v2")

        st.markdown("<hr style='border-color:#1E3A5F;margin:0.85rem 0;'>", unsafe_allow_html=True)

        # ── What-If Simulator ──────────────────────────────────
        st.markdown("""<p style="color:#94A3B8;font-size:0.7rem;font-weight:700;
            text-transform:uppercase;letter-spacing:0.08em;
            margin-bottom:0.25rem;padding:0 0.2rem;">⚙ Scenario Simulator</p>""",
            unsafe_allow_html=True)
        st.markdown("<p style='color:#334155;font-size:0.72rem;margin-bottom:0.6rem;line-height:1.4;'>Adjust levers to instantly re-run the AI forecast.</p>",
                    unsafe_allow_html=True)

        sim_price_pct  = st.slider("Price Adj. (%)", -30, 30, 0, step=5, key="sb_price_v2")
        sim_promo_on   = st.toggle("Activate Promotion", key="sb_promo_on_v2")
        sim_promo_rate = st.slider("Discount Rate (%)", 0, 50, 0, step=5,
                                   disabled=not sim_promo_on, key="sb_promo_rate_v2")
        sim_mktg       = st.slider("Ad Spend (₹/day)", 0, 50000, 5000, step=1000, key="sb_mktg_v2")

        # ── Reset Scenario button ──────────────────────────────
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        if st.button("↺ Reset Scenario", key="sb_reset_v2", use_container_width=True):
            for k in ["sb_price_v2", "sb_promo_rate_v2", "sb_mktg_v2",
                      "sb_promo_on_v2", "sb_horizon_v2", "sb_bands_v2"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1E3A5F;margin:0.85rem 0;'>", unsafe_allow_html=True)

        # ── Selected SKU card ──────────────────────────────────
        st.markdown("""<p style="color:#94A3B8;font-size:0.7rem;font-weight:700;
            text-transform:uppercase;letter-spacing:0.08em;
            margin-bottom:0.4rem;padding:0 0.2rem;">📦 Selected SKU</p>""",
            unsafe_allow_html=True)
        p          = SKU_MAP[selected_sku]
        full_stars = int(p["rating"])
        stars_str  = "★" * full_stars + "☆" * (5 - full_stars)
        plat_pills = " ".join([
            '<span class="platform-pill pill-amazon">Amazon</span>' if "Amazon" in p["platform"] else "",
            '<span class="platform-pill pill-flipkart">Flipkart</span>' if "Flipkart" in p["platform"] else "",
        ])
        plat_prices = _platform_prices(p)
        price_rows  = "".join([
            f"<div style='display:flex;justify-content:space-between;padding:0.15rem 0;'>"
            f"<span style='color:#64748B;font-size:0.7rem;'>{pl}</span>"
            f"<span style='color:#10B981;font-size:0.75rem;font-weight:700;'>₹{pr:,}</span></div>"
            for pl, pr in plat_prices.items()
        ])
        cat_em = CAT_EMOJI.get(p["cat"], "📦")
        st.markdown(f"""
        <div class="product-card">
            <div class="product-brand">{cat_em} {p['brand']}</div>
            <div class="product-name">{p['name']}</div>
            <div class="product-price">{inr(p['base_price'])}</div>
            <hr style="border:none;border-top:1px solid #1E3A5F;margin:0.5rem 0;">
            {price_rows}
            <div class="product-meta" style="margin-top:0.5rem;">
                <span style="color:#F59E0B;">{stars_str}</span> {p['rating']}
                &nbsp;·&nbsp; Lead: {p['lead_time']}d<br>
                <code style="color:#38BDF8;font-size:0.65rem;">{p['asin']}</code><br>
                <div style="margin-top:0.4rem;">{plat_pills}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── STICKY INFO BAR — always-visible context strip ────────────
    sku_info_early = SKU_MAP[selected_sku]
    sim_eff_price  = sku_info_early['base_price'] * (1 + sim_price_pct / 100.0)
    st.markdown(f"""
    <div class="sticky-bar">
      <span style="color:#38BDF8;font-size:0.7rem;font-weight:800;
                   text-transform:uppercase;letter-spacing:0.07em;flex-shrink:0;">&#128270; Active Analysis</span>
      <span class="sticky-chip">&#127919; <b>{selected_sku}</b></span>
      <span class="sticky-chip">&#128230; <b>{sku_info_early['brand']}</b> &middot; {sku_info_early['name'][:26]}&hellip;</span>
      <span class="sticky-chip">&#128197; Horizon: <b>{horizon}d</b></span>
      <span class="sticky-chip">&#128176; Base: <b>&#8377;{sku_info_early['base_price']:,}</b></span>
      <span class="sticky-chip">&#11088; {sku_info_early['rating']} &middot; Lead: <b>{sku_info_early['lead_time']}d</b></span>
      <span class="sticky-chip">&#128218; {sku_info_early['cat']}</span>
      {'<span class="sticky-chip" style="color:#F59E0B!important;border-color:rgba(245,158,11,0.4);">&#127991;&#65039; Promo ' + str(sim_promo_rate) + '% ON</span>' if sim_promo_on else ''}
      {'<span class="sticky-chip" style="color:#EF4444!important;border-color:rgba(239,68,68,0.4);">&#128184; Price ' + ("&#9650;" if sim_price_pct > 0 else "&#9660;") + str(abs(sim_price_pct)) + "%</span>" if sim_price_pct != 0 else ''}
      {'<span class="sticky-chip" style="color:#8B5CF6!important;border-color:rgba(139,92,246,0.4);">&#128202; Ad Spend &#8377;' + f"{sim_mktg:,}" + "/d</span>" if sim_mktg != 5000 else ''}
    </div>
    """, unsafe_allow_html=True)

    # ── § 1 — MULTI-PLATFORM PRODUCT SEARCH ─────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
      <span style="background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.3);
                   border-radius:6px;padding:0.15rem 0.55rem;font-size:0.65rem;
                   font-weight:800;color:#38BDF8;letter-spacing:0.06em;">STEP 1</span>
      <span style="color:#334155;font-size:0.78rem;font-weight:600;">Browse & compare all SKUs across platforms</span>
    </div>
    """, unsafe_allow_html=True)
    render_search_section(df)

    # ── § 2 — AI FORECAST + KPI STRIP ────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.75rem;">
      <span style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);
                   border-radius:6px;padding:0.15rem 0.55rem;font-size:0.65rem;
                   font-weight:800;color:#10B981;letter-spacing:0.06em;">STEP 2</span>
      <span style="color:#334155;font-size:0.78rem;font-weight:600;">
        AI-generated demand forecast &amp; executive KPIs for
        <b style="color:#F8FAFC;">selected SKU</b>
      </span>
    </div>
    """, unsafe_allow_html=True)

    sku_info = SKU_MAP[selected_sku]
    sku_df   = df[df["SKU_ID"] == selected_sku].copy()
    model_set= models[selected_sku]

    base_price     = sku_info["base_price"]
    sim_price      = base_price * (1 + sim_price_pct / 100.0)
    sim_promo_disc = (sim_promo_rate / 100.0) if sim_promo_on else 0.0
    if sim_promo_on:
        sim_price = sim_price * (1 - sim_promo_disc)

    future_df = build_forecast(sku_df, model_set, horizon, sim_price, sim_promo_disc, sim_mktg)

    predictions = future_df["Forecast"].tolist()
    lowers      = future_df["Lower"].tolist()
    uppers      = future_df["Upper"].tolist()
    last_date   = sku_df["Date"].max()

    # ── KPI Executive Strip ──────────────────────────────────────
    total_forecast  = sum(predictions)
    hist_comparable = sku_df["Demand"].iloc[-horizon:].sum()
    pct_change      = ((total_forecast - hist_comparable) / hist_comparable * 100) if hist_comparable else 0

    pipeline_rev = total_forecast * sim_price
    hist_rev     = (sku_df["Demand"].iloc[-horizon:] * sku_df["Current_Price"].iloc[-horizon:]).sum()
    rev_change   = ((pipeline_rev - hist_rev) / hist_rev * 100) if hist_rev else 0

    avg_demand   = float(sku_df["Demand"].iloc[-30:].mean())
    std_demand   = float(sku_df["Demand"].iloc[-30:].std())
    lead_time    = sku_info["lead_time"]
    safety_stock = int(1.65 * std_demand * np.sqrt(lead_time))
    rop          = int(lead_time * avg_demand + safety_stock)

    # Days of cover simulation
    current_stock = int(avg_demand * lead_time * 1.5)
    stockout_day  = None
    sim_stock     = current_stock
    for i, pred in enumerate(predictions):
        sim_stock -= pred
        if sim_stock <= safety_stock and stockout_day is None:
            stockout_day = future_df["Date"].iloc[i]
            break

    days_until_so = (stockout_day - last_date).days if stockout_day else 999
    risk_status   = "Critical" if stockout_day and days_until_so <= lead_time else ("Warning" if stockout_day else "Optimal")

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    def _delta(val: float) -> str:
        cls = "delta-up" if val >= 0 else "delta-down"
        sign = "▲" if val >= 0 else "▼"
        return f'<span class="{cls}">{sign} {abs(val):.1f}%</span>'

    with c1:
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="metric-label">Projected Demand ({horizon}d)</div>
            <div class="metric-value">{int(total_forecast):,}</div>
            <div class="metric-sub">Units &nbsp;{_delta(pct_change)} vs prev {horizon}d</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="metric-label">Pipeline Revenue</div>
            <div class="metric-value">{inr(pipeline_rev, compact=True)}</div>
            <div class="metric-sub">INR &nbsp;{_delta(rev_change)} vs prev {horizon}d</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        badge_cls = "badge-critical" if risk_status == "Critical" else ("badge-warning" if risk_status == "Warning" else "badge-optimal")
        so_txt    = stockout_day.strftime("%b %d") if stockout_day else "Safe"
        st.markdown(f"""
        <div class="metric-card {'red' if risk_status=='Critical' else 'amber' if risk_status=='Warning' else 'green'}">
            <div class="metric-label">Stockout Risk</div>
            <div style="margin:0.3rem 0 0.5rem;"><span class="badge {badge_cls}">{risk_status}</span></div>
            <div class="metric-sub">Breach Date: <b>{so_txt}</b></div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card amber">
            <div class="metric-label">Safety Stock (95%)</div>
            <div class="metric-value">{safety_stock:,}</div>
            <div class="metric-sub">ROP: <b>{rop:,} units</b> · Lead: {lead_time}d</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        # Top velocity SKU in category
        cat_df      = df[df["Category"] == sku_info["cat"]]
        top_sku_id  = cat_df.groupby("SKU_ID")["Demand"].sum().idxmax()
        top_sku_nm  = SKU_MAP[top_sku_id]["name"][:22] + "…"
        top_sku_br  = SKU_MAP[top_sku_id]["brand"]
        st.markdown(f"""
        <div class="metric-card violet">
            <div class="metric-label">Top Velocity SKU ({sku_info['cat'][:10]}…)</div>
            <div class="metric-value" style="font-size:1.15rem;">{top_sku_nm}</div>
            <div class="metric-sub" style="color:#38BDF8!important;">{top_sku_br} · {top_sku_id}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── § 3 — ANALYTICS TABS ──────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;">
      <span style="background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.35);
                   border-radius:6px;padding:0.15rem 0.55rem;font-size:0.65rem;
                   font-weight:800;color:#8B5CF6;letter-spacing:0.06em;">STEP 3</span>
      <span style="color:#334155;font-size:0.78rem;font-weight:600;">
        Deep-dive analytics &mdash; forecast, heatmap, elasticity &amp; category overview
      </span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "&#128200; Demand Forecast",
        "&#127777;&#65039; Demand Heatmap",
        "&#128185; Price Elasticity",
        "&#127970; Category Overview",
    ])

    # ─── TAB 1: FORECAST + FEATURE IMPORTANCE ─────────────────────
    with tab1:
        col_chart, col_feat = st.columns([3, 1])

        with col_chart:
            st.markdown("""
            <div class="section-header">
                <span style="font-size:1.4rem;">📈</span>
                <div>
                    <div class="section-title">AI-Powered Demand Forecast</div>
                    <div class="section-subtitle">Historical baseline + predicted trajectory with 80% confidence band</div>
                </div>
            </div>""", unsafe_allow_html=True)

            plot_hist = sku_df.iloc[-120:]  # last 120 days

            # ── Rolling 30-day moving average for historical baseline smoothing
            rolling_avg = plot_hist["Demand"].rolling(7, min_periods=1).mean()

            fig = go.Figure()

            # ── Shaded area under historical demand (visual depth)
            fig.add_trace(go.Scatter(
                x=plot_hist["Date"], y=plot_hist["Demand"],
                mode="none", fill="tozeroy",
                fillcolor="rgba(148,163,184,0.06)",
                showlegend=False, hoverinfo="skip"
            ))

            # ── 80% Confidence band (forecast uncertainty ribbon)
            if show_bands:
                fig.add_trace(go.Scatter(
                    x=pd.concat([future_df["Date"], future_df["Date"][::-1]]),
                    y=pd.concat([future_df["Upper"], future_df["Lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(56,189,248,0.14)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip", showlegend=True,
                    name="80% Confidence Band"
                ))

            # ── Raw historical daily sales (faint background)
            fig.add_trace(go.Scatter(
                x=plot_hist["Date"], y=plot_hist["Demand"],
                mode="lines", name="Historical Sales",
                line=dict(color="rgba(148,163,184,0.45)", width=1.2),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: <b>%{y} units</b><extra></extra>"
            ))

            # ── 7-day rolling average (smooth trend line)
            fig.add_trace(go.Scatter(
                x=plot_hist["Date"], y=rolling_avg,
                mode="lines", name="7-Day Rolling Avg",
                line=dict(color="#94A3B8", width=2, dash="dot"),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>7d Avg: <b>%{y:.0f} units</b><extra></extra>"
            ))

            # ── Shaded area under AI forecast (glowing fill)
            fig.add_trace(go.Scatter(
                x=future_df["Date"], y=future_df["Forecast"],
                mode="none", fill="tozeroy",
                fillcolor="rgba(56,189,248,0.08)",
                showlegend=False, hoverinfo="skip"
            ))

            # ── AI Forecast line with markers
            fig.add_trace(go.Scatter(
                x=future_df["Date"], y=future_df["Forecast"],
                mode="lines+markers", name="AI Forecast",
                line=dict(color="#38BDF8", width=3),
                marker=dict(size=5, color="#38BDF8",
                            line=dict(color="#0F172A", width=1.5)),
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Forecast: <b>%{y:.0f} units</b><br>"
                    "<span style='color:#64748B'>AI Quantile Regression · P50</span>"
                    "<extra></extra>"
                )
            ))

            # ── Daily Average ROP dashed reference line
            daily_rop = rop / horizon
            fig.add_hline(
                y=daily_rop,
                line_dash="dot", line_color="#EF4444", line_width=1.5,
                annotation_text=f"⚡ Reorder Trigger ({int(daily_rop)} u/day)",
                annotation_font_color="#EF4444",
                annotation_position="top left"
            )

            # ── Festive event vertical annotations on historical window
            FESTIVE_MARKS = [
                {"month": 10, "day": 9,  "label": "🎉 Big Billion Days",  "color": "#F59E0B"},
                {"month": 10, "day": 29, "label": "🪔 Diwali Week",        "color": "#F59E0B"},
                {"month": 1,  "day": 23, "label": "🇮🇳 Republic Day Sale", "color": "#10B981"},
                {"month": 6,  "day": 20, "label": "☀️ End-of-Season Sale", "color": "#8B5CF6"},
            ]
            min_hist_date = plot_hist["Date"].min()
            max_hist_date = plot_hist["Date"].max()
            for fm in FESTIVE_MARKS:
                for yr in [min_hist_date.year, max_hist_date.year]:
                    try:
                        mark_d = pd.Timestamp(year=yr, month=fm["month"], day=fm["day"])
                        if min_hist_date <= mark_d <= max_hist_date:
                            fig.add_vline(
                                x=mark_d.timestamp() * 1000,
                                line_dash="dot", line_color=fm["color"],
                                line_width=1,
                                annotation_text=fm["label"],
                                annotation_font_color=fm["color"],
                                annotation_font_size=9,
                                annotation_position="top right"
                            )
                    except Exception:
                        pass

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC", family="Inter, sans-serif"),
                margin=dict(l=0, r=0, t=10, b=0),
                height=420,
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    bgcolor="rgba(15,23,42,0.85)",
                    bordercolor="#1E3A5F", borderwidth=1,
                    font=dict(size=11)
                ),
                xaxis=dict(
                    showgrid=False, linecolor="#1E3A5F",
                    tickfont=dict(color="#64748B"),
                    title=dict(text="Date", font=dict(color="#64748B", size=11))
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="#1E293B",
                    linecolor="#1E3A5F",
                    title=dict(text="Units Sold / Day", font=dict(color="#64748B", size=11)),
                    tickfont=dict(color="#64748B")
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_feat:
            st.markdown("""
            <div class="section-header">
                <span style="font-size:1.4rem;">🎯</span>
                <div>
                    <div class="section-title">Demand Drivers</div>
                    <div class="section-subtitle">Top 7 features by model importance (%)</div>
                </div>
            </div>""", unsafe_allow_html=True)

            feat_df = pd.DataFrame({
                "Feature":    [FEATURE_DISPLAY.get(f, f) for f in FEATURES],
                "Importance": model_set["importance"]
            }).sort_values("Importance", ascending=True).tail(7)

            # Normalise to percentage for clarity
            feat_df["Pct"] = (feat_df["Importance"] / feat_df["Importance"].sum() * 100).round(1)
            feat_df["Label"] = feat_df["Pct"].map(lambda v: f"{v:.1f}%")

            fig_f = go.Figure(go.Bar(
                x=feat_df["Importance"],
                y=feat_df["Feature"],
                orientation="h",
                text=feat_df["Label"],
                textposition="outside",
                textfont=dict(color="#94A3B8", size=10),
                marker=dict(
                    color=feat_df["Importance"],
                    colorscale=[[0, "#1E3A5F"], [0.4, "#38BDF8"], [1.0, "#10B981"]],
                    line=dict(width=0)
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Importance: <b>%{text}</b><br>"
                    "<span style='color:#64748B'>of total model signal</span>"
                    "<extra></extra>"
                )
            ))
            fig_f.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC", size=10),
                margin=dict(l=0, r=30, t=10, b=0),
                height=400,
                xaxis=dict(showgrid=False, showticklabels=False,
                           title=dict(text="Relative Importance →", font=dict(color="#64748B", size=9))),
                yaxis=dict(showgrid=False, title="",
                           tickfont=dict(size=10, color="#94A3B8"))
            )
            st.plotly_chart(fig_f, use_container_width=True)

    # ─── TAB 2: DEMAND HEATMAP (Day x Month) ──────────────────────
    with tab2:
        st.markdown("""
        <div class="section-header">
            <span style="font-size:1.4rem;">🌡️</span>
            <div>
                <div class="section-title">Demand Heatmap — Day of Week × Month</div>
                <div class="section-subtitle">Identify festive peaks and weekly patterns at a glance</div>
            </div>
        </div>""", unsafe_allow_html=True)

        hm_df = sku_df.groupby(["Month", "DayOfWeek"])["Demand"].mean().reset_index()
        hm_pivot = hm_df.pivot(index="DayOfWeek", columns="Month", values="Demand")
        hm_pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        hm_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

        fig_hm = px.imshow(
            hm_pivot.fillna(0),
            color_continuous_scale=[[0,"#0F172A"],[0.3,"#1E3A5F"],[0.6,"#38BDF8"],[1.0,"#10B981"]],
            labels={"color": "Avg Daily Demand"},
            text_auto=".0f",
            aspect="auto"
        )
        fig_hm.update_traces(textfont=dict(size=11, color="#F8FAFC"))
        fig_hm.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC", family="Inter, sans-serif"),
            margin=dict(l=0, r=0, t=10, b=0), height=340,
            xaxis=dict(tickfont=dict(color="#94A3B8", size=11)),
            yaxis=dict(tickfont=dict(color="#94A3B8", size=11)),
            coloraxis_colorbar=dict(tickfont=dict(color="#94A3B8"), title=dict(text="Avg Demand", font=dict(color="#94A3B8")))
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # ── Weekly & Monthly demand breakdown ───────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div style="color:#94A3B8;font-size:0.85rem;font-weight:600;margin-bottom:0.4rem;">
                📅 Average Daily Demand by Day of Week
                <span style="color:#334155;font-weight:400;"> — Weekends highlighted in green</span>
            </div>""", unsafe_allow_html=True)

            dow_df = sku_df.groupby("DayOfWeek")["Demand"].mean().reset_index()
            dow_df["Day"] = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            dow_df["IsWeekend"] = dow_df["DayOfWeek"] >= 5

            # Gradient color: weekdays cool blue, weekends vibrant green
            dow_colors = [
                "#10B981" if w else "#38BDF8"
                for w in dow_df["IsWeekend"]
            ]
            overall_avg = dow_df["Demand"].mean()

            fig_dow = go.Figure()
            fig_dow.add_trace(go.Bar(
                x=dow_df["Day"],
                y=dow_df["Demand"],
                marker_color=dow_colors,
                marker_line_width=0,
                text=dow_df["Demand"].map(lambda v: f"{v:.0f}u"),
                textposition="outside",
                textfont=dict(color="#94A3B8", size=10),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Avg Demand: <b>%{y:.1f} units/day</b><br>"
                    "<span style='color:#64748B'>Historical average across all weeks</span>"
                    "<extra></extra>"
                )
            ))
            # Reference line: overall weekly average
            fig_dow.add_hline(
                y=overall_avg,
                line_dash="dot", line_color="#F59E0B", line_width=1.5,
                annotation_text=f"Weekly Avg: {overall_avg:.0f} u/day",
                annotation_font_color="#F59E0B",
                annotation_font_size=10,
                annotation_position="top right"
            )
            fig_dow.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                margin=dict(l=0, r=0, t=30, b=0), height=290,
                xaxis=dict(
                    showgrid=False,
                    title=dict(text="Day of Week", font=dict(color="#64748B", size=10))
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="#1E293B",
                    title=dict(text="Avg Units / Day", font=dict(color="#64748B", size=10)),
                    tickfont=dict(color="#64748B")
                )
            )
            st.plotly_chart(fig_dow, use_container_width=True)

        with col_b:
            st.markdown("""
            <div style="color:#94A3B8;font-size:0.85rem;font-weight:600;margin-bottom:0.4rem;">
                📆 Seasonal Demand by Month
                <span style="color:#334155;font-weight:400;"> — Oct/Nov festive surge expected</span>
            </div>""", unsafe_allow_html=True)

            mon_df = sku_df.groupby("Month")["Demand"].mean().reset_index()
            mon_names = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
            mon_df["MonthName"] = [mon_names[m - 1] for m in mon_df["Month"]]

            # Highlight festive months (Oct/Nov) in amber, others in violet
            mon_colors = [
                "#F59E0B" if m in [10, 11] else
                "#10B981" if m in [1]     else
                "#8B5CF6"
                for m in mon_df["Month"]
            ]
            annual_avg = mon_df["Demand"].mean()

            fig_mon = go.Figure()
            fig_mon.add_trace(go.Bar(
                x=mon_df["MonthName"],
                y=mon_df["Demand"],
                marker_color=mon_colors,
                marker_line_width=0,
                text=mon_df["Demand"].map(lambda v: f"{v:.0f}u"),
                textposition="outside",
                textfont=dict(color="#94A3B8", size=10),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Avg Demand: <b>%{y:.1f} units/day</b><br>"
                    "<span style='color:#64748B'>Monthly seasonal baseline</span>"
                    "<extra></extra>"
                )
            ))
            # Annual average reference line
            fig_mon.add_hline(
                y=annual_avg,
                line_dash="dot", line_color="#38BDF8", line_width=1.5,
                annotation_text=f"Annual Avg: {annual_avg:.0f} u/day",
                annotation_font_color="#38BDF8",
                annotation_font_size=10,
                annotation_position="top left"
            )
            # Add festive season annotation box
            fig_mon.add_vrect(
                x0="Oct", x1="Nov",
                fillcolor="rgba(245,158,11,0.08)",
                layer="below", line_width=0,
                annotation_text="🪔 Festive Peak",
                annotation_font_color="#F59E0B",
                annotation_font_size=9,
                annotation_position="top left"
            )
            fig_mon.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                margin=dict(l=0, r=0, t=30, b=0), height=290,
                xaxis=dict(
                    showgrid=False,
                    title=dict(text="Month", font=dict(color="#64748B", size=10))
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="#1E293B",
                    title=dict(text="Avg Units / Day", font=dict(color="#64748B", size=10)),
                    tickfont=dict(color="#64748B")
                )
            )
            st.plotly_chart(fig_mon, use_container_width=True)

    # ─── TAB 3: PRICE ELASTICITY CURVE ────────────────────────────
    with tab3:
        st.markdown("""
        <div class="section-header">
            <span style="font-size:1.4rem;">💹</span>
            <div>
                <div class="section-title">Price Elasticity Analysis</div>
                <div class="section-subtitle">Empirical demand response to price changes derived from historical data</div>
            </div>
        </div>""", unsafe_allow_html=True)

        price_bins = pd.cut(sku_df["Current_Price"], bins=12)
        el_df = sku_df.groupby(price_bins, observed=True).agg(
            avg_price=("Current_Price", "mean"),
            avg_demand=("Demand", "mean"),
        ).dropna().reset_index(drop=True)

        col_e1, col_e2 = st.columns([2, 1])
        with col_e1:
            fig_el = go.Figure()
            fig_el.add_trace(go.Scatter(
                x=el_df["avg_price"], y=el_df["avg_demand"],
                mode="lines+markers",
                line=dict(color="#F59E0B", width=3),
                marker=dict(size=8, color="#F59E0B", line=dict(color="#0F172A", width=2)),
                hovertemplate="Price: %{x:,.0f} ₹<br>Avg Demand: <b>%{y:.1f} units</b><extra></extra>"
            ))
            fig_el.add_vline(x=base_price, line_dash="dot", line_color="#38BDF8",
                             annotation_text=f"Base Price {inr(base_price)}", annotation_font_color="#38BDF8")
            if sim_price != base_price:
                fig_el.add_vline(x=sim_price, line_dash="dot", line_color="#10B981",
                                 annotation_text=f"Sim Price {inr(sim_price)}", annotation_font_color="#10B981")
            fig_el.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC", family="Inter,sans-serif"),
                margin=dict(l=0,r=0,t=10,b=0), height=380,
                xaxis=dict(showgrid=False, title="Price (₹)", tickprefix="₹", tickfont=dict(color="#64748B")),
                yaxis=dict(showgrid=True, gridcolor="#1E293B", title="Avg Daily Demand (units)", tickfont=dict(color="#64748B"))
            )
            st.plotly_chart(fig_el, use_container_width=True)

        with col_e2:
            st.markdown("""
            <div style="color:#94A3B8;font-size:0.85rem;font-weight:600;margin-bottom:0.5rem;">
                🏷️ Promotional Demand Lift Analysis
                <div style="color:#334155;font-size:0.75rem;font-weight:400;margin-top:0.15rem;">
                    Avg demand on promo days vs non-promo days
                </div>
            </div>""", unsafe_allow_html=True)

            promo_on  = sku_df[sku_df["Is_Promo"] == 1]["Demand"].mean()
            promo_off = sku_df[sku_df["Is_Promo"] == 0]["Demand"].mean()
            lift_pct  = ((promo_on - promo_off) / promo_off * 100) if promo_off > 0 else 0

            promo_counts = sku_df["Is_Promo"].value_counts()
            n_off = int(promo_counts.get(0, 0))
            n_on  = int(promo_counts.get(1, 0))

            fig_promo = go.Figure()
            fig_promo.add_trace(go.Bar(
                x=["No Promo", "With Promo"],
                y=[promo_off, promo_on],
                marker_color=["#1E3A5F", "#10B981"],
                marker_line_color=["#38BDF8", "#059669"],
                marker_line_width=1.5,
                width=0.55,
                text=[f"{promo_off:.0f}u", f"{promo_on:.0f}u"],
                textposition="outside",
                textfont=dict(color="#94A3B8", size=12, family="Inter"),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Avg Demand: <b>%{y:.1f} units/day</b><br>"
                    "<span style='color:#64748B'>Based on historical sales data</span>"
                    "<extra></extra>"
                )
            ))
            # Lift annotation arrow between the two bars
            if lift_pct > 0:
                fig_promo.add_annotation(
                    x="With Promo", y=promo_on * 1.18,
                    text=f"+{lift_pct:.1f}% lift 🚀",
                    showarrow=False,
                    font=dict(color="#10B981", size=11, family="Inter"),
                    bgcolor="rgba(16,185,129,0.12)",
                    bordercolor="#10B981", borderwidth=1,
                    borderpad=4, opacity=0.9
                )
            fig_promo.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC", family="Inter,sans-serif"),
                margin=dict(l=0, r=0, t=10, b=0), height=230,
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(color="#94A3B8", size=11)
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="#1E293B",
                    title=dict(text="Avg Units / Day", font=dict(color="#64748B", size=10)),
                    tickfont=dict(color="#64748B")
                )
            )
            st.plotly_chart(fig_promo, use_container_width=True)

            st.markdown(f"""
            <div class="metric-card green" style="padding:1rem;">
                <div class="metric-label">Promo Demand Lift</div>
                <div class="metric-value" style="font-size:1.8rem;">{lift_pct:+.1f}%</div>
                <div class="metric-sub">{inr(promo_off, True)}/d → {inr(promo_on, True)}/d avg</div>
                <div class="metric-sub" style="margin-top:0.4rem;">📊 {n_on} promo days · {n_off} regular days</div>
            </div>""", unsafe_allow_html=True)

    # ─── TAB 4: CATEGORY OVERVIEW ──────────────────────────────────
    with tab4:
        st.markdown("""
        <div class="section-header">
            <span style="font-size:1.4rem;">🏢</span>
            <div>
                <div class="section-title">Category Overview — All Platforms</div>
                <div class="section-subtitle">Cross-category demand volume and revenue comparison</div>
            </div>
        </div>""", unsafe_allow_html=True)

        last_30 = df[df["Date"] >= (last_date - timedelta(days=30))]
        cat_agg = last_30.groupby("Category").agg(
            Total_Demand=("Demand", "sum"),
            Total_Revenue=("Current_Price", lambda x: (x * last_30.loc[x.index, "Demand"]).sum()),
            Avg_Price=("Current_Price", "mean"),
        ).reset_index()

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
            <div style="color:#94A3B8;font-size:0.85rem;font-weight:600;margin-bottom:0.4rem;">
                📦 Total Units Sold by Category — Last 30 Days
                <div style="color:#334155;font-size:0.75rem;font-weight:400;margin-top:0.1rem;">
                    Colour intensity indicates volume; labels show exact unit counts
                </div>
            </div>""", unsafe_allow_html=True)

            cat_sorted = cat_agg.sort_values("Total_Demand")
            # Revenue-per-unit as annotation
            cat_sorted["RevPerUnit"] = (cat_sorted["Total_Revenue"] / cat_sorted["Total_Demand"]).round(0)

            fig_cv = go.Figure(go.Bar(
                x=cat_sorted["Total_Demand"],
                y=cat_sorted["Category"],
                orientation="h",
                marker=dict(
                    color=cat_sorted["Total_Demand"],
                    colorscale=[[0, "#1E3A5F"], [0.5, "#38BDF8"], [1.0, "#10B981"]],
                    line=dict(width=0)
                ),
                text=cat_sorted["Total_Demand"].map(lambda v: f"{v:,.0f} units"),
                textposition="outside",
                textfont=dict(color="#94A3B8", size=11),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Units Sold: <b>%{x:,}</b><br>"
                    "Revenue / Unit: <b>₹%{customdata:,.0f}</b><br>"
                    "<span style='color:#64748B'>Last 30 days aggregate</span>"
                    "<extra></extra>"
                ),
                customdata=cat_sorted["RevPerUnit"]
            ))
            fig_cv.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                margin=dict(l=0, r=80, t=10, b=0), height=290,
                xaxis=dict(
                    showgrid=False, showticklabels=False,
                    title=dict(text="Total Units →", font=dict(color="#64748B", size=10))
                ),
                yaxis=dict(
                    showgrid=False, title="",
                    tickfont=dict(color="#94A3B8", size=11)
                )
            )
            st.plotly_chart(fig_cv, use_container_width=True)

        with col_t2:
            st.markdown("""
            <div style="color:#94A3B8;font-size:0.85rem;font-weight:600;margin-bottom:0.4rem;">
                💰 Revenue Share by Category — Last 30 Days
                <div style="color:#334155;font-size:0.75rem;font-weight:400;margin-top:0.1rem;">
                    Donut shows INR revenue distribution across categories
                </div>
            </div>""", unsafe_allow_html=True)

            fig_rv = go.Figure(go.Pie(
                labels=cat_agg["Category"],
                values=cat_agg["Total_Revenue"],
                hole=0.52,
                marker=dict(
                    colors=["#38BDF8", "#10B981", "#F59E0B", "#8B5CF6"],
                    line=dict(color="#0F172A", width=2.5)
                ),
                textinfo="percent+label",
                textfont=dict(size=11, color="#F8FAFC"),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Revenue: <b>₹%{value:,.0f}</b><br>"
                    "Share: <b>%{percent}</b>"
                    "<extra></extra>"
                ),
                pull=[0.04] * len(cat_agg)  # slight pull for visual pop
            ))
            # Centre annotation showing total revenue
            total_rev = cat_agg["Total_Revenue"].sum()
            rev_label = f"₹{total_rev/1e5:.1f}L" if total_rev >= 1e5 else f"₹{total_rev:,.0f}"
            fig_rv.add_annotation(
                text=f"<b>{rev_label}</b><br><span style='font-size:9px;color:#64748B'>Total Revenue</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#F8FAFC", size=13, family="Inter")
            )
            fig_rv.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                margin=dict(l=0, r=0, t=10, b=0), height=290,
                showlegend=True,
                legend=dict(
                    orientation="v", yanchor="middle", y=0.5,
                    xanchor="left", x=1.01,
                    font=dict(size=10, color="#94A3B8"),
                    bgcolor="rgba(0,0,0,0)"
                )
            )
            st.plotly_chart(fig_rv, use_container_width=True)

        # SKU-level performance table
        st.markdown("**SKU Performance Scorecard — Last 30 Days**")
        sku_agg = last_30.groupby(["SKU_ID", "SKU_Name", "Brand", "Category", "Platform"]).agg(
            Units_Sold=("Demand", "sum"),
            Avg_Daily_Demand=("Demand", "mean"),
            Revenue_INR=("Current_Price", lambda x: (x * last_30.loc[x.index, "Demand"]).sum()),
            Avg_Price_INR=("Current_Price", "mean"),
            Promo_Days=("Is_Promo", "sum"),
        ).reset_index().sort_values("Revenue_INR", ascending=False)
        sku_agg["Revenue_INR"] = sku_agg["Revenue_INR"].map(lambda x: f"₹{x:,.0f}")
        sku_agg["Avg_Price_INR"] = sku_agg["Avg_Price_INR"].map(lambda x: f"₹{x:,.0f}")
        sku_agg["Avg_Daily_Demand"] = sku_agg["Avg_Daily_Demand"].map(lambda x: f"{x:.0f} u/day")
        sku_agg = sku_agg.rename(columns={
            "SKU_ID": "SKU ID", "SKU_Name": "Product Name", "Brand": "Brand",
            "Category": "Category", "Platform": "Platform",
            "Units_Sold": "Units Sold", "Avg_Daily_Demand": "Avg Daily Demand",
            "Revenue_INR": "Revenue (INR)", "Avg_Price_INR": "Avg Price",
            "Promo_Days": "Promo Days"
        })
        st.dataframe(sku_agg, use_container_width=True, hide_index=True, height=400)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1E3A5F;'>", unsafe_allow_html=True)

    # ── MODULE E: INVENTORY REPLENISHMENT ADVISOR ─────────────────
    st.markdown("""
    <div class="section-header">
        <span style="font-size:1.6rem;">📦</span>
        <div>
            <div class="section-title">Inventory Replenishment &amp; Decision Advisor</div>
            <div class="section-subtitle">
                Safety Stock = Z × σ_demand × √Lead Time &nbsp;|&nbsp;
                ROP = (Lead Time × ADemand) + Safety Stock &nbsp;|&nbsp;
                Z = 1.65 → 95% Service Level
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Scope to current category filter
    scope_ids = (df["SKU_ID"].unique().tolist() if selected_cat == "All"
                 else df[df["Category"] == selected_cat]["SKU_ID"].unique().tolist())

    rp_rows: list[dict] = []
    rp_seed = np.random.RandomState(99)   # Fixed seed for stable on-hand values
    for sid in scope_ids:
        sinfo = SKU_MAP[sid]
        sdf   = df[df["SKU_ID"] == sid]
        s_avg = float(sdf["Demand"].iloc[-30:].mean())
        s_std = float(sdf["Demand"].iloc[-30:].std())
        s_lt  = sinfo["lead_time"]

        s_ss  = int(1.65 * s_std * np.sqrt(s_lt))
        s_rop = int(s_lt * s_avg + s_ss)
        s_oh  = int(s_avg * s_lt * rp_seed.uniform(0.55, 2.1))

        days_cover  = (s_oh / s_avg) if s_avg > 0 else 999
        s_so_date   = (last_date + timedelta(days=int(days_cover))).strftime("%d %b %Y") if days_cover < 90 else "≥90d Safe"

        if s_oh <= s_ss:
            action = f"🔴 URGENT PO: Order {int(s_rop * 1.5):,} units NOW"
        elif s_oh <= s_rop:
            action = f"🟠 PO Required: Order {int(s_rop):,} units"
        elif s_oh <= s_rop * 1.2:
            action = "🟡 Monitor Closely"
        else:
            action = "🟢 Optimal"

        rp_rows.append({
            "SKU ID":             sid,
            "Product Name":       sinfo["name"][:35] + "…" if len(sinfo["name"]) > 35 else sinfo["name"],
            "Brand":              sinfo["brand"],
            "Platform":           sinfo["platform"],
            "On-Hand (Units)":    s_oh,
            "Avg Daily Demand":   int(s_avg),
            "Safety Stock":       s_ss,
            "Reorder Point (ROP)":s_rop,
            "Stockout ETA":       s_so_date,
            "Action":             action,
        })

    rp_df = pd.DataFrame(rp_rows)

    def _style_action(val: str) -> str:
        if "URGENT" in val:    return "background:rgba(239,68,68,.2);color:#EF4444;font-weight:700"
        if "PO Required" in val: return "background:rgba(245,158,11,.2);color:#F59E0B;font-weight:700"
        if "Monitor" in val:   return "background:rgba(245,158,11,.1);color:#F59E0B"
        return "color:#10B981"

    col_tbl, col_exp = st.columns([5, 1])
    with col_tbl:
        st.dataframe(
            rp_df.style.map(_style_action, subset=["Action"]),
            use_container_width=True,
            hide_index=True,
            height=400,
        )
    with col_exp:
        st.markdown("<br>", unsafe_allow_html=True)
        csv_bytes = rp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export CSV",
            data=csv_bytes,
            file_name=f"replenishment_plan_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown(f"""
        <div class="metric-card red" style="padding:1rem;margin-top:1rem;">
            <div class="metric-label">Critical SKUs</div>
            <div class="metric-value" style="font-size:2rem;">{rp_df['Action'].str.contains('URGENT').sum()}</div>
            <div class="metric-sub">Require immediate PO</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card amber" style="padding:1rem;margin-top:0.75rem;">
            <div class="metric-label">PO Required</div>
            <div class="metric-value" style="font-size:2rem;">{rp_df['Action'].str.contains('PO Required').sum()}</div>
            <div class="metric-sub">Reorder triggered</div>
        </div>""", unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────────
    st.markdown("""
    <hr style="border-color:#1E3A5F;margin-top:2rem;">
    <div style="text-align:center;padding:1.5rem 0 1rem;color:#334155;font-size:0.78rem;">
        <span style="color:#38BDF8;font-weight:700;">OptiDemand AI v2.0</span>
        &nbsp;·&nbsp; Built with Streamlit &amp; Scikit-Learn
        &nbsp;·&nbsp; Amazon India &amp; Flipkart Product Intelligence
        &nbsp;·&nbsp; All data is synthetic for demonstration purposes
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
