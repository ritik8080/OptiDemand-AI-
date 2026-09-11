# 🚀 OptiDemand AI — E-Commerce Demand Forecasting & Inventory Intelligence

> **Enterprise-grade AI demand forecasting platform built for Amazon India & Flipkart product catalogues.**
> Powered by Gradient Boosting Quantile Regression with a real-time interactive Streamlit dashboard.

---

## 🏷️ Badges

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **AI Forecasting** | Per-SKU Gradient Boosting Quantile Regression (P10 / P50 / P90) |
| 📊 **Interactive Charts** | Plotly-powered demand forecast charts with confidence bands |
| 🏷️ **14 Real SKUs** | Samsung, OnePlus, boAt, Sony, Levi's, Puma, Philips, Mamaearth & more |
| 🛍️ **Multi-Platform** | Amazon India & Flipkart price comparison with Meesho fallback |
| 🎯 **Scenario Simulator** | Adjust price, promotions & ad spend to re-run AI forecast live |
| 📅 **Indian Festive Calendar** | Big Billion Days, Diwali, Republic Day, Holi & seasonal multipliers |
| 📦 **Inventory Optimiser** | EOQ, safety stock, and reorder point calculations per SKU |
| 📈 **Executive KPIs** | MAPE, MAD, revenue forecast, elasticity, GMV projections |
| 🔍 **Product Search** | Full catalogue browse with platform price cards |
| 📉 **Feature Importance** | Visual breakdown of demand drivers per SKU |

---

## 🧠 ML Pipeline

```
Raw Synthetic Data (2-year daily, 14 SKUs)
        │
        ▼
Feature Engineering
  ├─ Lag features   : Demand_Lag_1, Demand_Lag_7
  ├─ Rolling stats  : Rolling_Mean_7, Rolling_Mean_30, Rolling_Std_7
  ├─ Calendar       : DayOfWeek, Month, Is_Weekend
  └─ Exogenous      : Current_Price, Promo_Discount, Marketing_Spend
        │
        ▼
Per-SKU GradientBoostingRegressor
  ├─ P10 (lower bound)
  ├─ P50 (point forecast / median)
  └─ P90 (upper bound)
        │
        ▼
Auto-Regressive Forward Forecast (up to 90 days)
  └─ Rolling lag update on each predicted step
```

---

## 🗂️ Project Structure

```
E-commerce Product Demand Prediction/
│
├── app.py              # 🚀 Main Streamlit application (single-file architecture)
├── requirements.txt    # 📦 Python dependencies
├── README.md           # 📖 This file
├── CONTRIBUTING.md     # 🤝 Contribution guidelines
├── LICENSE             # ⚖️  MIT License
└── .gitignore          # 🙈 Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites
- Python **3.10** or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/E-commerce-Product-Demand-Prediction.git
cd E-commerce-Product-Demand-Prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501** 🎉

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.31.0 | Web application framework |
| `pandas` | ≥ 2.0.0 | Data manipulation |
| `numpy` | ≥ 1.24.0 | Numerical computing |
| `plotly` | ≥ 5.18.0 | Interactive visualisations |
| `scikit-learn` | ≥ 1.3.0 | ML models (Gradient Boosting) |

---

## 🛒 Product Catalogue

The platform covers **14 real-world SKUs** across **4 categories**:

| Category | Products |
|---|---|
| 📱 Electronics | Samsung Galaxy S24 Ultra, OnePlus 12R, boAt Rockerz 550, Sony WH-1000XM5, AmazonBasics 43" Fire TV |
| 👔 Apparel | Levi's 501 Jeans, Puma Softride Running Shoes, Peter England Formal Shirt, Wildcraft Packable Jacket |
| 🍳 Home & Kitchen | Philips HD9252 Air Fryer, Prestige Iris Mixer Grinder |
| 💄 Beauty & Personal Care | Mamaearth Vitamin C Serum, Lakmé Absolute Lipstick, Dot & Key Moisturizer |

---

## 🧪 Data Generation

The app generates **2 years of synthetic daily demand data** (~10,136 records) at startup using:

- **Indian festive events**: Big Billion Days (4.5×), Diwali (3.8×), Great Indian Festival (4.0×), Republic Day (2.2×), Holi (1.8×)
- **Weekend sensitivity** per category (e.g., Electronics: 1.5× on weekends)
- **Price elasticity**: −2.0 for Electronics, −1.5 for other categories
- **Marketing spend effects**: correlated with ad spend (INR/day)
- **Random noise**: ±10% Gaussian noise for realism

> All data is deterministically seeded for reproducible results across reruns.

---

## 🎮 Using the Scenario Simulator

The left sidebar contains the **Scenario Simulator** panel:

1. **Horizon (Days)** — Set the forecast window (7–90 days)
2. **Show 80% Confidence Band** — Toggle P10/P90 uncertainty bands
3. **Price Adj. (%)** — Simulate price increases / decreases
4. **Activate Promotion** — Toggle a promotional campaign
5. **Promo Discount Rate (%)** — Set the discount percentage
6. **Ad Spend (₹/day)** — Adjust daily marketing spend

Changes take effect **instantly** — the AI forecast reruns in real time.

---

## 📊 Dashboard Sections

| Section | What it shows |
|---|---|
| **Step 1** | Multi-platform product search & price comparison cards |
| **Step 2** | AI forecast chart + executive KPI strip |
| **Step 3** | Historical demand analysis & category benchmarking |
| **Step 4** | Inventory optimisation (EOQ, safety stock, reorder point) |
| **Step 5** | Feature importance & model diagnostics |
| **Step 6** | Scenario comparison & what-if analysis |

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

Built with ❤️ for the Indian e-commerce ecosystem.
Inspired by real-world demand planning challenges at Amazon India & Flipkart scale.
