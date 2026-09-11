# 🤝 Contributing to OptiDemand AI

Thank you for your interest in contributing! Here is how to get involved.

---

## 🐛 Reporting Bugs

1. Check existing [Issues](../../issues) to avoid duplicates.
2. Open a new issue with:
   - A clear title and description
   - Steps to reproduce the bug
   - Expected vs actual behaviour
   - Python and library version info (`pip list`)
   - Screenshot or error traceback if applicable

---

## 💡 Suggesting Features

Open a [Feature Request](../../issues/new) issue describing:
- The problem your feature solves
- How you envision it working
- Any alternatives you have considered

---

## 🛠️ Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/E-commerce-Product-Demand-Prediction.git
cd E-commerce-Product-Demand-Prediction

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📁 Codebase Overview

The entire application is a **single-file Streamlit app** (`app.py`) organised into clearly commented modules:

| Section | Lines | Description |
|---|---|---|
| Page Config & CSS | 1–351 | Global dark-mode styles, sidebar, sticky bar CSS |
| Module A — Catalogue | 353–529 | `PRODUCT_CATALOGUE` list with 14 real SKUs |
| Module A — Data Gen | 531–666 | `generate_data()` — 2-year synthetic demand with Indian festive events |
| Module A — ML Engine | 669–714 | `train_models()` — per-SKU quantile GBR models |
| Forecast Builder | 717–761 | `build_forecast()` — auto-regressive forward prediction |
| Helpers | 764–850 | `inr()`, `_platform_prices()`, `CAT_EMOJI`, `SKU_MAP` |
| UI — Sidebar | ~1050 | Category filter, SKU selector, scenario sliders |
| UI — Main | ~1100 | Sticky bar, search, forecast chart, KPIs, inventory, diagnostics |

---

## ✅ Pull Request Guidelines

- **One feature / fix per PR** — keeps reviews focused.
- Follow the existing code style (clear comments, type hints where practical).
- Test your changes locally with `streamlit run app.py`.
- Update `README.md` if you add a new feature or change behaviour.
- Keep commit messages descriptive: `feat: add category-level demand heatmap`.

---

## 📄 Code Style

- Use **4-space indentation**.
- Keep functions small and well-commented.
- Group related code with `# ── Section Name ──────` headers (matches existing style).
- Prefer `f-strings` over `.format()`.

---

## 📜 Licence

By contributing, you agree that your contributions will be licensed under the **MIT License**.
