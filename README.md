# CreditShield.AI: Banking Credit Risk & Expected Credit Loss (ECL) Analytics

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.136.3-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.6.1-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**CreditShield.AI** is a premium, interactive machine learning dashboard designed for the **Banking & Financial Services (BFSI) domain**. It implements a regulatory **Expected Credit Loss (ECL)** estimation pipeline—compliant with **IFRS 9 & CECL guidelines**—using the classic German Credit loan portfolio.

Powered by a lightweight **FastAPI** backend and an **SQLite** database, the application enables risk managers to evaluate individual loan underwriting decisions, simulate portfolio-wide macroeconomic stress scenarios (CCAR), and perform **Ethical AI Fairness Audits** to actively detect and mitigate algorithmic gender and age bias in real-time.

---

## 🌟 Key Features

*   **Ingestion & Data Pipeline**: Parses raw space-delimited UCI Statlog credit data, cleans categories, derives sensitive features (Sex, Age Group), and populates a local relational SQLite database (`database/credit_records.db`).
*   **Three-Tier Regulatory ML Models**: 
    1.  **PD (Probability of Default)**: Logistic Regression classifier modeling default probability ($AUC = 0.792$).
    2.  **LGD (Loss Given Default)**: Random Forest regressor modeling loss percentage based on property collateral coverage ($R^2 = 0.994$).
    3.  **EAD (Exposure at Default)**: Random Forest regressor modeling outstanding balance at default ($R^2 = 0.997$).
*   **Jupyter Notebook Workspace**: A comprehensive `ecl_analysis.ipynb` mapping exploratory analysis, SQL queries, ROC evaluations, and stress testing.
*   **Macroeconomic Stress-Testing Lab**: Live interactive sliders simulating custom PD multipliers (unemployment shocks) and LGD haircuts (housing market crashes) to re-evaluate portfolio capital requirements instantly.
*   **Ethical AI Fairness Console**: Audits credit approvals for protected attributes (Sex and Age) using **Disparate Impact** and **Demographic Parity**, featuring a dual-threshold slider to mitigate systemic biases dynamically.

---

## 🧮 Regulatory Risk Formulation

Expected Credit Loss is modeled mathematically as:

$$\text{ECL} = \text{PD} \times \text{LGD} \times \text{EAD}$$

Where:
*   **PD (Probability of Default)**: Continuous probability ($0.0 \text{ to } 1.0$) that a borrower defaults within a 12-month horizon.
*   **LGD (Loss Given Default)**: The percentage ($0.0 \text{ to } 1.0$) of the loan balance lost after repossessing assets.
*   **EAD (Exposure at Default)**: The total outstanding dollar value (DM) exposed to loss at the exact time of default.

---

## 📁 Repository Directory Structure

```text
credit_risk_ecl_analytics/
├── data/raw/              # Raw Statlog German Credit dataset files
├── database/              # Relational SQLite credit database
├── models/                # Serialized ML pipeline estimators (.pkl)
├── src/
│   ├── data_pipeline.py   # Ingestion, cleaning, and SQL database loader
│   ├── model_pipeline.py  # Model training, validation, and fairness audits
│   └── server.py          # FastAPI server serving REST API prediction endpoints
├── templates/
│   └── index.html         # Front-end dashboard structure
├── static/
│   ├── css/
│   │   └── style.css      # Premium glassmorphic styling system
│   └── js/
│       └── app.js         # Reactive controller, SPA router, and Chart.js feeds
├── requirements.txt       # Pin-pointed Python library dependencies
├── ecl_analysis.ipynb     # Detailed mathematical analytical workspace notebook
└── README.md              # Project manual
```

---

## 🚀 Setup & Execution Guide

### 1. Clone the repository and install requirements
Ensure you have Python 3.12+ installed, then install dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Run the Data Pipeline
Incorporate raw files and build the SQLite database tables (`borrowers` and `loans`):
```powershell
python src/data_pipeline.py
```

### 3. Run the Machine Learning Training Pipeline
Train the PD, LGD, and EAD models, perform the fairness audit, and save serialized pipelines:
```powershell
python src/model_pipeline.py
```

### 4. Boot the FastAPI Server
Launch the asynchronous web server:
```powershell
python -m uvicorn src.server:app --reload
```

### 5. Open the Dashboard in your Browser
Navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## ⚖️ Ethical Bias Audit & Mitigation

Anti-discrimination laws state that approval distributions must satisfy the Disparate Impact ratio ($DI \ge 0.80$, the standard 80% rule). 

Due to systemic historical inequalities in the raw dataset, the baseline model presents gender bias ($DI = 0.7963$). To solve this, **CreditShield.AI** implements a **Dual-Threshold Optimization Strategy**:
1.   risk managers can adjust the unprivileged group's (Female) approval boundary from the general threshold of $PD \le 0.35$ to a customized threshold of $PD \le 0.41$.
2.  This compensates for credit history discrepancies, balancing group opportunity distributions and moving the disparate impact meter back into the green **legally compliant** zone.
