import os
import pickle
import sqlite3
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Credit Risk & Expected Credit Loss Analytics API", version="1.0.0")

# Paths
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"
MODELS_DIR = "models"
DB_PATH = os.path.join("database", "credit_records.db")

# Mount Static Directories (ensure folder exists first)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load Models globally
models = {}
def load_models():
    pd_path = os.path.join(MODELS_DIR, "pd_model.pkl")
    lgd_path = os.path.join(MODELS_DIR, "lgd_model.pkl")
    ead_path = os.path.join(MODELS_DIR, "ead_model.pkl")
    
    if os.path.exists(pd_path) and os.path.exists(lgd_path) and os.path.exists(ead_path):
        with open(pd_path, "rb") as f:
            models["pd"] = pickle.load(f)
        with open(lgd_path, "rb") as f:
            models["lgd"] = pickle.load(f)
        with open(ead_path, "rb") as f:
            models["ead"] = pickle.load(f)
        print("All machine learning pipelines loaded successfully.")
    else:
        print("WARNING: ML model pipelines missing. Please run model_pipeline.py first.")

@app.on_event("startup")
def startup_event():
    load_models()

# Pydantic Schemas for validation
class BorrowerFeatures(BaseModel):
    # PD features
    status_checking: str
    credit_history: str
    purpose: str
    savings: str
    employment: str
    personal_status: str
    property: str
    other_installments: str
    housing: str
    job: str
    duration_months: int
    credit_amount: float
    installment_rate: int
    residence_since: int
    age: int
    existing_credits: int
    people_liable: int
    
    # EAD/LGD specific features
    current_balance: float
    collateral_value: float
    monthly_payment: float
    sex: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend templates/index.html not found.")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/predict")
def predict_ecl(data: BorrowerFeatures):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded. Model pipeline execution required.")
    
    # Construct Pandas Row for inference
    pd_row = pd.DataFrame([{
        'status_checking': data.status_checking,
        'credit_history': data.credit_history,
        'purpose': data.purpose,
        'savings': data.savings,
        'employment': data.employment,
        'personal_status': data.personal_status,
        'property': data.property,
        'other_installments': data.other_installments,
        'housing': data.housing,
        'job': data.job,
        'duration_months': data.duration_months,
        'credit_amount': data.credit_amount,
        'installment_rate': data.installment_rate,
        'residence_since': data.residence_since,
        'age': data.age,
        'existing_credits': data.existing_credits,
        'people_liable': data.people_liable
    }])
    
    lgd_row = pd.DataFrame([{
        'property': data.property,
        'housing': data.housing,
        'status_checking': data.status_checking,
        'savings': data.savings,
        'credit_amount': data.credit_amount,
        'current_balance': data.current_balance,
        'collateral_value': data.collateral_value
    }])
    
    ead_row = pd.DataFrame([{
        'purpose': data.purpose,
        'other_installments': data.other_installments,
        'credit_amount': data.credit_amount,
        'duration_months': data.duration_months,
        'installment_rate': data.installment_rate,
        'current_balance': data.current_balance,
        'monthly_payment': data.monthly_payment
    }])
    
    # Run pipelines
    pd_prob = float(models["pd"].predict_proba(pd_row)[0][1])
    lgd_val = float(np.clip(models["lgd"].predict(lgd_row)[0], 0.0, 1.0))
    ead_val = float(models["ead"].predict(ead_row)[0])
    
    # Calculate ECL
    ecl_val = pd_prob * lgd_val * ead_val
    
    # Calculate risk rating category
    if pd_prob <= 0.05:
        risk_rating = "Low Risk (AAA-A)"
        risk_color = "var(--success)"
    elif pd_prob <= 0.15:
        risk_rating = "Medium-Low Risk (BBB-BB)"
        risk_color = "var(--accent)"
    elif pd_prob <= 0.35:
        risk_rating = "Medium-High Risk (B-CCC)"
        risk_color = "var(--warning)"
    else:
        risk_rating = "High Risk (Substandard/Default)"
        risk_color = "var(--danger)"
        
    return {
        "pd": pd_prob,
        "lgd": lgd_val,
        "ead": ead_val,
        "ecl": ecl_val,
        "risk_rating": risk_rating,
        "risk_color": risk_color,
        "is_approved": bool(pd_prob <= 0.35)
    }

@app.get("/portfolio")
def get_portfolio_stats():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="SQLite credit database is missing.")
        
    conn = sqlite3.connect(DB_PATH)
    
    # Aggregate Stats
    summary = pd.read_sql_query("""
    SELECT 
        COUNT(*) as total_accounts,
        SUM(credit_amount) as total_exposure,
        AVG(credit_amount) as average_loan_size,
        AVG(duration_months) as average_duration
    FROM loans
    """, conn).to_dict(orient="records")[0]
    
    # Fetch all data to run prediction aggregations for baseline ECL
    query_all = "SELECT b.*, l.* FROM borrowers b JOIN loans l ON b.customer_id = l.customer_id"
    df = pd.read_sql_query(query_all, conn)
    conn.close()
    
    if models:
        # PD features
        pd_features = ['status_checking', 'credit_history', 'purpose', 'savings', 
                       'employment', 'personal_status', 'property', 'other_installments', 'housing', 'job',
                       'duration_months', 'credit_amount', 'installment_rate', 'residence_since', 
                       'age', 'existing_credits', 'people_liable']
        # LGD features
        lgd_features = ['property', 'housing', 'status_checking', 'savings',
                        'credit_amount', 'current_balance', 'collateral_value']
        # EAD features
        ead_features = ['purpose', 'other_installments',
                        'credit_amount', 'duration_months', 'installment_rate', 'current_balance', 'monthly_payment']
        
        df['PD'] = models["pd"].predict_proba(df[pd_features])[:, 1]
        df['LGD'] = np.clip(models["lgd"].predict(df[lgd_features]), 0.0, 1.0)
        df['EAD'] = models["ead"].predict(df[ead_features])
        df['ECL'] = df['PD'] * df['LGD'] * df['EAD']
        
        baseline_ecl = float(df['ECL'].sum())
        average_pd = float(df['PD'].mean())
        average_lgd = float(df['LGD'].mean())
    else:
        baseline_ecl = summary["total_exposure"] * 0.05
        average_pd = 0.05
        average_lgd = 0.45
        
    # Group distribution by Purpose
    purpose_dist = df.groupby('purpose')['credit_amount'].sum().reset_index()
    purpose_list = purpose_dist.to_dict(orient="records")
    
    # Risk Categories count
    low_risk = int(np.sum(df['PD'] <= 0.05)) if models else 200
    med_low_risk = int(np.sum((df['PD'] > 0.05) & (df['PD'] <= 0.15))) if models else 400
    med_high_risk = int(np.sum((df['PD'] > 0.15) & (df['PD'] <= 0.35))) if models else 250
    high_risk = int(np.sum(df['PD'] > 0.35)) if models else 150
    
    return {
        "summary": summary,
        "ecl": {
            "baseline_total_ecl": baseline_ecl,
            "average_pd": average_pd,
            "average_lgd": average_lgd
        },
        "purpose_distribution": purpose_list,
        "risk_distribution": {
            "Low Risk": low_risk,
            "Medium-Low Risk": med_low_risk,
            "Medium-High Risk": med_high_risk,
            "High Risk": high_risk
        }
    }

@app.get("/stress_test")
def stress_test_portfolio(pd_shock: float = 1.0, collateral_haircut: float = 0.0):
    """
    Stress-test endpoint to run macroeconomic simulations.
    - pd_shock: multiplier applied to PD (e.g., 1.2 for 20% spike).
    - collateral_haircut: collateral asset drop percentage (e.g., 0.15 for 15% drop).
    """
    if not models:
        raise HTTPException(status_code=503, detail="ML Models not available.")
        
    conn = sqlite3.connect(DB_PATH)
    query_all = "SELECT b.*, l.* FROM borrowers b JOIN loans l ON b.customer_id = l.customer_id"
    df = pd.read_sql_query(query_all, conn)
    conn.close()
    
    # Baseline calculations
    pd_features = ['status_checking', 'credit_history', 'purpose', 'savings', 
                   'employment', 'personal_status', 'property', 'other_installments', 'housing', 'job',
                   'duration_months', 'credit_amount', 'installment_rate', 'residence_since', 
                   'age', 'existing_credits', 'people_liable']
    lgd_features = ['property', 'housing', 'status_checking', 'savings',
                    'credit_amount', 'current_balance', 'collateral_value']
    ead_features = ['purpose', 'other_installments',
                    'credit_amount', 'duration_months', 'installment_rate', 'current_balance', 'monthly_payment']
    
    df['PD'] = models["pd"].predict_proba(df[pd_features])[:, 1]
    df['LGD'] = np.clip(models["lgd"].predict(df[lgd_features]), 0.0, 1.0)
    df['EAD'] = models["ead"].predict(df[ead_features])
    df['ECL'] = df['PD'] * df['LGD'] * df['EAD']
    
    baseline_ecl = df['ECL'].sum()
    
    # Stress calculations
    df['PD_stressed'] = np.clip(df['PD'] * pd_shock, 0.0, 1.0)
    
    stressed_lgd_df = df[lgd_features].copy()
    stressed_lgd_df['collateral_value'] = stressed_lgd_df['collateral_value'] * (1.0 - collateral_haircut)
    df['LGD_stressed'] = np.clip(models["lgd"].predict(stressed_lgd_df), 0.0, 1.0)
    
    df['ECL_stressed'] = df['PD_stressed'] * df['LGD_stressed'] * df['EAD']
    stressed_ecl = df['ECL_stressed'].sum()
    
    return {
        "baseline_ecl": float(baseline_ecl),
        "stressed_ecl": float(stressed_ecl),
        "ecl_increase_pct": float(((stressed_ecl - baseline_ecl) / baseline_ecl) * 100),
        "average_pd_base": float(df['PD'].mean()),
        "average_pd_stressed": float(df['PD_stressed'].mean()),
        "average_lgd_base": float(df['LGD'].mean()),
        "average_lgd_stressed": float(df['LGD_stressed'].mean())
    }

@app.get("/fairness")
def audit_fairness(threshold: float = 0.35, female_threshold: float = 0.35):
    """
    Computes algorithmic fairness and disparate impact across sensitive groups.
    - threshold: default threshold for general approvals.
    - female_threshold: customized threshold for female unprivileged group (bias mitigation!).
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models missing.")
        
    conn = sqlite3.connect(DB_PATH)
    query_all = "SELECT b.*, l.* FROM borrowers b JOIN loans l ON b.customer_id = l.customer_id"
    df = pd.read_sql_query(query_all, conn)
    conn.close()
    
    pd_features = ['status_checking', 'credit_history', 'purpose', 'savings', 
                   'employment', 'personal_status', 'property', 'other_installments', 'housing', 'job',
                   'duration_months', 'credit_amount', 'installment_rate', 'residence_since', 
                   'age', 'existing_credits', 'people_liable']
                   
    df['PD'] = models["pd"].predict_proba(df[pd_features])[:, 1]
    
    # Decisions based on thresholds
    df['decision'] = df.apply(
        lambda r: 1 if (r['sex'] == 'Female' and r['PD'] <= female_threshold) or (r['sex'] != 'Female' and r['PD'] <= threshold) else 0,
        axis=1
    )
    
    # Audit Sex
    male_approved = float(df[df['sex'] == 'Male']['decision'].mean())
    female_approved = float(df[df['sex'] == 'Female']['decision'].mean())
    sex_di = female_approved / (male_approved + 1e-9)
    sex_dp = male_approved - female_approved
    
    # Audit Age (Mature vs Young, standard threshold)
    df['age_decision'] = (df['PD'] <= threshold).astype(int)
    mature_approved = float(df[df['age_group'] == 'Mature (>30)']['age_decision'].mean())
    young_approved = float(df[df['age_group'] == 'Young (<=30)']['age_decision'].mean())
    age_di = young_approved / (mature_approved + 1e-9)
    age_dp = mature_approved - young_approved
    
    return {
        "sex_audit": {
            "privileged_rate": male_approved,
            "unprivileged_rate": female_approved,
            "disparate_impact": sex_di,
            "demographic_parity": sex_dp,
            "status": "COMPLIANT" if sex_di >= 0.8 and sex_di <= 1.25 else "BIASED_WARNING"
        },
        "age_audit": {
            "privileged_rate": mature_approved,
            "unprivileged_rate": young_approved,
            "disparate_impact": age_di,
            "demographic_parity": age_dp,
            "status": "COMPLIANT" if age_di >= 0.8 and age_di <= 1.25 else "BIASED_WARNING"
        }
    }
