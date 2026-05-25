import os
import pickle
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, mean_squared_error, r2_score

# Paths
DB_PATH = os.path.join("database", "credit_records.db")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data_from_db():
    """Loads combined customer and loan profiles from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        b.customer_id, b.age, b.sex, b.personal_status, b.age_group, b.job, b.employment,
        b.residence_since, b.people_liable, b.telephone, b.foreign_worker,
        l.duration_months, l.credit_history, l.purpose, l.credit_amount, l.current_balance,
        l.monthly_payment, l.savings, l.status_checking, l.installment_rate, l.other_debtors,
        l.property, l.collateral_value, l.other_installments, l.housing, l.existing_credits,
        l.default_label
    FROM borrowers b
    JOIN loans l ON b.customer_id = l.customer_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Loaded {len(df)} records from SQLite database.")
    return df

def generate_historical_losses(df):
    """
    Derives realistic historical EAD and LGD figures for model training.
    For defaulting borrowers, we simulate actual exposures (EAD) and losses (LGD) based on credit size and collateral.
    """
    np.random.seed(42)
    
    # 1. EAD (Exposure at Default) represents outstanding balance at default.
    # We model EAD as current balance with minor variation (95% to 105% of outstanding balance)
    df['ead_actual'] = df['current_balance'] * np.random.uniform(0.95, 1.05, len(df))
    
    # 2. LGD (Loss Given Default) is percentage lost (1 - Recovery Rate)
    # Recoveries are driven by collateral coverage. If collateral >= current balance, loss is low (10-20%).
    # If collateral is 0, loss is high (80-95%).
    collateral_ratio = df['collateral_value'] / (df['current_balance'] + 1)
    # Recovery rate is simulated based on collateral ratio
    recovery_rate = np.clip(collateral_ratio * np.random.uniform(0.7, 0.95, len(df)), 0.05, 0.95)
    df['lgd_actual'] = 1.0 - recovery_rate
    
    return df

def train_pd_model(df):
    """Trains a Probability of Default (PD) classification pipeline."""
    print("\n--- Training PD (Probability of Default) Model ---")
    
    # Define features
    categorical_features = ['status_checking', 'credit_history', 'purpose', 'savings', 
                            'employment', 'personal_status', 'property', 'other_installments', 'housing', 'job']
    numeric_features = ['duration_months', 'credit_amount', 'installment_rate', 'residence_since', 
                        'age', 'existing_credits', 'people_liable']
    
    X = df[categorical_features + numeric_features]
    y = df['default_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # PD Model: Logistic Regression (preferred in banking for explainability and coefficients)
    pd_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, C=0.5))
    ])
    
    # Train
    pd_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pd_pipeline.predict(X_test)
    y_pred_proba = pd_pipeline.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"PD Model ROC-AUC: {auc:.4f}")
    print(f"PD Model Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Pipeline
    model_path = os.path.join(MODELS_DIR, "pd_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pd_pipeline, f)
    print(f"PD Model saved to {model_path}.")
    
    return pd_pipeline

def train_lgd_model(df):
    """Trains a Loss Given Default (LGD) regression pipeline."""
    print("\n--- Training LGD (Loss Given Default) Model ---")
    
    # LGD features emphasize housing security, collateral value, and borrower savings
    categorical_features = ['property', 'housing', 'status_checking', 'savings']
    numeric_features = ['credit_amount', 'current_balance', 'collateral_value']
    
    X = df[categorical_features + numeric_features]
    y = df['lgd_actual']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # LGD Model: RandomForestRegressor to capture non-linear recovery behaviors
    lgd_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))
    ])
    
    lgd_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = lgd_pipeline.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 1.0) # LGD is bounded [0, 1]
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"LGD Model MSE: {mse:.4f}")
    print(f"LGD Model R2 Score: {r2:.4f}")
    
    # Save Pipeline
    model_path = os.path.join(MODELS_DIR, "lgd_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(lgd_pipeline, f)
    print(f"LGD Model saved to {model_path}.")
    
    return lgd_pipeline

def train_ead_model(df):
    """Trains an Exposure at Default (EAD) regression pipeline."""
    print("\n--- Training EAD (Exposure at Default) Model ---")
    
    # EAD features emphasize total borrowing limit, monthly schedule, and duration
    categorical_features = ['purpose', 'other_installments']
    numeric_features = ['credit_amount', 'duration_months', 'installment_rate', 'current_balance', 'monthly_payment']
    
    X = df[categorical_features + numeric_features]
    y = df['ead_actual']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # EAD Model: Random Forest Regressor
    ead_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))
    ])
    
    ead_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = ead_pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"EAD Model MSE: {mse:.4f}")
    print(f"EAD Model R2 Score: {r2:.4f}")
    
    # Save Pipeline
    model_path = os.path.join(MODELS_DIR, "ead_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(ead_pipeline, f)
    print(f"EAD Model saved to {model_path}.")
    
    return ead_pipeline

def calculate_ecl_and_fairness(df, pd_pipeline):
    """
    Computes overall ECL on the dataset and performs algorithmic fairness audit.
    We check for potential bias in loan approvals (low PD vs high PD) based on age and sex.
    """
    print("\n--- Performing Regulatory Fairness & Bias Audit ---")
    
    # Predict default probabilities (PD)
    categorical_features = ['status_checking', 'credit_history', 'purpose', 'savings', 
                            'employment', 'personal_status', 'property', 'other_installments', 'housing', 'job']
    numeric_features = ['duration_months', 'credit_amount', 'installment_rate', 'residence_since', 
                        'age', 'existing_credits', 'people_liable']
    
    X = df[categorical_features + numeric_features]
    df['pd_pred'] = pd_pipeline.predict_proba(X)[:, 1]
    
    # Define Approval Decision (Approved if PD <= 0.35, representing a standard high/medium credit standard)
    approval_threshold = 0.35
    df['approved'] = (df['pd_pred'] <= approval_threshold).astype(int)
    
    # Helper to calculate fairness metrics
    def audit_demographic(sensitive_col, privileged_group, unprivileged_group):
        priv_mask = df[sensitive_col] == privileged_group
        unpriv_mask = df[sensitive_col] == unprivileged_group
        
        priv_approved = df[priv_mask]['approved'].mean()
        unpriv_approved = df[unpriv_mask]['approved'].mean()
        
        disparate_impact = unpriv_approved / (priv_approved + 1e-9)
        demographic_parity = priv_approved - unpriv_approved
        
        print(f"\nAudit for '{sensitive_col}':")
        print(f"  Privileged Group ({privileged_group}) Approval Rate: {priv_approved:.2%}")
        print(f"  Unprivileged Group ({unprivileged_group}) Approval Rate: {unpriv_approved:.2%}")
        print(f"  Disparate Impact Ratio (Standard threshold >0.8): {disparate_impact:.4f}")
        print(f"  Demographic Parity Difference (Goal near 0.0): {demographic_parity:.4f}")
        
        # Check compliance status
        status = "COMPLIANT [PASS]" if disparate_impact >= 0.8 and disparate_impact <= 1.25 else "POTENTIAL BIAS [WARNING]"
        print(f"  Status: {status}")
        
        return {
            'disparate_impact': disparate_impact,
            'demographic_parity': demographic_parity,
            'status': status
        }
    
    # 1. Audit Sex
    sex_audit = audit_demographic('sex', 'Male', 'Female')
    
    # 2. Audit Age
    age_audit = audit_demographic('age_group', 'Mature (>30)', 'Young (<=30)')
    
    return sex_audit, age_audit

def main():
    print("Initializing Machine Learning Training Pipeline...")
    df = load_data_from_db()
    df = generate_historical_losses(df)
    
    pd_model = train_pd_model(df)
    lgd_model = train_lgd_model(df)
    ead_model = train_ead_model(df)
    
    calculate_ecl_and_fairness(df, pd_model)
    print("\nMachine Learning Pipelines Completed Successfully!")

if __name__ == "__main__":
    main()
