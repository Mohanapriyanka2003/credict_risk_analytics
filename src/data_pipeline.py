import os
import sqlite3
import pandas as pd
import numpy as np
import requests

# Constants
DATA_DIR = os.path.join("data", "raw")
DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "credit_records.db")
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

def ensure_directories():
    """Create data and database directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    print("Directory structure verified.")

def download_dataset():
    """Downloads the German Credit dataset from UCI. Falls back to generating a synthetic dataset on failure."""
    ensure_directories()
    raw_file_path = os.path.join(DATA_DIR, "german.data")
    
    if os.path.exists(raw_file_path):
        print(f"Dataset already exists at {raw_file_path}.")
        return raw_file_path

    print(f"Downloading German Credit dataset from {UCI_DATA_URL}...")
    try:
        response = requests.get(UCI_DATA_URL, timeout=15)
        response.raise_for_status()
        with open(raw_file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Dataset downloaded successfully.")
        return raw_file_path
    except Exception as e:
        print(f"Download failed: {e}. Generating high-fidelity synthetic fallback dataset...")
        return generate_synthetic_data(raw_file_path)

def generate_synthetic_data(file_path):
    """Generates a synthetic German Credit dataset with similar distributions to ensure robust offline execution."""
    np.random.seed(42)
    n_samples = 1000
    
    # Categorical distributions approximations based on German credit characteristics
    checking_choices = ['A11', 'A12', 'A13', 'A14']
    checking_probs = [0.3, 0.3, 0.1, 0.3]
    
    history_choices = ['A30', 'A31', 'A32', 'A33', 'A34']
    history_probs = [0.05, 0.05, 0.55, 0.1, 0.25]
    
    purpose_choices = ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410']
    purpose_probs = [0.25, 0.1, 0.18, 0.28, 0.02, 0.02, 0.05, 0.01, 0.08, 0.01]
    
    savings_choices = ['A61', 'A62', 'A63', 'A64', 'A65']
    savings_probs = [0.6, 0.1, 0.06, 0.05, 0.19]
    
    employment_choices = ['A71', 'A72', 'A73', 'A74', 'A75']
    employment_probs = [0.06, 0.17, 0.34, 0.17, 0.26]
    
    sex_status_choices = ['A91', 'A92', 'A93', 'A94']
    sex_status_probs = [0.05, 0.31, 0.54, 0.1]
    
    debtors_choices = ['A101', 'A102', 'A103']
    debtors_probs = [0.9, 0.05, 0.05]
    
    property_choices = ['A121', 'A122', 'A123', 'A124']
    property_probs = [0.28, 0.23, 0.33, 0.16]
    
    installments_choices = ['A141', 'A142', 'A143']
    installments_probs = [0.14, 0.05, 0.81]
    
    housing_choices = ['A151', 'A152', 'A153']
    housing_probs = [0.18, 0.71, 0.11]
    
    job_choices = ['A171', 'A172', 'A173', 'A174']
    job_probs = [0.02, 0.2, 0.63, 0.15]
    
    lines = []
    for _ in range(n_samples):
        # Durations tend to cluster around 12, 18, 24, 36 months
        duration = np.random.choice([6, 9, 12, 15, 18, 24, 30, 36, 42, 48, 60, 72], 
                                    p=[0.05, 0.02, 0.3, 0.05, 0.15, 0.25, 0.02, 0.08, 0.01, 0.04, 0.02, 0.01])
        
        # Credit amount scales heavily with duration
        base_amt = duration * 150
        credit_amt = int(np.clip(np.random.lognormal(mean=np.log(base_amt), sigma=0.5), 250, 20000))
        
        installment_rate = np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.2, 0.5])
        residence = np.random.choice([1, 2, 3, 4], p=[0.15, 0.3, 0.15, 0.4])
        age = int(np.clip(np.random.normal(loc=35, scale=11), 18, 75))
        existing_cr = np.random.choice([1, 2, 3, 4], p=[0.65, 0.3, 0.04, 0.01])
        liable = np.random.choice([1, 2], p=[0.85, 0.15])
        
        checking = np.random.choice(checking_choices, p=checking_probs)
        history = np.random.choice(history_choices, p=history_probs)
        purpose = np.random.choice(purpose_choices, p=purpose_probs)
        savings = np.random.choice(savings_choices, p=savings_probs)
        employment = np.random.choice(employment_choices, p=employment_probs)
        sex_status = np.random.choice(sex_status_choices, p=sex_status_probs)
        debtors = np.random.choice(debtors_choices, p=debtors_probs)
        prop = np.random.choice(property_choices, p=property_probs)
        other_inst = np.random.choice(installments_choices, p=installments_probs)
        house = np.random.choice(housing_choices, p=housing_probs)
        job = np.random.choice(job_choices, p=job_probs)
        phone = np.random.choice(['A191', 'A192'], p=[0.6, 0.4])
        foreign = np.random.choice(['A201', 'A202'], p=[0.96, 0.04])
        
        # Determine risk based on logical credit metrics
        score = 0
        if checking == 'A11': score -= 2
        if checking == 'A14': score += 2
        if history in ['A30', 'A31']: score -= 1
        if history == 'A34': score += 1
        if savings in ['A61', 'A65']: score -= 1
        if savings in ['A63', 'A64']: score += 1.5
        if age < 25: score -= 0.5
        if duration > 24: score -= 1
        if credit_amt > 5000: score -= 1
        
        # Calculate PD and map to risk labels (1: Good, 2: Bad)
        pd_prob = 1 / (1 + np.exp(score + np.random.normal(loc=0.5, scale=1.0)))
        risk = 2 if pd_prob > 0.5 else 1
        
        row = f"{checking} {duration} {history} {purpose} {credit_amt} {savings} {employment} {installment_rate} {sex_status} {debtors} {residence} {prop} {age} {other_inst} {house} {existing_cr} {job} {liable} {phone} {foreign} {risk}"
        lines.append(row)
        
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Synthetic dataset created successfully.")
    return file_path

# Mappings for categorical codes based on German Credit specifications
CODE_MAPPINGS = {
    'status_checking': {
        'A11': '< 0 DM',
        'A12': '0 to < 200 DM',
        'A13': '>= 200 DM / salary assignments',
        'A14': 'no checking account'
    },
    'credit_history': {
        'A30': 'no credits taken',
        'A31': 'all credits paid back duly at this bank',
        'A32': 'existing credits paid back duly till now',
        'A33': 'delay in paying off in the past',
        'A34': 'critical account / other credits elsewhere'
    },
    'purpose': {
        'A40': 'car (new)',
        'A41': 'car (used)',
        'A42': 'furniture/equipment',
        'A43': 'radio/television',
        'A44': 'domestic appliances',
        'A45': 'repairs',
        'A46': 'education',
        'A47': 'vacation',
        'A48': 'retraining',
        'A49': 'business',
        'A410': 'others'
    },
    'savings': {
        'A61': '< 100 DM',
        'A62': '100 to < 500 DM',
        'A63': '500 to < 1000 DM',
        'A64': '>= 1000 DM',
        'A65': 'unknown / no savings account'
    },
    'employment': {
        'A71': 'unemployed',
        'A72': '< 1 year',
        'A73': '1 to < 4 years',
        'A74': '4 to < 7 years',
        'A75': '>= 7 years'
    },
    'personal_status_sex': {
        'A91': 'male : divorced/separated',
        'A92': 'female : divorced/separated/married',
        'A93': 'male : single',
        'A94': 'male : married/widowed',
        'A95': 'female : single'
    },
    'other_debtors': {
        'A101': 'none',
        'A102': 'co-applicant',
        'A103': 'guarantor'
    },
    'property': {
        'A121': 'real estate',
        'A122': 'building society savings agreement/life insurance',
        'A123': 'car or other',
        'A124': 'no property / unknown'
    },
    'other_installments': {
        'A141': 'bank',
        'A142': 'stores',
        'A143': 'none'
    },
    'housing': {
        'A151': 'rent',
        'A152': 'own',
        'A153': 'for free'
    },
    'job': {
        'A171': 'unemployed/unskilled non-resident',
        'A172': 'unskilled resident',
        'A173': 'skilled employee/official',
        'A174': 'management/highly qualified/self-employed'
    },
    'telephone': {
        'A191': 'none',
        'A192': 'yes'
    },
    'foreign_worker': {
        'A201': 'yes',
        'A202': 'no'
    }
}

def parse_and_process_data(file_path):
    """Loads raw German credit data, maps codes, derives sensitive attributes and simulated LGD/EAD metrics."""
    cols = [
        'status_checking', 'duration_months', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment', 'installment_rate', 'personal_status_sex', 'other_debtors',
        'residence_since', 'property', 'age', 'other_installments', 'housing',
        'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'credit_risk'
    ]
    
    # Read space delimited data
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=cols)
    
    # Pre-mapping copy for original codes if needed
    df_mapped = df.copy()
    
    # Apply category maps
    for col, mappings in CODE_MAPPINGS.items():
        df_mapped[col] = df_mapped[col].map(mappings)
    
    # Map target credit_risk (1=Good, 2=Bad) -> 0=Paid Duly, 1=Defaulted
    df_mapped['default'] = df_mapped['credit_risk'].apply(lambda x: 1 if x == 2 else 0)
    df_mapped.drop(columns=['credit_risk'], inplace=True)
    
    # 1. Derive sensitive demographic fields
    # personal_status_sex mappings:
    # A91 (male: divorced/separated) -> Male, Divorced/Separated
    # A92 (female: divorced/sep/married) -> Female, Married/Divorced/Separated
    # A93 (male: single) -> Male, Single
    # A94 (male: married/widowed) -> Male, Married/Widowed
    # A95 (female: single) -> Female, Single
    def extract_sex(val):
        if 'female' in str(val).lower():
            return 'Female'
        return 'Male'
        
    def extract_status(val):
        parts = str(val).split(':')
        if len(parts) > 1:
            return parts[1].strip()
        return 'single'
        
    df_mapped['sex'] = df_mapped['personal_status_sex'].apply(extract_sex)
    df_mapped['personal_status'] = df_mapped['personal_status_sex'].apply(extract_status)
    
    # Group age into regulatory buckets for fairness analysis (IFRS 9 guidelines)
    df_mapped['age_group'] = df_mapped['age'].apply(lambda x: 'Young (<=30)' if x <= 30 else 'Mature (>30)')
    
    # 2. Simulate metrics required for ECL calculation (since German Credit lacks raw LGD/EAD figures)
    np.random.seed(42)
    
    # Collateral Value simulation based on property type
    # Real estate: ~120% of credit, life insurance: ~80%, car: ~50%, no property: 0
    property_cov_ratios = {
        'real estate': 1.20,
        'building society savings agreement/life insurance': 0.80,
        'car or other': 0.50,
        'no property / unknown': 0.0
    }
    
    df_mapped['collateral_value'] = df_mapped.apply(
        lambda r: round(r['credit_amount'] * property_cov_ratios.get(r['property'], 0.0) * np.random.uniform(0.9, 1.1), 2),
        axis=1
    )
    
    # EAD (Exposure at Default) represents outstanding balance. Typically ranges from 70% to 100% of the loan amount
    df_mapped['current_balance'] = df_mapped.apply(
        lambda r: round(r['credit_amount'] * np.random.uniform(0.75, 1.0), 2),
        axis=1
    )
    
    # Monthly Payment estimate based on simple amortized model
    df_mapped['monthly_payment'] = df_mapped.apply(
        lambda r: round((r['credit_amount'] * (1 + 0.08 * (r['duration_months'] / 12))) / r['duration_months'], 2),
        axis=1
    )
    
    # Generate unique Customer IDs
    df_mapped['customer_id'] = [f"CUST-{i:05d}" for i in range(len(df_mapped))]
    
    print("Data parsing and feature derivation complete.")
    return df_mapped

def save_to_sqlite(df):
    """Loads parsed loan records into structured relational SQLite database tables."""
    ensure_directories()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop existing tables to avoid schema mismatches
    cursor.execute("DROP TABLE IF EXISTS loans")
    cursor.execute("DROP TABLE IF EXISTS borrowers")
    
    # Create Borrowers Table (demographics and employment)
    cursor.execute("""
    CREATE TABLE borrowers (
        customer_id TEXT PRIMARY KEY,
        age INTEGER,
        sex TEXT,
        personal_status TEXT,
        age_group TEXT,
        job TEXT,
        employment TEXT,
        residence_since INTEGER,
        people_liable INTEGER,
        telephone TEXT,
        foreign_worker TEXT
    )
    """)
    
    # Create Loans Table (credit parameters, history, collateral, outcomes)
    cursor.execute("""
    CREATE TABLE loans (
        customer_id TEXT PRIMARY KEY,
        duration_months INTEGER,
        credit_history TEXT,
        purpose TEXT,
        credit_amount REAL,
        current_balance REAL,
        monthly_payment REAL,
        savings TEXT,
        status_checking TEXT,
        installment_rate INTEGER,
        other_debtors TEXT,
        property TEXT,
        collateral_value REAL,
        other_installments TEXT,
        housing TEXT,
        existing_credits INTEGER,
        default_label INTEGER,
        FOREIGN KEY (customer_id) REFERENCES borrowers(customer_id)
    )
    """)
    
    # Split dataframe for insert
    borrower_cols = ['customer_id', 'age', 'sex', 'personal_status', 'age_group', 'job', 'employment', 
                     'residence_since', 'people_liable', 'telephone', 'foreign_worker']
    loan_cols = ['customer_id', 'duration_months', 'credit_history', 'purpose', 'credit_amount', 
                 'current_balance', 'monthly_payment', 'savings', 'status_checking', 'installment_rate', 
                 'other_debtors', 'property', 'collateral_value', 'other_installments', 'housing', 
                 'existing_credits', 'default']
    
    borrowers_df = df[borrower_cols]
    loans_df = df[loan_cols].rename(columns={'default': 'default_label'})
    
    borrowers_df.to_sql('borrowers', conn, if_exists='append', index=False)
    loans_df.to_sql('loans', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()
    print(f"Successfully populated SQLite database at {DB_PATH}.")

def run_sample_sql_queries():
    """Runs a series of representative SQL queries on our relational credit database to verify and analyze data."""
    conn = sqlite3.connect(DB_PATH)
    
    print("\n--- SQL Query 1: Default rates and average loan amounts by housing status ---")
    query1 = """
    SELECT 
        housing, 
        COUNT(*) as total_loans, 
        AVG(credit_amount) as avg_amount, 
        ROUND(AVG(default_label) * 100, 2) as default_rate_percentage
    FROM loans
    GROUP BY housing
    ORDER BY default_rate_percentage DESC;
    """
    res1 = pd.read_sql_query(query1, conn)
    print(res1)
    
    print("\n--- SQL Query 2: Credit characteristics by age groups ---")
    query2 = """
    SELECT 
        b.age_group, 
        COUNT(*) as customer_count,
        AVG(l.credit_amount) as avg_credit_limit,
        ROUND(AVG(l.default_label) * 100, 2) as default_rate_percentage,
        ROUND(AVG(l.collateral_value), 2) as avg_collateral_value
    FROM borrowers b
    JOIN loans l ON b.customer_id = l.customer_id
    GROUP BY b.age_group;
    """
    res2 = pd.read_sql_query(query2, conn)
    print(res2)
    
    conn.close()

def main():
    print("Starting Credit Risk Data Pipeline...")
    raw_path = download_dataset()
    processed_df = parse_and_process_data(raw_path)
    save_to_sqlite(processed_df)
    run_sample_sql_queries()
    print("Data Pipeline Completed successfully!")

if __name__ == "__main__":
    main()
