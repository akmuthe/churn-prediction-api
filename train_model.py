import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib

np.random.seed(42)
n = 3000

tenure = np.random.randint(0, 72, n)
monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
tech_support = np.random.choice(["Yes", "No"], n, p=[0.4, 0.6])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
)
total_charges = np.round(monthly_charges * (tenure + 1) * np.random.uniform(0.9, 1.1, n), 2)

# churn probability logic: short tenure, month-to-month, fiber, no tech support, electronic check -> higher churn
churn_score = (
    (tenure < 12).astype(int) * 0.35
    + (contract == "Month-to-month").astype(int) * 0.30
    + (internet_service == "Fiber optic").astype(int) * 0.15
    + (tech_support == "No").astype(int) * 0.15
    + (payment_method == "Electronic check").astype(int) * 0.15
    + np.random.normal(0, 0.15, n)
)
churn = (churn_score > 0.55).astype(int)

df = pd.DataFrame({
    "tenure": tenure,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "contract": contract,
    "internet_service": internet_service,
    "tech_support": tech_support,
    "payment_method": payment_method,
    "churn": churn,
})

print("Churn rate:", df["churn"].mean().round(3))
df.to_csv("churn_data.csv", index=False)

# Encode categoricals
encoders = {}
df_enc = df.copy()
for col in ["contract", "internet_service", "tech_support", "payment_method"]:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le

feature_cols = ["tenure", "monthly_charges", "total_charges", "contract",
                 "internet_service", "tech_support", "payment_method"]
X = df_enc[feature_cols]
y = df_enc["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("\n--- Model Evaluation ---")
print(classification_report(y_test, preds))
print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))

joblib.dump(model, "churn_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
print("\nSaved: churn_model.pkl, encoders.pkl, feature_cols.pkl")
