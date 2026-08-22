import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv("train-data.csv")

features = [
    "age_first_funding_year",
    "age_last_funding_year",
    "age_first_milestone_year",
    "age_last_milestone_year",
    "relationships",
    "funding_rounds",
    "funding_total_usd",
    "milestones",
    "is_CA",
    "is_NY",
    "is_MA",
    "is_TX",
    "is_otherstate",
    "is_software",
    "is_web",
    "is_mobile",
    "is_enterprise",
    "is_advertising",
    "is_gamesvideo",
    "is_ecommerce",
    "is_biotech",
    "is_consulting",
    "is_othercategory",
    "has_VC",
    "has_angel",
    "has_roundA",
    "has_roundB",
    "has_roundC",
    "has_roundD",
    "avg_participants",
    "is_top500"
]

X = data[features]
y = data["labels"]

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Model Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

joblib.dump(
    model,
    "startup_model.pkl"
)

joblib.dump(
    features,
    "startup_features.pkl"
)

print("\nModel saved successfully!")