import joblib

model = joblib.load("startup_model.pkl")
features = joblib.load("startup_features.pkl")

print("Model:")
print(model)

print("\nFeatures:")
for feature in features:
    print(feature)