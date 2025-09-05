import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
pairs = pd.read_csv("labeled_pairs.csv")

# Drop IDs and target
X = pairs.drop(columns=["compatibility_score", "userId1", "userId2"])
y = pairs["compatibility_score"]

# Detect categorical columns automatically (object or category dtype)
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical features:", cat_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train CatBoost model with categorical features
model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    verbose=100
)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

model.fit(train_pool, eval_set=test_pool)

# Predictions
y_pred = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained!")
print(f"📊 RMSE: {rmse:.2f}")
print(f"📊 MAE: {mae:.2f}")
print(f"📊 R² Score: {r2:.4f}")

# Save model
model.save_model("compatibility_model.cbm")
print("💾 Model saved as compatibility_model.cbm")
