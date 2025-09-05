import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
pairs = pd.read_csv("labeled_pairs.csv")

# -------------------------------
# Feature Engineering
# -------------------------------
def jaccard_similarity(hobbies1, hobbies2):
    set1, set2 = set(str(hobbies1).split(",")), set(str(hobbies2).split(","))
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

pairs["hobby_similarity"] = pairs.apply(
    lambda row: jaccard_similarity(row.get("hobbies_user1", ""), row.get("hobbies_user2", "")), axis=1
)

pairs["budget_diff"] = abs(pairs["budgetRange_user1"] - pairs["budgetRange_user2"]) / pairs[
    ["budgetRange_user1", "budgetRange_user2"]
].max(axis=1)

pairs["location_match"] = (pairs["preferredLocation_user1"] == pairs["preferredLocation_user2"]).astype(int)

# Drop unused raw text columns if present
drop_cols = [c for c in pairs.columns if "hobbies_user" in c]
X = pairs.drop(columns=["compatibility_score"] + drop_cols)
y = pairs["compatibility_score"]

# Identify categorical features
categorical_features = [col for col in X.columns if X[col].dtype == "object"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Train CatBoost Regressor
# -------------------------------
cat_model = CatBoostRegressor(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100
)

train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

cat_model.fit(train_pool, eval_set=test_pool)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = cat_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# -------------------------------
# Save Model
# -------------------------------
cat_model.save_model("roommate_matcher_catboost.cbm")
joblib.dump(cat_model, "roommate_matcher_catboost.pkl")

print("✅ Model saved as roommate_matcher_catboost.cbm and .pkl")
