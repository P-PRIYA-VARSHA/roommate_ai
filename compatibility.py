# compatibility.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("users.csv")

# Feature weights
weights = {
    "cleanlinessLevel": 2.0,
    "budgetRange": 2.0,
    "smokingPreference": 1.5,
    "drinkingPreference": 1.5,
    "sleepingSchedule": 1.0,
    "hobbies": 1.0,
    "preferredLocation": 2.5
}

# Scale numerical features before vectorization
scaler = MinMaxScaler()
df["cleanlinessLevel"] = scaler.fit_transform(df[["cleanlinessLevel"]])
df["budgetRange"] = scaler.fit_transform(df[["budgetRange"]])

# Convert rows to dicts
user_dicts = df.drop(columns=["userId"]).to_dict(orient="records")

# One-hot encode categorical features
vectorizer = DictVectorizer(sparse=False)
user_vectors = vectorizer.fit_transform(user_dicts)

# Apply weights
feature_names = vectorizer.get_feature_names_out()
weighted_vectors = user_vectors.copy()
for i, feature in enumerate(feature_names):
    base_feature = feature.split("=")[0]
    weighted_vectors[:, i] *= weights.get(base_feature, 1.0)

# Compute similarity
similarity_matrix = cosine_similarity(weighted_vectors)

def get_matches(user_id, top_n=3):
    if user_id not in df["userId"].values:
        return f"❌ userId {user_id} not found."

    user_index = df[df["userId"] == user_id].index[0]
    sim_scores = list(enumerate(similarity_matrix[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != user_index]

    top_matches = sim_scores[:top_n]

    results = []
    for idx, score in top_matches:
        match_user = df.iloc[idx]
        results.append({
            "match_userId": int(match_user["userId"]),
            "similarity_score": round(float(score)*100, 2),  # force float not np.float64
            "cleanlinessLevel": float(match_user["cleanlinessLevel"]),
            "smokingPreference": match_user["smokingPreference"],
            "drinkingPreference": match_user["drinkingPreference"],
            "sleepingSchedule": match_user["sleepingSchedule"],
            "hobbies": match_user["hobbies"],
            "budgetRange": float(match_user["budgetRange"]),
            "preferredLocation": match_user["preferredLocation"],
        })
    return results


if __name__ == "__main__":
    print("✅ Dataset loaded with", len(df), "users.")
    user_id = 10
    matches = get_matches(user_id, top_n=3)
    print(f"\nTop 3 matches for user {user_id}:")
    for m in matches:
        print(m)
