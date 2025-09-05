import pandas as pd
import random

# Load base roommate dataset
df = pd.read_csv("roommate_dataset.csv")

pairs = []
for _ in range(5000):  # create 5000 pairs
    u1, u2 = random.sample(df.index.tolist(), 2)
    user1, user2 = df.loc[u1], df.loc[u2]

    # Synthetic compatibility scoring
    score = 50
    if user1["smokingPreference"] == user2["smokingPreference"]:
        score += 10
    if user1["drinkingPreference"] == user2["drinkingPreference"]:
        score += 10
    if user1["sleepingSchedule"] == user2["sleepingSchedule"]:
        score += 10
    if user1["preferredLocation"] == user2["preferredLocation"]:
        score += 10

    # Hobbies overlap
    hobbies1 = set(str(user1["hobbies"]).split(","))
    hobbies2 = set(str(user2["hobbies"]).split(","))
    overlap = len(hobbies1 & hobbies2)
    score += overlap * 3

    # Budget similarity
    budget_diff = abs(user1["budgetRange"] - user2["budgetRange"])
    if budget_diff < 2000:
        score += 7
    elif budget_diff < 5000:
        score += 4

    # Cleanliness closeness
    score -= abs(user1["cleanlinessLevel"] - user2["cleanlinessLevel"]) * 2

    # Clamp score between 0 and 100
    score = max(0, min(100, score))

    # Build row with full attributes
    row = {
        "userId1": user1["userId"],
        "userId2": user2["userId"],
        "cleanlinessLevel_user1": user1["cleanlinessLevel"],
        "cleanlinessLevel_user2": user2["cleanlinessLevel"],
        "smokingPreference_user1": user1["smokingPreference"],
        "smokingPreference_user2": user2["smokingPreference"],
        "drinkingPreference_user1": user1["drinkingPreference"],
        "drinkingPreference_user2": user2["drinkingPreference"],
        "sleepingSchedule_user1": user1["sleepingSchedule"],
        "sleepingSchedule_user2": user2["sleepingSchedule"],
        "budgetRange_user1": user1["budgetRange"],
        "budgetRange_user2": user2["budgetRange"],
        "preferredLocation_user1": user1["preferredLocation"],
        "preferredLocation_user2": user2["preferredLocation"],
        "hobbies_user1": user1["hobbies"],
        "hobbies_user2": user2["hobbies"],
        "compatibility_score": score
    }
    pairs.append(row)

# Save expanded labeled dataset
pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv("labeled_pairs.csv", index=False)

print("✅ New labeled_pairs.csv generated with expanded user attributes!")
