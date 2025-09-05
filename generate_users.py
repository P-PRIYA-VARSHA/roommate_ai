# generate_users.py
import pandas as pd
import random

# Possible categorical values
smoking_pref = ["yes", "no"]
drinking_pref = ["yes", "no"]
sleeping_schedule = ["early", "late", "flexible"]
locations = ["Downtown", "Suburbs", "Near College", "City Center"]
hobbies_list = ["sports", "music", "reading", "gaming", "travel", "cooking"]

# Generate fake users
users = []
for user_id in range(1, 21):  # 20 users
    user = {
        "userId": user_id,
        "cleanlinessLevel": random.randint(1, 5),  # scale 1–5
        "smokingPreference": random.choice(smoking_pref),
        "drinkingPreference": random.choice(drinking_pref),
        "sleepingSchedule": random.choice(sleeping_schedule),
        "hobbies": random.choice(hobbies_list),
        "budgetRange": random.randint(5000, 20000),  # in INR
        "preferredLocation": random.choice(locations),
    }
    users.append(user)

# Save to CSV
df = pd.DataFrame(users)
df.to_csv("users.csv", index=False)

print("✅ users.csv generated with", len(users), "users")
print(df.head())
