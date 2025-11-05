🏠 Roommate & PG Recommendation System

This project is a Recommendation System that helps users find suitable roommates and PG (Paying Guest) accommodations based on compatibility scores.
It uses the CatBoost Regressor model to predict compatibility levels.

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

2. Install the required libraries

Make sure you have Python installed. Then run:
pip install catboost pandas numpy scikit-learn


3. Run the model
python compatibility.py

🧠 Model Performance
Metric	Score
R²	0.96
RMSE	2.15
MAE	1.71


✅ Interpretation:
The model explains 96% of the variance in compatibility scores — indicating excellent predictive performance.

📊 Tech Stack

Python
CatBoost Regressor
Pandas, NumPy, Scikit-learn

💡 Future Enhancements

Add a web interface for users to input preference.
Integrate a map-based PG location search
Implement collaborative filtering for better recommendations


![Result Description

For User 10, the system identified three top roommate matches based on lifestyle compatibility and preferences:

User 7 is the best match (84.72%), sharing similar flexibility in lifestyle and location preferences, with a travel-oriented hobby and a moderate budget range.

User 13 follows closely with an 80.12% match, having similar smoking and drinking preferences, and a balanced cleanliness level.

User 9 shows a 68.63% match, differing slightly in sleeping habits (late schedule) but aligning well in terms of non-smoking, non-drinking preferences, and suburban location choice.

Overall, the recommendation system effectively suggests roommates with closely aligned habits, budgets, and lifestyle choices, demonstrating how the model can assist users in finding compatible living partners.](Result.png)

