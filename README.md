🏠 Roommate & PG Recommendation System

This project is a Machine Learning–based Recommendation System that helps users find suitable roommates and PG (Paying Guest) accommodations based on compatibility scores.
It leverages the CatBoost Regressor model to predict compatibility levels between users using lifestyle and preference-based attributes.

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/P-PRIYA-VARSHA/roommate_ai.git
cd roommate_ai

2️⃣ Install the Required Libraries

Ensure you have Python 3.8+ installed. Then run:

pip install catboost pandas numpy scikit-learn



3️⃣ Run the Model
python compatibility.py

🧠 Model Performance
Metric	Score
R²	0.96
RMSE	2.15
MAE	1.71

✅ Interpretation:
The model explains 96% of the variance in compatibility scores — indicating excellent predictive performance and strong generalization.

📊 Tech Stack

🐍 Python

🧩 CatBoost Regressor

📘 Pandas, NumPy, Scikit-learn

🧾 Sample Output

Top 3 matches for User 10:

Rank	Matched User ID	Similarity Score	Cleanliness	Smoking	Drinking	Sleep Schedule	Hobbies	Budget Range	Preferred Location
🥇	7	84.72%	Low	Yes	No	Flexible	Travel	0.49	Suburbs
🥈	13	80.12%	Medium	Yes	No	Flexible	Reading	0.06	Suburbs
🥉	9	68.63%	High	No	No	Late	Gaming	0.76	Suburbs


![Description](Result.png)

Result Summary:

User 7 is the best match (84.72%), sharing flexible lifestyle habits and travel interests.

User 13 has an 80.12% match, aligning well on habits and location preferences.

User 9 shows a 68.63% match, differing slightly in sleep habits but similar in non-smoking and non-drinking preferences.

👉 Overall, the model successfully recommends roommates with compatible lifestyles, budgets, and preferences.

💡 Future Enhancements

🌐 Add a web interface for user-friendly input and result display.

🗺️ Integrate a map-based PG location search.

🤝 Implement collaborative filtering to enhance recommendation accuracy.

📱 Create a mobile version using Streamlit or Flask API backend.

👩‍💻 Author

P. Priya Varsha
