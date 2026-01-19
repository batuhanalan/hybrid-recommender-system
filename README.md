🎬 Hybrid Recommender System

This project implements a Hybrid Recommendation System by combining User-Based Collaborative Filtering and Item-Based Collaborative Filtering techniques.
The goal is to generate 10 movie recommendations for a given user:
- 5 recommendations from User-Based model
- 5 recommendations from Item-Based model

📌 Project Overview
Recommendation systems are widely used in platforms such as Netflix, Amazon, and Spotify.
In this project:
- Users with similar watching/rating behavior are identified
- Item similarities are calculated using correlation
- Final recommendations are generated using a hybrid approach

🧠 Methods Used

🔹 User-Based Collaborative Filtering
- User–movie rating matrix created using pivot table
- Users who watched at least 60% of the same movies are considered similar
- Pearson correlation used to measure user similarity
- Weighted average rating is calculated for recommendations
🔹 Item-Based Collaborative Filtering
- Item–item similarity calculated using correlation
- The most recently and highest-rated movie by the user is selected
- Movies most similar to this item are recommended

📂 Project Structure
hybrid-recommender-system/
│
├── hybrid_recommender.py
├── README.md
└── data/
    ├── movie.csv
    └── rating.csv

⚠️ Note: The data/ folder is not included in this repository due to file size limitations.

📊 Dataset
This project uses the MovieLens dataset.
Due to file size constraints, the dataset is not included in the repository.
To run the project locally:
Download the MovieLens dataset
Create a folder named data
Place the files as follows:
data/movie.csv
data/rating.csv

⚙️ Requirements
- Python 3.x
- pandas
- numpy
You can install required packages with:
pip install pandas numpy

▶️ How to Run
python hybrid_recommender.py
The script will:
Randomly select a user
Generate 5 user-based recommendations
Generate 5 item-based recommendations
Output 10 hybrid movie recommendations

🚀 Output Example
User ID: 108170
Most recently highly rated movie: The Matrix (1999)

Recommended Movies:
['Inception', 'Interstellar', 'The Dark Knight', 'Fight Club', ...]

🎯 Key Learning Outcomes
- Building user–item matrices
- Collaborative filtering logic
- Similarity calculations using correlation
- Hybrid recommendation strategy
- Handling large datasets in real-world projects

👤 Author
Batuhan Alan
Actuarial Science Student | Data Science & Recommendation Systems

⭐ Notes
This project is designed for:
- Data Science portfolio
- Academic assignments
- Demonstrating recommender system fundamentals
