# 🎬 Neural Movie Recommendation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)
![Made with](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Stars](https://img.shields.io/github/stars/RanuK12/movie-recommender?style=social)

![Training History](https://github.com/RanuK12/movie-recommender/blob/main/training_history.png)
![Movie Embedding Space](https://github.com/RanuK12/movie-recommender/blob/main/movie_embeddings.png)

A movie recommendation system built with TensorFlow that uses collaborative filtering with neural networks to provide personalized movie recommendations.
---

## 🧠 Overview

This project implements a neural collaborative filtering approach to movie recommendations using the [MovieLens dataset](https://grouplens.org/datasets/movielens/).  
The system analyzes user-movie interactions to detect patterns in viewing preferences and makes personalized suggestions based on learned behaviors.

---

## 🚀 Key Features

- ✅ Neural network-based collaborative filtering  
- 🎯 Personalized recommendation generation  
- 🧩 User and movie embedding visualization  
- 🧹 Data preprocessing and exploration  
- 🧠 Model training and evaluation  
- 📊 Visualization of recommendation patterns  

---

## ⚙️ Technical Implementation

The recommendation engine is built with TensorFlow and uses:

- 🔗 Embedding layers for users and movies  
- 🧱 Dense neural network layers for learning interaction patterns  
- 🛡️ Regularization techniques to prevent overfitting  
- 📉 Visualization tools to understand learned relationships  

---

## 🎬 Dataset

We use the [MovieLens Small Dataset](https://grouplens.org/datasets/movielens/latest/) which includes:

- 100,000+ ratings  
- 9,000+ movies  
- 600+ users  
- Ratings from 0.5 to 5.0  
- Metadata: titles, genres, release years  

---

## 🛠️ Installation

---

## ▶️ Usage

from movie_recommender import MovieRecommender

# Initialize the recommender
recommender = MovieRecommender()

# Download and load the dataset
recommender.download_dataset()
recommender.load_data()

# Train the model
train_data, val_data = recommender.prepare_training_data()
history = recommender.train_model(train_data, val_data, epochs=10)

# Get recommendations for a user
recommendations = recommender.get_movie_recommendations(user_id=42, top_n=10)
print(recommendations)

## 📈 Results

- 📉 Mean Absolute Error (MAE): ~0.7 stars  
- ✅ Recommendations verified through cross-validation  
- 🌀 Meaningful clusters in movie embedding space  

---

## 🗺️ Embedding Visualization

![Movie Embedding Space](https://github.com/RanuK12/movie-recommender/blob/main/movie_embeddings.png)

The t‑SNE visualization shows how the model clusters similar movies together in the latent embedding space.  
Genres and themes naturally group, proving the model has learned useful representations.

---

## 🔮 Future Improvements

- 🧠 Integration of content-based features  
- 🧲 Implementation of attention mechanisms  
- ❄️ Support for cold‑start problems  
- 🌐 Simple web interface for live recommendations  

---

## 📦 Requirements

- `tensorflow >= 2.8.0`  
- `numpy >= 1.20.0`  
- `pandas >= 1.3.0`  
- `matplotlib >= 3.4.0`  
- `seaborn >= 0.11.0`  
- `scikit-learn >= 1.0.0`  
- `requests >= 2.27.0`  

---

## 📜 License

[MIT License](https://github.com/RanuK12/movie-recommender/blob/main/LICENSE)

---

## 🙌 Acknowledgements

- [MovieLens](https://grouplens.org/datasets/movielens/) — for the dataset  
- [TensorFlow](https://www.tensorflow.org/) — for the deep learning framework  
