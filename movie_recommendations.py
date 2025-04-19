# Movie Recommendation System using TensorFlow
# ============================================
# Author: RanuK12
# 
# This project implements a collaborative filtering recommendation system
# using TensorFlow's deep learning capabilities to analyze user-movie
# interactions and recommend similar movies based on user preferences.

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
import random
from typing import Dict, List, Tuple

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class MovieRecommender:
    """
    A collaborative filtering movie recommendation system using TensorFlow.
    
    This class handles data preparation, model building, training, and inference
    for a neural network-based movie recommendation engine.
    """
    
    def __init__(self, data_path: str = "data/", embedding_size: int = 50):
        """
        Initialize the movie recommender system.
        
        Args:
            data_path: Path to the directory containing MovieLens dataset
            embedding_size: Size of the embedding vectors for users and movies
        """
        self.data_path = data_path
        self.embedding_size = embedding_size
        self.model = None
        self.user_to_index = {}
        self.movie_to_index = {}
        self.movie_data = None
        self.rating_data = None
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)
    
    def download_dataset(self):
        """
        Download the MovieLens dataset if not already present.
        We're using the small dataset (100k) for demonstration purposes.
        """
        # Check if data already exists
        if os.path.exists(os.path.join(self.data_path, "ml-latest-small")):
            print("Dataset already downloaded.")
            return
        
        # Download and extract dataset
        import requests
        import zipfile
        
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        zip_path = os.path.join(self.data_path, "ml-latest-small.zip")
        
        print("Downloading MovieLens dataset...")
        response = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_path)
        
        # Remove zip file
        os.remove(zip_path)
        print("Dataset downloaded and extracted successfully.")
    
    def load_data(self):
        """
        Load and preprocess the MovieLens dataset.
        """
        print("Loading dataset...")
        
        # Load movie data
        movies_path = os.path.join(self.data_path, "ml-latest-small", "movies.csv")
        self.movie_data = pd.read_csv(movies_path)
        
        # Load ratings data
        ratings_path = os.path.join(self.data_path, "ml-latest-small", "ratings.csv")
        self.rating_data = pd.read_csv(ratings_path)
        
        # Create user and movie indices
        user_ids = self.rating_data["userId"].unique().tolist()
        movie_ids = self.movie_data["movieId"].unique().tolist()
        
        self.user_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
        self.movie_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}
        
        # Add index columns
        self.rating_data["user_index"] = self.rating_data["userId"].map(self.user_to_index)
        self.rating_data["movie_index"] = self.rating_data["movieId"].map(
            lambda x: self.movie_to_index.get(x, -1)
        )
        
        # Remove any rows where movie_index is -1 (not found in movie_data)
        self.rating_data = self.rating_data[self.rating_data["movie_index"] != -1]
        
        print(f"Loaded {len(user_ids)} users and {len(movie_ids)} movies")
        print(f"Total ratings: {len(self.rating_data)}")
        
        # Extract movie genres
        self.movie_data["genres"] = self.movie_data["genres"].str.split("|")
        
        # Extract year from title
        self.movie_data["year"] = self.movie_data["title"].apply(
            lambda x: int(re.findall(r"\((\d{4})\)$", x)[0]) 
            if re.findall(r"\((\d{4})\)$", x) 
            else None
        )
        
        print("Data loaded and preprocessed successfully.")
    
    def prepare_training_data(self, test_size: float = 0.2):
        """
        Prepare training and validation datasets.
        
        Args:
            test_size: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_data, val_data)
        """
        # Split data into training and validation sets
        train_data, val_data = train_test_split(
            self.rating_data, test_size=test_size, random_state=42
        )
        
        # Normalize ratings
        scaler = StandardScaler()
        train_data["rating_scaled"] = scaler.fit_transform(train_data[["rating"]])
        val_data["rating_scaled"] = scaler.transform(val_data[["rating"]])
        
        self.scaler = scaler
        return train_data, val_data
    
    def build_model(self, num_users: int, num_movies: int):
        """
        Build the neural collaborative filtering model using TensorFlow without explicit Keras imports.
        
        Args:
            num_users: Number of unique users in the dataset
            num_movies: Number of unique movies in the dataset
        """
        # User input and embedding
        user_input = tf.keras.Input(shape=(1,), name="user_input")
        user_embedding = tf.keras.layers.Embedding(
            num_users, self.embedding_size, name="user_embedding"
        )(user_input)
        user_vector = tf.keras.layers.Flatten(name="user_flatten")(user_embedding)
        
        # Movie input and embedding
        movie_input = tf.keras.Input(shape=(1,), name="movie_input") 
        movie_embedding = tf.keras.layers.Embedding(
            num_movies, self.embedding_size, name="movie_embedding"
        )(movie_input)
        movie_vector = tf.keras.layers.Flatten(name="movie_flatten")(movie_embedding)
        
        # Combine user and movie embeddings
        concat = tf.keras.layers.Concatenate()([user_vector, movie_vector])
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(128, activation="relu")(concat)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(64, activation="relu")(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
        output = tf.keras.layers.Dense(1)(dropout2)
        
        # Create model
        model = tf.keras.Model(
            inputs=[user_input, movie_input], outputs=output, name="movie_recommender"
        )
        
        # Compile model
        model.compile(
            loss="mean_squared_error",
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["mean_absolute_error"]
        )
        
        self.model = model
        return model
    
    def train_model(self, train_data, val_data, epochs=10, batch_size=64):
        """
        Train the recommendation model.
        
        Args:
            train_data: Training data dataframe
            val_data: Validation data dataframe
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            num_users = len(self.user_to_index)
            num_movies = len(self.movie_to_index)
            self.build_model(num_users, num_movies)
        
        # Prepare input data
        train_user_indices = train_data["user_index"].values
        train_movie_indices = train_data["movie_index"].values
        train_ratings = train_data["rating_scaled"].values
        
        val_user_indices = val_data["user_index"].values
        val_movie_indices = val_data["movie_index"].values
        val_ratings = val_data["rating_scaled"].values
        
        # Train model
        history = self.model.fit(
            [train_user_indices, train_movie_indices],
            train_ratings,
            validation_data=([val_user_indices, val_movie_indices], val_ratings),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=2
                ),
            ],
        )
        
        return history
    
    def plot_training_history(self, history):
        """
        Plot the training and validation loss.
        
        Args:
            history: Training history returned by model.fit()
        """
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history["mean_absolute_error"], label="Training MAE")
        plt.plot(history.history["val_mean_absolute_error"], label="Validation MAE")
        plt.title("Mean Absolute Error")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.show()
    
    def get_movie_recommendations(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Get movie recommendations for a specific user.
        
        Args:
            user_id: The user ID to get recommendations for
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame of recommended movies
        """
        if user_id not in self.user_to_index:
            raise ValueError(f"User ID {user_id} not found in training data")
        
        user_index = self.user_to_index[user_id]
        user_indices = np.array([user_index] * len(self.movie_to_index))
        movie_indices = np.array(list(range(len(self.movie_to_index))))
        
        # Get predictions
        predictions = self.model.predict([user_indices, movie_indices], verbose=0)
        predictions = self.scaler.inverse_transform(predictions)
        
        # Create a dataframe with movie IDs and predictions
        movie_id_to_index = {v: k for k, v in self.movie_to_index.items()}
        movie_ids = [movie_id_to_index[i] for i in range(len(self.movie_to_index))]
        
        # Get the rated movies by the user
        rated_movies = self.rating_data[self.rating_data["userId"] == user_id]["movieId"].tolist()
        
        # Filter out already rated movies
        recommendations = pd.DataFrame({
            "movieId": movie_ids,
            "predicted_rating": predictions.flatten()
        })
        recommendations = recommendations[~recommendations["movieId"].isin(rated_movies)]
        
        # Get the top N recommendations
        top_recommendations = recommendations.sort_values(
            by="predicted_rating", ascending=False
        ).head(top_n)
        
        # Merge with movie data to get titles and other info
        top_recommendations = top_recommendations.merge(
            self.movie_data, on="movieId", how="left"
        )
        
        return top_recommendations[["movieId", "title", "genres", "year", "predicted_rating"]]
    
    def save_model(self, path: str = "movie_recommender_model"):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        self.model.save(path)
        print(f"Model saved to {path}")
        
        # Save mapping dictionaries and scaler
        import pickle
        with open(f"{path}_metadata.pkl", "wb") as f:
            pickle.dump({
                "user_to_index": self.user_to_index,
                "movie_to_index": self.movie_to_index,
                "scaler": self.scaler
            }, f)
        
    def load_model(self, path: str = "movie_recommender_model"):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
        """
        self.model = tf.keras.models.load_model(path)
        
        # Load mapping dictionaries and scaler
        import pickle
        with open(f"{path}_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            self.user_to_index = metadata["user_to_index"]
            self.movie_to_index = metadata["movie_to_index"]
            self.scaler = metadata["scaler"]
        
        print(f"Model loaded from {path}")
    
    def analyze_movie_embeddings(self, n_movies: int = 50):
        """
        Analyze movie embeddings using dimensionality reduction.
        
        Args:
            n_movies: Number of random movies to visualize
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Get the movie embedding layer
        movie_embedding_layer = self.model.get_layer("movie_embedding")
        movie_embeddings = movie_embedding_layer.get_weights()[0]
        
        # Use t-SNE for dimensionality reduction
        from sklearn.manifold import TSNE
        
        # Sample n_movies if there are too many
        if len(movie_embeddings) > n_movies:
            indices = np.random.choice(
                len(movie_embeddings), n_movies, replace=False
            )
        else:
            indices = range(len(movie_embeddings))
        
        sample_embeddings = movie_embeddings[indices]
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        movie_embeddings_2d = tsne.fit_transform(sample_embeddings)
        
        # Get movie titles for the sampled movies
        movie_id_to_index = {v: k for k, v in self.movie_to_index.items()}
        sampled_movie_ids = [movie_id_to_index[i] for i in indices]
        sampled_movies = self.movie_data[self.movie_data["movieId"].isin(sampled_movie_ids)]
        
        # Create a DataFrame for plotting
        movie_coords = pd.DataFrame({
            "x": movie_embeddings_2d[:, 0],
            "y": movie_embeddings_2d[:, 1],
            "title": sampled_movies["title"].values,
            "genres": sampled_movies["genres"].apply(lambda x: "|".join(x) if isinstance(x, list) else x).values,
            "year": sampled_movies["year"].values
        })
        
        # Plot
        plt.figure(figsize=(15, 10))
        sns.scatterplot(x="x", y="y", data=movie_coords, hue="genres", alpha=0.7, s=100)
        
        # Add labels for some points
        for i in range(len(movie_coords)):
            if i % 5 == 0:  # Label every 5th point to avoid clutter
                plt.text(
                    movie_coords.iloc[i]["x"] + 0.01, 
                    movie_coords.iloc[i]["y"] + 0.01,
                    movie_coords.iloc[i]["title"],
                    fontsize=8
                )
        
        plt.title("t-SNE Visualization of Movie Embeddings")
        plt.savefig("movie_embeddings.png")
        plt.show()
        
        return movie_coords

# Example usage
if __name__ == "__main__":
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Download and load data
    recommender.download_dataset()
    recommender.load_data()
    
    # Prepare training data
    train_data, val_data = recommender.prepare_training_data()
    
    # Build and train model
    history = recommender.train_model(train_data, val_data, epochs=5)
    
    # Plot training history
    recommender.plot_training_history(history)
    
    # Get recommendations for a user
    sample_user_id = recommender.rating_data["userId"].sample(1).iloc[0]
    recommendations = recommender.get_movie_recommendations(sample_user_id, top_n=10)
    print(f"\nRecommendations for User {sample_user_id}:")
    print(recommendations)
    
    # Visualize movie embeddings
    recommender.analyze_movie_embeddings(n_movies=100)
    
    # Save model
    recommender.save_model()