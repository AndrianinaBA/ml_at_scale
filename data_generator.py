# This section is reserved for the generation of the vector embeddings
import numpy as np

def generate_features(num_features, embedding_dim, beta=0.01):
    """
    Generates random feature embeddings for a given number of features.

    Parameters:
    - num_features: int, the number of unique features.
    - embedding_dim: int, the dimension of each embedding vector.

    Returns:
    - A numpy array of shape (num_features, embedding_dim) containing the embeddings.
    """

    features = np.sqrt(beta) * np.random.randn(num_features, embedding_dim).astype(np.float64)
    return features


def generate_biases(length_user, length_movie):
    """
    Generating the biases for users/movies.
    """

    user_biases = np.zeros(length_user, dtype=np.float64)
    movie_biases = np.zeros(length_movie, dtype=np.float64)
    return user_biases, movie_biases


def generate_user_embeddings(length_user, embedding_dim):
    """
    Generating the user embeddings.
    """

    user_embeddings = 1.0/np.sqrt(embedding_dim) * np.random.randn(length_user, embedding_dim).astype(np.float64)
    return user_embeddings


def generate_movie_embeddings(length_movie, embedding_dim):
    """
    Generating the movie embeddings.
    """

    movie_embeddings = 1.0/np.sqrt(embedding_dim) * np.random.randn(length_movie, embedding_dim).astype(np.float64)
    return movie_embeddings


# Will use flatten_data from data_transformer.py

def generate_movie_embedings_with_genres(embedding_dim, genres_of_a_single_movie_FLAT, index_movie, features_embedding, movie_tau):
    """
    Generating the movie embeddings with genres.
    """

    num_movies = len(index_movie) - 1
    movie_embeddings = np.zeros((num_movies, embedding_dim), dtype=np.float64)

    for index in range(num_movies):
        slices = genres_of_a_single_movie_FLAT[index_movie[index] : index_movie[index + 1]]
        if len(slices) == 0:
            continue

        loc = 1.0 / np.sqrt(slices.shape[0]) * np.sum(features_embedding[slices], axis=0)
        movie_embeddings[index, :] = 1.0/np.sqrt(embedding_dim) * np.random.multivariate_normal(
            loc,
            cov=movie_tau * np.identity(embedding_dim)
        )
    
    return movie_embeddings