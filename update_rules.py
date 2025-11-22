# This section is the core of the update rules for the model
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def update_movie(length_movie,
                 embedding_dim,
                 data_movie_train,
                 movie_embeddings,
                 user_embeddings,
                 user_biases,
                 movie_biases,
                 p_lambda=0.05,
                 p_gamma=0.1,
                 user_tau=0.01):
      
    """
    Placeholder function for updating movie parameters.
    """
    # Implementation of movie update rules goes here
    for n in prange(length_movie):
        num_ratings = len(data_movie_train[n])
        if num_ratings == 0:
            continue

        user_idx = np.empty(num_ratings, dtype=np.int64)
        ratings = np.empty(num_ratings, dtype=np.float64)

        for i, (m, r) in enumerate(data_movie_train[n]):
            user_idx[i] = m
            ratings[i] = r

        users_embeddings_n = user_embeddings[user_idx]
        users_biases_n = user_biases[user_idx]

        # Bias update
        residuals = ratings - (users_embeddings_n @ movie_embeddings[n]) - users_biases_n
        movie_biases[n] = p_lambda * np.sum(residuals) / (p_lambda * num_ratings + p_gamma)

        # Embedding update
        outer_product_users = users_embeddings_n.T @ users_embeddings_n

        weighted_residuals = ratings - users_biases_n - movie_biases[n]
        diff_movie = users_embeddings_n.T @ weighted_residuals

        diff_movie = p_lambda * diff_movie
        denominator = p_lambda * outer_product_users + user_tau * np.eye(embedding_dim)
        movie_embeddings[n, :] = np.linalg.solve(denominator, diff_movie)
    
    return movie_embeddings, movie_biases


@njit(parallel=True, fastmath=True)
def update_user(length_user,
                embedding_dim,
                data_user_train,
                movie_embeddings,
                user_embeddings,
                user_biases,
                movie_biases,
                p_lambda=0.05,
                p_gamma=0.1,
                user_tau=0.01):
      
    """
    Placeholder function for updating movie parameters.
    """
    # Implementation of user update rules goes here
    for m in prange(length_user):
        num_ratings = len(data_user_train[m])
        if num_ratings == 0:
            continue

        movie_idx = np.empty(num_ratings, dtype=np.int64)
        ratings = np.empty(num_ratings, dtype=np.float64)

        for i, (u, r) in enumerate(data_user_train[m]):
            movie_idx[i] = u
            ratings[i] = r

        movies_embeddings_m = movie_embeddings[movie_idx]
        movies_biases_m = movie_biases[movie_idx]

        # Bias update
        residuals = ratings - (movies_embeddings_m @ user_embeddings[m]) - movies_biases_m
        user_biases[m] = p_lambda * np.sum(residuals) / (p_lambda * num_ratings + p_gamma)

        # Embedding update
        outer_product_movies = movies_embeddings_m.T @ movies_embeddings_m

        weighted_residuals = ratings - movies_biases_m - user_biases[m]
        diff_user = movies_embeddings_m.T @ weighted_residuals

        diff_user = p_lambda * diff_user
        denominator = p_lambda * outer_product_movies + user_tau * np.identity(embedding_dim)
        user_embeddings[m, :] = np.linalg.solve(denominator, diff_user)

    return user_embeddings, user_biases
                


@njit(parallel=True, fastmath=True)
def update_movie_0(length_movie,
                 embedding_dim,
                 data_movie_train,
                 movie_embeddings,
                 user_embeddings,
                 user_biases,
                 movie_biases,
                 p_lambda=0.05,
                 p_gamma=0.1,
                 user_tau=0.01):
      
    for n in prange(length_movie):
        # data_movie_train[n] is now a numpy array of shape (k, 2)
        current_data = data_movie_train[n]
        num_ratings = len(current_data)

        if num_ratings == 0:
            continue
        
        # Vectorized extraction (Column 0 is ID, Column 1 is Rating)
        # We must cast ID to int64 because the array is float64
        user_idx = current_data[:, 0].astype(np.int64)
        ratings = current_data[:, 1]

        users_embeddings_n = user_embeddings[user_idx]
        users_biases_n = user_biases[user_idx]

        # Bias update
        residuals = ratings - (users_embeddings_n @ movie_embeddings[n]) - users_biases_n
        movie_biases[n] = p_lambda * np.sum(residuals) / (p_lambda * num_ratings + p_gamma)

        # Embedding update
        outer_product_users = users_embeddings_n.T @ users_embeddings_n

        weighted_residuals = ratings - users_biases_n - movie_biases[n]
        diff_movie = users_embeddings_n.T @ weighted_residuals

        diff_movie = p_lambda * diff_movie
        denominator = p_lambda * outer_product_users + user_tau * np.eye(embedding_dim)
        
        # Solve
        movie_embeddings[n, :] = np.linalg.solve(denominator, diff_movie)
    
    return movie_embeddings, movie_biases


@njit(parallel=True, fastmath=True)
def update_user_0(length_user,
                embedding_dim,
                data_user_train, # Expects List[np.ndarray]
                movie_embeddings,
                user_embeddings,
                user_biases,
                movie_biases,
                p_lambda,
                p_gamma,
                user_tau):
      
    for m in prange(length_user):
        # data_user_train[m] is now a numpy array of shape (k, 2)
        current_data = data_user_train[m]
        num_ratings = len(current_data)
        
        if num_ratings == 0:
            continue

        # Vectorized extraction
        movie_idx = current_data[:, 0].astype(np.int64)
        ratings = current_data[:, 1]

        movies_embeddings_m = movie_embeddings[movie_idx]
        movies_biases_m = movie_biases[movie_idx]

        # Bias update
        residuals = ratings - (movies_embeddings_m @ user_embeddings[m]) - movies_biases_m
        user_biases[m] = p_lambda * np.sum(residuals) / (p_lambda * num_ratings + p_gamma)

        # Embedding update
        outer_product_movies = movies_embeddings_m.T @ movies_embeddings_m

        weighted_residuals = ratings - movies_biases_m - user_biases[m]
        diff_user = movies_embeddings_m.T @ weighted_residuals

        diff_user = p_lambda * diff_user
        denominator = p_lambda * outer_product_movies + user_tau * np.eye(embedding_dim)
        
        # Solve
        user_embeddings[m, :] = np.linalg.solve(denominator, diff_user)

    return user_embeddings, user_biases
