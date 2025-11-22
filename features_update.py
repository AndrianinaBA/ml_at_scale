import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def update_movie_with_features(
                length_movie,
                embedding_dim,
                data_movie_train,
                movie_embeddings,
                user_embeddings,
                user_biases,
                movie_biases,
                features_embedding,
                genres_of_a_single_movie,
                index_movie,
                p_lambda,
                p_gamma,
                user_tau
):
      
    """
    Placeholder function for updating movie parameters.
    """
    # Implementation of movie update rules goes here
    for n in prange(length_movie):
        num_ratings = len(data_movie_train[n])

        genres_of_current = genres_of_a_single_movie[index_movie[n] : index_movie[n+ 1] ]

        curr_features = features_embedding[genres_of_current]
        sum_features_current = 1.0 / np.sqrt(genres_of_current.shape[0]) * np.sum(curr_features, axis = 0)
        
        if num_ratings == 0:
            continue

        user_idx = np.zeros(num_ratings, dtype=np.int64)
        ratings = np.zeros(num_ratings, dtype=np.float64)

        for i, (m, r) in enumerate(data_movie_train[n]):
            user_idx[i] = m
            ratings[i] = r
        users_embeddings_n = user_embeddings[user_idx]
        users_biases_n = user_biases[user_idx]

        # Bias update
        residuals = ratings - (users_embeddings_n @ movie_embeddings[n]) - users_biases_n
        movie_biases[n] = p_lambda * np.sum(residuals, axis=0) / (p_lambda * num_ratings + p_gamma)

        # Embedding update
        outer_product_users = users_embeddings_n.T @ users_embeddings_n

        weighted_residuals = ratings - users_biases_n - movie_biases[n]
        # print(weighted_residuals.shape)
        diff_movie = users_embeddings_n.T @ weighted_residuals
        # print(diff_movie.shape)

        diff_movie = p_lambda * diff_movie + user_tau * sum_features_current
        denominator = p_lambda * outer_product_users + user_tau * np.identity(embedding_dim)
        movie_embeddings[n, :] = np.linalg.solve(denominator, diff_movie)
    
    return movie_embeddings, movie_biases


# Feature-based updates can be added here in the future
@njit(fastmath=True)
def update_features(
    features_embedding,
    movie_embeddings,
    genres_flat,
    index_genre,
    genres_of_a_single_movie,
    index_movie,
    embedding_dim,
):
    for k in range(features_embedding.shape[0]):
        tmp_vec = np.zeros(embedding_dim, dtype=np.float64)
        sum_inv_squared = 0.0

        actual_genre_list_movies = genres_of_a_single_movie[index_genre[k] : index_genre[k + 1]]
        for mv_idx in actual_genre_list_movies:
            genres_curr_movie = genres_flat[index_movie[mv_idx] : index_movie[mv_idx + 1]] # .ravel()
            # print(genres_curr_movie)
            # print(mv_idx)
            F_n = len(genres_curr_movie)
            inv_length_sqrt = 1.0 / np.sqrt(F_n)
            sum_inv_squared += inv_length_sqrt**2

            sum_features_per_movie = inv_length_sqrt * (np.sum(features_embedding[genres_curr_movie], axis=0) - features_embedding[k])
            tmp_vec += inv_length_sqrt * (movie_embeddings[mv_idx] - sum_features_per_movie)
        # features_embedding[k, :] = tmp_vec / (sum_inv_squared + 1.0)
        for j in range(features_embedding.shape[1]):
            features_embedding[k, j] = tmp_vec[j] / (sum_inv_squared + 1.0)


    return features_embedding
