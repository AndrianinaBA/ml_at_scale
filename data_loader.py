import csv
from numba.typed import List
from numba import types


def load_main_data(filename):
    """
    Loading the data and returning the main content of the data structure
    """

    # Initialize mappings
    user_to_idx = {}
    idx_to_user = []
    data_user = List()

    movie_to_idx = {}
    idx_to_movie = []
    data_movie = List()

    # Define the tuple type (index, rating)
    pair_type = types.Tuple((types.int64, types.float64))

    # Counters 
    new_user_count = 0
    new_movie_count = 0

    # Loading and writing the data
    with open(filename, 'r') as file:
        my_reader = csv.reader(file, delimiter=',')
        for i, row in enumerate(my_reader):
            if i == 0:
                continue  # Skip header

            user_id = int(row[0])
            movie_id = int(row[1])
            rating = float(row[2])

            # Create new user entry if needed
            if user_id not in user_to_idx:
                user_to_idx[user_id] = new_user_count
                data_user.append(List.empty_list(pair_type))
                idx_to_user.append(user_id)
                new_user_count += 1

            # Create new movie entry if needed
            if movie_id not in movie_to_idx:
                movie_to_idx[movie_id] = new_movie_count
                data_movie.append(List.empty_list(pair_type))
                idx_to_movie.append(movie_id)
                new_movie_count += 1

            # Add rating to user and movie data
            u_idx = user_to_idx[user_id]
            m_idx = movie_to_idx[movie_id]

            data_user[u_idx].append((m_idx, rating))
            data_movie[m_idx].append((u_idx, rating))

    return data_user, data_movie, idx_to_user, idx_to_movie, movie_to_idx, user_to_idx


def load_movie_titles(filename, movie_to_idx):
    """
    Loading the movie titles, genres from a CSV file.
    """

    genres_of_a_single_movie = [[] for _ in range(len(movie_to_idx))]
    list_movies_per_genre = {}
    new_genre_count = 0
    genres_to_idx = {}
    
    # Loading movie titles and genres
    with open(filename, "r") as f:
        movie_reader = csv.reader(f, delimiter=',')
        for i, element in enumerate(movie_reader):
            if i == 0:
                continue
            mv, _, genre = element
            if len(genre) == 0:
                continue

            movie_index = movie_to_idx.get(int(mv))
            if movie_index is None:
                continue

            for gn in genre.split('|'):
                if gn not in list_movies_per_genre:
                    list_movies_per_genre[gn] = []
                    genres_to_idx[gn] = new_genre_count
                    new_genre_count += 1

                list_movies_per_genre[gn].append(movie_index)
                genres_of_a_single_movie[movie_index].append(genres_to_idx[gn])

    return genres_of_a_single_movie, list_movies_per_genre, genres_to_idx