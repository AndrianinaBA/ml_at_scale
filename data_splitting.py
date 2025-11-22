from numba.typed import List
from numba import types
from numba import jit

pair_type = types.Tuple((types.int64, types.float64))

# Create typed lists for training and testing data
@jit
def data_split(data_user, data_movie, pair_type=pair_type, ratio=0.8):
    """
    Splitting the data accordingly :
     - 80% for training for data_user and data_movie
     - 20% for testing for data_user and data_movie
    """

    # Create typed lists for train/test splits
    data_by_user_train = List()
    data_by_user_test = List()
    data_by_movie_train = List()
    # data_by_movie_test = List()

    # Split the user data
    for user in data_user:
        user_train = List.empty_list(pair_type)
        user_test = List.empty_list(pair_type)

        lim = len(user)
        curr_counter_user = 0

        for content in user:
            if curr_counter_user / lim <= ratio:
                user_train.append(content)
            else:
                user_test.append(content)
            curr_counter_user += 1

        data_by_user_train.append(user_train)
        data_by_user_test.append(user_test)

    # Split the movie data
    for movie in data_movie:
        movie_train = List.empty_list(pair_type)
        movie_test = List.empty_list(pair_type)

        lim = len(movie)
        curr_counter_movie = 0

        for content in movie:
            if curr_counter_movie / lim <= ratio:
                movie_train.append(content)
            else:
                movie_test.append(content)
            curr_counter_movie += 1

        data_by_movie_train.append(movie_train)
        # data_by_movie_test.append(movie_test)

    return data_by_user_train, data_by_user_test, data_by_movie_train