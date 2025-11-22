# This part is reserved for the flattening of the data structure
import numpy as np


# Both genres_of_a_single_movie and list_movies_per_genre are standard Python lists.
# They should pass though this function if order to be passed in the main loop
def flatten_data(data):
    """
    Data is assumed to be a list of lists.
    This function flattens the data into a single list and creates an index list
    to keep track of the starting index of each sublist in the flattened list.

    Notes : 
    - data_flat : concatenated list of all elements from the sublists
    - index : Has to have len(data) + 1 elements in order for future slicing
    """
    data_flat = []
    index = [0]

    # Flatten the list and create the correspondance for the features first
    for item in data:
        data_flat.extend(item)
        index.append(index[-1] + len(item))

    index = np.array(index, dtype=np.int64)
    data_flat = np.array(data_flat, dtype=np.int64)

    return data_flat, index