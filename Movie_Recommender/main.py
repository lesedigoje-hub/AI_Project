from data_loader import get_movies, get_ratings, create_users
import aiNet

import numpy as np


if __name__ == "__main__":
    movies = get_movies('data/ml-100k/u.item')
    print(movies[0])

    antigen = np.array(movies[0].genres)
 
    movie_ratings = get_ratings('data/ml-100k/u.data')

    n_users = 943
    users = create_users(movie_ratings, n_users, movies)

    print(users[942])

    antibody = np.array(users[942].preference_profile)

    print('\nDot product:', aiNet.dot_prod_affinity(antigen, antibody))
    print('Cosine:', aiNet.cosine_sim_affinity(antigen, antibody))
    print('Euclidean affinity:', aiNet.euclid_distance_affinity(antigen, antibody))

    affinities = aiNet.calculate_movie_affinities(movies, users[942], aiNet.cosine_sim_affinity)

    aiNet.sort_by_affinity(affinities)

    print('Top 10 Recommendations for user 943')
    for i in range(10):
        print(f'{i+1}. {affinities[i].name} affinity: {affinities[i][1]}')


    
