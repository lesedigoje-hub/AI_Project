from .data_loader import get_movies, get_ratings, create_users
from .movie_recommender import recommend_for_user
from . import aiNet

import random


if __name__ == "__main__":
    movies = get_movies('data/ml-100k/u.item')
    movie_ratings = get_ratings('data/ml-100k/u.data')

    n_users = 943
    users = create_users(movie_ratings, n_users, movies)

    print(f'{len(movies)} movies, {len(movie_ratings)} ratings, {len(users)} users')

    # shuffle the movie vectors so the initial arbs aren't just the first movies in the file
    training_vectors = [movie.genres for movie in movies]
    random.shuffle(training_vectors)

    arbs, affinity_threshold = aiNet.train_aiNet(training_data=training_vectors, number_of_initial_arbs=250,
        maximum_resources=250,
        alpha=1.0,
        stimulation_threshold=0.1,
        maximum_iterations=3
    )

    user = users[92]  # get user 93

    print(f'\nUser {user.id} has rated {len(user.rated_movie_ids)} movies')

    recommendations = recommend_for_user(user, arbs, movies, affinity_function=aiNet.cosine_sim_affinity, top_k_arbs=5, top_n_movies=20)

    print(f'\nTop {len(recommendations)} recommendations for user {user.id}:')
    for i in range(len(recommendations)):
        movie, affinity = recommendations[i]
        print(f'{i+1}. {movie.name} (affinity: {affinity})')
