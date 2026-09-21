from .data_loader import get_movies, get_ratings, create_users
from .movie_recommender import recommend_for_user
from . import aiNet

if __name__ == "__main__":
    movies = get_movies('data/ml-100k/u.item')
    movie_ratings = get_ratings('data/ml-100k/u.data')

    n_users = 943
    users = create_users(movie_ratings, n_users, movies)

    print(f'{len(movies)} movies, {len(movie_ratings)} ratings, {len(users)} users')

    user = users[92]  # user 93

    print(f'\nUser {user.id} has rated {len(user.rated_movie_ids)} movies')

    recommendations, _, _ = recommend_for_user(user, movies, affinity_function=aiNet.euclid_distance_affinity,
        population_size=100,
        n_iterations=50,
        n_best=15,
        n_clones=8,
        suppression_threshold=0.15,
        stimulation_threshold=0.0,
        top_n_movies=20
    )

    print(f'\nTop {len(recommendations)} recommendations for user {user.id}:')
    for i, (movie, affinity) in enumerate(recommendations):
        print(f'{i+1}. {movie.name} (affinity: {affinity})')