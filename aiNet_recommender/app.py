from flask import Flask, render_template, request, redirect, url_for

from src.data_loader import get_movies, get_ratings, create_users
from src.models import MovieRating, User
from src.movie_recommender import recommend_for_user
from src import aiNet

import random

app = Flask(__name__)

GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

AFFINITY_FUNCTIONS = {
    'cosine': aiNet.cosine_sim_affinity,
    'euclidean': aiNet.euclid_distance_affinity,
    'dot': aiNet.dot_prod_affinity,
}

N_EXISTING_USERS = 943
N_MOVIES_TO_RATE = 10

# load the movies and the full rating history
movies = get_movies('data/ml-100k/u.item')
movie_ratings = get_ratings('data/ml-100k/u.data')
users = create_users(movie_ratings, N_EXISTING_USERS, movies)


# default aiNet hyperparameters
settings = {
    'population_size': 100,
    'iterations': 50,
    'n_best': 20,
    'n_clones': 10,
    'mutation_lb': -0.1,
    'mutation_ub': 0.1,
    'suppression_threshold': 0.15,
    'top_n_movies': 10,
    'affinity_function': 'cosine',
    'user_type': 'new',
    'existing_user_id': 1,
}

# state for whichever user (guest or existing) is currently being shown.
guest_state = {
    # every MovieRating the current user has given, historical + this session
    'movie_ratings': [],

    # keeping track of already rated movies
    'ratings': {},

    # (movie, affinity) pairs from the last time recommendations were generated
    'recommendations': [],

    # the current user's preference profile, to be shown in the table on the recommendations page
    'profile': [0.0] * 19,

    # the movies currently offered up to be rated
    'movies_to_rate': [],
}


def run_recommendation_for(user):
    affinity_function = AFFINITY_FUNCTIONS.get(settings['affinity_function'], AFFINITY_FUNCTIONS['cosine'])
    mutation_bounds = (settings['mutation_lb'], settings['mutation_ub'])

    return recommend_for_user(user, movies,
        affinity_function=affinity_function,
        population_size=settings['population_size'],
        n_iterations=settings['iterations'],
        n_best=settings['n_best'],
        n_clones=settings['n_clones'],
        suppression_threshold=settings['suppression_threshold'],
        top_n_movies=settings['top_n_movies'],
        mutation_bounds=mutation_bounds,
    )


def pick_movies_to_rate(n: int):
    already_rated_ids = {rating.movie_id for rating in guest_state['movie_ratings']}
    unrated = [movie for movie in movies if movie.id not in already_rated_ids]

    n = min(n, len(unrated))

    return random.sample(unrated, n)

# update the profile and recommendations, also get fresh random set of unrated movies 
def refresh_state():

    guest_state['ratings'] = {rating.movie_id: rating.rating_score for rating in guest_state['movie_ratings']}

    combined_user = User(0, guest_state['movie_ratings'], movies)
    guest_state['profile'] = combined_user.preference_profile
    guest_state['recommendations'] = run_recommendation_for(combined_user)

    guest_state['movies_to_rate'] = pick_movies_to_rate(N_MOVIES_TO_RATE)


@app.route('/', methods=['GET', 'POST'])
def parameters():
    error_message = None

    if request.method == 'POST':
        settings['population_size'] = int(request.form.get('population_size'))
        settings['iterations'] = int(request.form.get('iterations'))
        settings['n_best'] = int(request.form.get('n_best'))
        settings['n_clones'] = int(request.form.get('n_clones'))
        settings['mutation_lb'] = float(request.form.get('mutation_lb'))
        settings['mutation_ub'] = float(request.form.get('mutation_ub'))
        settings['suppression_threshold'] = float(request.form.get('suppression_threshold'))
        settings['top_n_movies'] = int(request.form.get('top_n_movies'))
        settings['affinity_function'] = request.form.get('affinity_function', 'cosine')
        settings['user_type'] = request.form.get('user_type', 'new')

        # if the user is existing in the dataset
        if settings['user_type'] == 'existing':
            user_id = int(request.form.get('user_id'))

            if user_id is None or not (1 <= user_id <= len(users)):
                error_message = f"Please enter a valid user id between 1 and {len(users)}."
                return render_template('parameters.html', settings=settings, error=error_message)

            settings['existing_user_id'] = user_id

            existing_user = users[user_id - 1]


            # building on top of the user history as they rate more movies
            guest_state['movie_ratings'] = list(existing_user.movie_ratings)
        else:
            # for the new user with no history yet
            guest_state['movie_ratings'] = []

        refresh_state()

        return redirect(url_for('recommendations'))

    return render_template('parameters.html', settings=settings, error=error_message)


@app.route('/recommendations')
def recommendations():
    profile_rows = list(zip(GENRE_NAMES, guest_state['profile']))

    return render_template('recommendations.html', movie_objects=guest_state['movies_to_rate'],
        ratings=guest_state['ratings'],
        recommendations=guest_state['recommendations'],
        profile_rows=profile_rows,
        user_type=settings['user_type'],
        existing_user_id=settings['existing_user_id'],
    )


@app.route('/rate', methods=['POST'])
def rate():
    for movie in guest_state['movies_to_rate']:
        field_name = f'rating_{movie.id}'
        rating_value = request.form.get(field_name)

        if rating_value is not None:
            rating_value = int(rating_value)

            if rating_value > 0:
                guest_state['movie_ratings'].append(MovieRating(0, movie.id, rating_value))

    # every click is a new exposure, update the profile and pull a fresh unrated batch of movies for the next round
    refresh_state()

    return redirect(url_for('recommendations'))


if __name__ == '__main__':
    app.run(debug=True)
