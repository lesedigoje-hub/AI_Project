from flask import Flask, render_template, request, redirect, url_for

from src.data_loader import get_movies, get_ratings, create_users
from src.models import MovieRating, User
from src.movie_recommender import recommend_for_user
from src import aiNet
from src import knn_recommender
from src import evaluation

import random
import io
import base64
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
KNN_K = 20

# load the movies and the full rating history
movies = get_movies('data/ml-100k/u.item')
movie_ratings = get_ratings('data/ml-100k/u.data')
users = create_users(movie_ratings, N_EXISTING_USERS, movies)

# compute movie popularity as fraction of users who rated each movie used for the novelty metric
rating_counts = {}
for rating in movie_ratings:
    rating_counts[rating.movie_id] = rating_counts.get(rating.movie_id, 0) + 1

movie_popularity = {
    movie_id: count / N_EXISTING_USERS
    for movie_id, count in rating_counts.items()
}

# default aiNet hyperparameters
settings = {
    'population_size': 100,
    'iterations': 50,
    'n_best': 20,
    'n_clones': 10,
    'mutation_lb': -0.1,
    'mutation_ub': 0.1,
    'suppression_threshold': 0.15,
    'stimulation_threshold': 0.0,
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

    # genre vectors of the top recommended movies from the last round,
    # used to seed the next round's population instead of starting from scratch
    'population': [],

    # population snapshots at key checkpoints from the most recent aiNet run
    'snapshots': [],
}


def run_recommendation_for(user):
    affinity_function = AFFINITY_FUNCTIONS.get(settings['affinity_function'], AFFINITY_FUNCTIONS['cosine'])
    mutation_bounds = (settings['mutation_lb'], settings['mutation_ub'])

    recommendations, next_population, snapshots = recommend_for_user(user, movies,
        affinity_function=affinity_function,
        population_size=settings['population_size'],
        n_iterations=settings['iterations'],
        n_best=settings['n_best'],
        n_clones=settings['n_clones'],
        suppression_threshold=settings['suppression_threshold'],
        stimulation_threshold=settings['stimulation_threshold'],
        top_n_movies=settings['top_n_movies'],
        mutation_bounds=mutation_bounds,
        initial_population=guest_state['population'],
    )

    guest_state['population'] = next_population
    guest_state['snapshots'] = snapshots

    return recommendations


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
        settings['stimulation_threshold'] = float(request.form.get('stimulation_threshold'))
        settings['top_n_movies'] = int(request.form.get('top_n_movies'))
        settings['affinity_function'] = request.form.get('affinity_function', 'cosine')
        settings['user_type'] = request.form.get('user_type', 'new')

        # reset carried-over state whenever parameters or user change
        guest_state['population'] = []
        guest_state['snapshots'] = []

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


@app.route('/plots')
def plots():
    snapshots = guest_state['snapshots']
    antigen = guest_state['profile']

    if not snapshots:
        return render_template('plots.html', plot_images=[])

    # collect all vectors across all snapshots and the antigen so PCA is fitted on the full space, keeping all plots on the same axes
    all_vectors = [antigen]
    for snapshot in snapshots:
        if snapshot['population']:
            all_vectors.extend(snapshot['population'])

    pca = PCA(n_components=2)
    pca.fit(all_vectors)

    antigen_2d = pca.transform([antigen])[0]

    plot_images = []

    for snapshot in snapshots:
        if not snapshot['population']:
            continue

        pop_2d = pca.transform(snapshot['population'])
        affinities = snapshot['affinities']

        fig, ax = plt.subplots(figsize=(6, 5))

        scatter = ax.scatter(
            pop_2d[:, 0], pop_2d[:, 1],
            c=affinities,
            cmap='viridis',
            vmin=0, vmax=1,
            alpha=0.7,
            s=40,
            label='Antibodies (movie vectors)',
        )

        ax.scatter(
            antigen_2d[0], antigen_2d[1],
            c='red', marker='*', s=250,
            zorder=5, label='User profile (antigen)',
        )

        plt.colorbar(scatter, ax=ax, label='Affinity to user profile')
        ax.set_title(snapshot['label'])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        plot_images.append({
            'label': snapshot['label'],
            'img': img_base64,
        })

    return render_template('plots.html', plot_images=plot_images)


@app.route('/compare')
def compare():
    all_ratings = guest_state['movie_ratings']

    if len(all_ratings) < 5:
        return render_template('compare.html',
            error="Not enough ratings to run a comparison. Please rate at least 5 movies first.",
            ainet_recs=[], knn_recs=[], ainet_metrics={}, knn_metrics={},
            held_out_movies=[], n_train=0, n_test=0, k=KNN_K,
            affinity_function_name=settings['affinity_function'],
        )

    # 80/20 train/test split on the current user's accumulated ratings
    split_idx = max(1, int(len(all_ratings) * 0.8))
    train_ratings = all_ratings[:split_idx]
    test_ratings = all_ratings[split_idx:]

    # held out relevant movies: only those rated >= 4 count as ground truth positives
    movie_lookup = {movie.id: movie for movie in movies}
    held_out_movies = [
        movie_lookup[r.movie_id]
        for r in test_ratings
        if r.rating_score >= 4 and r.movie_id in movie_lookup
    ]

    # training user preference profile built only from training ratings
    train_user = User(0, train_ratings, movies)

    affinity_function = AFFINITY_FUNCTIONS.get(settings['affinity_function'], AFFINITY_FUNCTIONS['cosine'])
    mutation_bounds = (settings['mutation_lb'], settings['mutation_ub'])

    # aiNet recommendations always fresh (initial_population=None) for a fair comparison
    ainet_recs, _, _ = recommend_for_user(train_user, movies,
        affinity_function=affinity_function,
        population_size=settings['population_size'],
        n_iterations=settings['iterations'],
        n_best=settings['n_best'],
        n_clones=settings['n_clones'],
        suppression_threshold=settings['suppression_threshold'],
        stimulation_threshold=settings['stimulation_threshold'],
        top_n_movies=settings['top_n_movies'],
        mutation_bounds=mutation_bounds,
        initial_population=None,
    )

    # kNN recommendations using the same training ratings
    train_ratings_dict = knn_recommender.build_ratings_dict(train_ratings)
    train_rated_ids = set(train_ratings_dict.keys())

    knn_recs = knn_recommender.recommend_knn(train_ratings_dict, train_rated_ids, users, movies,
        k=KNN_K, top_n=settings['top_n_movies'],
    )

    # evaluate both systems on the same held out set with the same metrics
    ainet_metrics = evaluation.evaluate(ainet_recs, held_out_movies, affinity_function, movie_popularity)
    knn_metrics = evaluation.evaluate(knn_recs, held_out_movies, affinity_function, movie_popularity)

    return render_template('compare.html',
        error=None,
        ainet_recs=ainet_recs,
        knn_recs=knn_recs,
        ainet_metrics=ainet_metrics,
        knn_metrics=knn_metrics,
        held_out_movies=held_out_movies,
        n_train=len(train_ratings),
        n_test=len(test_ratings),
        k=KNN_K,
        affinity_function_name=settings['affinity_function'],
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