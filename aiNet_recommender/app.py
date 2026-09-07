from flask import Flask, render_template, request, redirect, url_for

from src.data_loader import get_movies
from src.models import MovieRating, User
from src.movie_recommender import recommend_for_user
from src import aiNet

import random

app = Flask(__name__)

# load the movies
movies = get_movies('data/ml-100k/u.item')

training_vectors = [movie.genres for movie in movies]
random.shuffle(training_vectors)

arbs, affinity_threshold = aiNet.train_aiNet( training_data=training_vectors,
    number_of_initial_arbs=250,
    maximum_resources=250,
    alpha=1.0,
    stimulation_threshold=0.1,
    maximum_iterations=3
)


def find_movie(movies, keyword):
    for movie in movies:
        if keyword.lower() in movie.name.lower():
            return movie

    return None


# a subset of movies for the guest to rate
subset_movies = ['Star Wars', 'Toy Story', 'Titanic', 'Pulp Fiction', 'Forrest Gump', 'Fargo']

movie_objects = []

for keyword in subset_movies:
    movie = find_movie(movies, keyword)

    if movie is not None:
        movie_objects.append(movie)
    else:
        print(f"warning: couldn't find a movie matching '{keyword}' in the dataset")


# in-memory dict for now
guest_state = {
    # rating score (1-5) for movies the guest has rated so far
    'ratings': {}, 

    # (movie, affinity) pairs from the last time Update was clicked         
    'recommendations': []   
}


@app.route('/')
def home():
    return render_template('home.html', movie_objects=movie_objects,
        ratings=guest_state['ratings'],
        recommendations=guest_state['recommendations']
    )


@app.route('/rate', methods=['POST'])
def rate():
    movie_ratings = []

    for movie in movie_objects:
        field_name = f'rating_{movie.id}'
        rating_value = request.form.get(field_name)

        if rating_value is not None:
            rating_value = int(rating_value)

            if rating_value > 0:
                guest_state['ratings'][movie.id] = rating_value
                movie_ratings.append(MovieRating(0, movie.id, rating_value))

                #remove the movie
            elif movie.id in guest_state['ratings']:
                del guest_state['ratings'][movie.id]

    if len(movie_ratings) > 0:
        guest_user = User(0, movie_ratings, movies)

        recommendations = recommend_for_user(guest_user, arbs, movies, affinity_function=aiNet.cosine_sim_affinity, top_k_arbs=5,
            top_n_movies=6
        )

        guest_state['recommendations'] = recommendations

    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
