from . import aiNet
import random

# create a initial population of random candidate movie vectors
def build_initial_population(candidate_movies: list[list[int]], population_size: int) -> list[list[int]]:
    population = []

    for i in range(population_size):
        random_movie = random.choice(candidate_movies)

        population.append(random_movie.genres)

    return population

# get candidate movies that best match the hypermutated vectors which may have drifted away from valid binary vectors
def match_population_to_movies(population: list[list[float]], candidate_movies: list, affinity_function) -> list[tuple]:
    ''' keep only the single best matching movie per population member so the
        same movie doesn't get recommended multiple times.'''

    best_matching_movies = []
    already_picked = []

    for vector in population:
        best_movie = None
        best_affinity = 0

        for movie in candidate_movies:
            affinity = affinity_function(vector, movie.genres)

            if affinity > best_affinity and movie not in already_picked:
                best_movie = movie
                best_affinity = affinity

        if best_movie is not None:
            best_matching_movies.append((best_movie, best_affinity))
            already_picked.append(best_movie)

    return best_matching_movies
            
# run the aiNet recommendation
def recommend_for_user(user, movies, affinity_function, population_size, n_iterations,
                        n_best, n_clones, suppression_threshold, top_n_movies,
                        mutation_bounds=(-0.1, 0.1)):

    antigen = user.preference_profile

    # only recommend movies the user hasn't already rated
    candidate_movies = [movie for movie in movies if movie.id not in user.rated_movie_ids]

    population = build_initial_population(candidate_movies, population_size)

    for _ in range(n_iterations):
        population = aiNet.ainet_iteration(population, antigen, n_best, n_clones,
                                            suppression_threshold, affinity_function, mutation_bounds)

    scored_movies = match_population_to_movies(population, candidate_movies, affinity_function)

    scored_movies.sort(key=lambda pair: pair[1], reverse=True)

    return scored_movies[:top_n_movies]
