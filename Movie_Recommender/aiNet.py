from models import User, Movie, MovieRating

import numpy as np
import random

def dot_prod_affinity(antigen, antibody) -> float:
    return np.dot(antigen, antibody)

def euclid_distance_affinity(antigen, antibody) -> float:
    squared_diff = 0

    for x, y in zip(antigen, antibody):
        squared_diff += (x-y)**2

    distance = np.sqrt(squared_diff)

    return 1 / (1 + distance)

def norm(alc) -> float:
    squared_sum = sum([x**2 for x in alc])

    return np.sqrt(squared_sum)


def cosine_sim_affinity(antigen, antibody) -> float:
    return np.dot(antigen, antibody)/(norm(antigen)*norm(antibody))

def calculate_movie_affinities(movies: list[Movie], user: User, affinity_function) -> tuple[Movie, float]:
    antigen = np.array(user.preference_profile)
    affinities = []

    for movie in movies:
        antibody = np.array(movie.genres)

        affinity = affinity_function(user, movie)
        affinities.append((movie, affinity))

    return affinities

    
# sort affinities in ascending order using selection sort
def sort_by_affinity(affinities: tuple[Movie, float]) -> None:
    for i in range(len(affinities)):
        for j in range(i, len(affinities)):
            if affinities[i][1] > affinities[j][1]:
                temp_affinity = affinities[i]

                #swap
                affinities[i] = affinities[j]
                affinities[j] = temp_affinity

# Get the hypermutation rate 
def mutation_rate_from_affinity(affinity: float) -> float:
    
    mutation_rate = 1 - affinity

    # clamp the mutaton rate
    if mutation_rate < 0:
        mutation_rate = 0.0

    if mutation_rate > 1:
        mutation_rate = 1.

    return mutation_rate


def hypermutate(genres: list[int], mutation_rate: float) -> list[int]:
    mutated = genres.copy()

    rand = random.uniform(0, 1)

    for i in range(len(mutated)):
        if rand < mutation_rate:
            if mutated[i] == 0:
                mutated[i] = 1

            else:
                mutated[i] = 0

    return mutated

def clone_antibody(antibody: Movie, number_of_clones: int) -> list[Movie]:
    
    clones = []

    for i in range(number_of_clones):
        clones.append(antibody.copy())

    return clones

def create_initial_population(movies: list[Movie], population_size: int) -> list[Movie]:

    # sample from the list of movies without replacement
    selected_movies = np.random.choice(movies,size = population_size,replace = False)

    population = []

    for movie in selected_movies:
        population.append(movie)

    return population

def calc_population_affinities(population: list[Movie], antigen: User, affinity_function):
    
    for antibody in population:
        
        antibody.affinity = affinity_function(antibody, antigen.preference_profile)

def select_best_antibodies(population, number_to_select):
    
    sort_by_affinity(population)

    return population[:number_to_select]

def clone_and_mutate(selected_antibodies: list[Movie], clones_per_antibody: int):
    
    mutated_clones = []

    for antibody in selected_antibodies:
        
        mutation_rate = mutation_rate_from_affinity(antibody.affinity)

        clones = clone_antibody(antibody.genres, clones_per_antibody)

        for clone in clones:
            
            mutated_genres = hypermutate(clone,mutation_rate)

            mutated_clones.append(Movie('', '', mutated_genres))  # challenge

    return mutated_clones


def antibody_similarity(antibody_a, antibody_b, affinity_function):
    
    return affinity_function(antibody_a.vector, antibody_b.vector)

def suppress_similar_antibodies(population: list[Movie], suppression_threshold: float):
    
    survivors = []

    population.sort(key=lambda antibody: antibody.affinity, reverse=True)

    for antibody in population:
        
        too_similar = False

        for survivor in survivors:
            
            similarity = antibody_similarity(antibody, survivor)

            if similarity >= suppression_threshold:
                too_similar = True
                break

        if not too_similar:
            survivors.append(antibody)

    return survivors