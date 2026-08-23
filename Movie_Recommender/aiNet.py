from models import User, Movie, MovieRating

import numpy as np
import random

def dot_prod_affinity(antigen, antibody):
    return np.dot(antigen, antibody)

def euclid_distance_affinity(antigen, antibody):
    squared_diff = 0

    for x, y in zip(antigen, antibody):
        squared_diff += (x-y)**2

    distance = np.sqrt(squared_diff)

    return 1 / (1 + distance)

def norm(alc):
    squared_sum = sum([x**2 for x in alc])

    return np.sqrt(squared_sum)


def cosine_sim_affinity(antigen, antibody):
    return np.dot(antigen, antibody)/(norm(antigen)*norm(antibody))

def calculate_movie_affinities(movies, user, affinity_function):
    antigen = np.array(user.preference_profile)
    affinities = []

    for movie in movies:
        antibody = np.array(movie.genres)

        affinity = affinity_function(user, movie)
        affinities.append((movie, affinity))

    return affinities

    
# sort affinities in ascending order using selection sort
def sort_by_affinity(affinities):
    for i in range(len(affinities)):
        for j in range(i, len(affinities)):
            if affinities[i][1] > affinities[j][1]:
                temp_affinity = affinities[i]

                #swap
                affinities[i] = affinities[j]
                affinities[j] = temp_affinity

# Get the hypermutation rate 
def mutation_rate_from_affinity(affinity):
    
    mutation_rate = 1 - affinity

    # clamp the mutaton rate
    if mutation_rate < 0:
        mutation_rate = 0.0

    if mutation_rate > 1:
        mutation_rate = 1.0


def hypermutate(antibody, mutation_rate):
    mutated = antibody.copy()

    rand = random.uniform(0, 1)

    for i in range(len(mutated)):
        if rand < mutation_rate:
            if mutated[i] == 0:
                mutated[i] = 1

            else:
                mutated[i] = 0

    return mutated

def clone_antibody(antibody, number_of_clones):
    
    clones = []

    for i in range(number_of_clones):
        clones.append(antibody.copy())

    return clones

def create_initial_population(movies, population_size):
    
    selected_movies = np.random.choice(movies,size=population_size,replace=False)

    population = []

    for movie in selected_movies:
        population.append(movie)

    return population

def evaluate_population(population, antigen, affinity_function):
    
    for antibody in population:
        
        antibody.affinity = affinity_function(antigen, antibody.genres)

def select_best_antibodies(population, number_to_select):
    
    sort_by_affinity(population)

    return population[:number_to_select]

def clone_and_mutate(selected_antibodies: list[Movie], clones_per_antibody: int, antigen: User, affinity_function):
    
    mutated_clones = []

    for antibody in selected_antibodies:
        
        mutation_rate = mutation_rate_from_affinity(affinity_function(antibody.genres, antigen.preference_profile))

        clones = clone_antibody(antibody.vector,clones_per_antibody)

        for clone in clones:
            
            mutated_vector = hypermutate(clone,mutation_rate)

            mutated_clones.append(Antibody(antibody.movie, mutated_vector))

    return mutated_clones


def evaluate_clones(clones, antigen, affinity_function):
    
    for clone in clones:
        
        clone.affinity = affinity_function(antigen, clone.vector)

def antibody_similarity(antibody_a, antibody_b):
    
    return cosine_affinity(antibody_a.vector, antibody_b.vector)

def suppress_similar_antibodies(population, suppression_threshold):
    
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