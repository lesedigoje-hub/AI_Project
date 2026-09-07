from . import aiNet


def recommend_for_user(user, arbs, movies, affinity_function=aiNet.cosine_sim_affinity, top_k_arbs=5, top_n_movies=10):

    # antigen = the user's preference profile
    profile = user.preference_profile

    # find which arbs in the trained network are the closest match to this user using normalized distance

    arb_distances = []

    for arb in arbs:
        distance = aiNet.normalized_distance(profile, arb.vector)
        arb_distances.append((arb, distance))

    arb_distances.sort(key=lambda pair: pair[1])

    closest_arbs = []
    for i in range(top_k_arbs):
        closest_arbs.append(arb_distances[i][0])

    # only recommend movies the user hasn't already rated
    candidate_movies = []
    for movie in movies:
        if movie.id not in user.rated_movie_ids:
            candidate_movies.append(movie)

    # score each candidate movie against the closest arbs
    scored_movies = []

    for movie in candidate_movies:
        best_affinity = None

        for arb in closest_arbs:
            affinity = affinity_function(movie.genres, arb.vector)

            if best_affinity is None or affinity > best_affinity:
                best_affinity = affinity

        scored_movies.append((movie, best_affinity))

    scored_movies.sort(key=lambda pair: pair[1], reverse=True)

    recommendations = []
    for i in range(top_n_movies):
        recommendations.append(scored_movies[i])

    return recommendations
