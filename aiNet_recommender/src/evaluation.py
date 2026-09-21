import math


def genre_affinity_score(recommendations: list, held_out_movies: list, affinity_function) -> float:
    """
    For each held-out highly-rated movie, find its best-matching recommendation by genre affinity.
    Average these best-match scores across all held-out movies.

    This metric is appropriate for genre-based systems: since many movies share the same
    19-bit genre vector, exact movie matching is unreliable. What matters is whether
    the system recommends movies in the right genre region.
    """
    if not recommendations or not held_out_movies:
        return 0.0

    rec_vectors = [movie.genres for movie, _ in recommendations]
    total = 0.0

    for held_out_movie in held_out_movies:
        best_match = max(
            affinity_function(rec_vec, held_out_movie.genres)
            for rec_vec in rec_vectors
        )
        total += best_match

    return round(total / len(held_out_movies), 4)


def intra_list_diversity(recommendations: list) -> float:
    """
    Average pairwise Euclidean distance between the genre vectors of all recommended movies.
    Higher = more diverse recommendations.
    aiNet's suppression step is expected to score well here by design.
    """
    if len(recommendations) < 2:
        return 0.0

    vectors = [movie.genres for movie, _ in recommendations]
    n = len(vectors)
    total_distance = 0.0
    pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vectors[i], vectors[j])))
            total_distance += dist
            pairs += 1

    return round(total_distance / pairs, 4)


def novelty(recommendations: list, movie_popularity: dict) -> float:
    """
    Average -log2(popularity) of recommended movies.
    Popularity is defined as the fraction of users who have rated the movie.
    Higher novelty = recommending less popular, more niche movies.
    """
    if not recommendations:
        return 0.0

    total = 0.0

    for movie, _ in recommendations:
        pop = movie_popularity.get(movie.id, 0)

        if pop > 0:
            total += -math.log2(pop)
        else:
            # movie has never been rated: treat as maximally novel
            total += -math.log2(1e-10)

    return round(total / len(recommendations), 4)


def evaluate(recommendations: list, held_out_movies: list, affinity_function, movie_popularity: dict) -> dict:
    """Compute all evaluation metrics for a set of recommendations."""
    return {
        'genre_affinity_score': genre_affinity_score(recommendations, held_out_movies, affinity_function),
        'intra_list_diversity': intra_list_diversity(recommendations),
        'novelty': novelty(recommendations, movie_popularity),
    }