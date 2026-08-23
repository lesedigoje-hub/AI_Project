import aiNet

def recommend_movies(user, movies, affinity_function, number_of_recommendations=10):
    
    network = run_ainet(user, movies, affinity_function)

    recommendations = []

    rated_movie_ids = {rating.movie_id for rating in user.movie_ratings}

    for antibody in network:
        
        movie = antibody.movie

        if movie.id not in rated_movie_ids:
            
            recommendations.append((movie, antibody.affinity))

        if len(recommendations) == number_of_recommendations:
            break

    return recommendations
