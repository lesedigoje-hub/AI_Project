import numpy as np


class Movie:
    def __init__(self, name, movie_id, genres):
        self.name = name
        self.id = movie_id

        ''' in the order (unknown, Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy,
                          Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western) '''
        self.genres = genres

    def __str__(self):
        name = f'Movie name: {self.name}\n' 
        id = f'Movie id: {self.id}\n'
        antibody = f'Antibody representation: {self.genres}\n'

        return name + id + antibody


class MovieRating:
    def __init__(self, user_id, movie_id, rating_score):
        self.user_id = user_id
        self.movie_id = movie_id
        self.rating_score = rating_score

    def __str__(self):
        #print(f"Rating user's id: {self.user_id}")
        id = f'Movie id: {self.movie_id}\n'
        score = f'Rating score: {self.rating_score}\n'

        return id + score


class User:
    def __init__(self, user_id, movie_ratings, movie_list):
        self.id = user_id
        self.movie_ratings = movie_ratings
        self.preference_profile = self.create_preference_profile(movie_list)

    def __str__(self):
        id = f'User id: {self.id}\n'
        antigen = f'Antigen representation: {self.preference_profile}\n'
        ratings = f'Number of Movie Ratings: {len(self.movie_ratings)}'

        return id + antigen + ratings

    # Helper methods

    def get_movie_from_rating(self, movie_rating: MovieRating, movie_list: list[Movie]) -> Movie:
        movie_id = movie_rating.movie_id

        for movie in movie_list:
            if movie_rating.movie_id == movie.id:
                rated_movie = movie
                break

        return rated_movie

    def create_preference_profile(self, movie_list: list[Movie]):

        # start with a total rating score of zero for each profile
        total_rating_scores = [0.0 for i in range(19)]

        # capture the numbers of movies that fall under a genre( to be used for calculating average scores)
        movies_per_genre = [0 for i in range(19)]

        for rating in self.movie_ratings:
            movie = self.get_movie_from_rating(rating, movie_list)

            for i in range(len(movie.genres)):
                if movie.genres[i] == 1:

                    # add rating score to the total for the genre
                    total_rating_scores[i] += rating.rating_score
                    movies_per_genre[i] += 1

        # get average rating scores
        average_rating_scores = [0.0 for i in range(19)]

        for i in range(19):
            if movies_per_genre[i] > 0:
                average_rating_scores[i] = total_rating_scores[i] / movies_per_genre[i]
            else:
                average_rating_scores[i] = 0.0

        # normalize the rating scores to 0 <= score <= 1 rounded to three decimal places
        preference_profile = [round(x / 5, 3) for x in average_rating_scores]

        return preference_profile
