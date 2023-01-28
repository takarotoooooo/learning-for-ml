import os
import pandas as pd
from pathlib import Path


class MovieLensDataset:
    def __init__(self):
        self.dir_path = Path(os.path.dirname(__file__)).parent
        self.ratings = pd.read_csv(self.dir_path.joinpath('datasets/ml-latest/ratings.csv'))
        self.__init_users()
        self.__init_movies()
        self.__init_ratings()

        # self.tags = pd.read_csv(self.dir_path.joinpath('datasets/ml-latest/tags.csv'))

    def __init_users(self):
        users_summary = self.ratings.groupby('userId').agg({
            'movieId': ['count'],
            'rating': ['mean', 'min', 'max']
        })
        users_summary.columns = ['ratingMovieCnt', 'ratingMean', 'ratingMin', 'ratingMax']

        user_rating_score_dist = self.ratings.groupby(['userId', 'rating']).count()[['movieId']].unstack()
        user_rating_score_dist.columns = [f'{c[1]}-ScoreCnt' for c in user_rating_score_dist.columns]
        self.users = pd.merge(users_summary, user_rating_score_dist, on='userId').reset_index()

        self.users.reset_index(inplace=True)
        self.users = self.users.rename(columns={'index': 'userIdx'})

    def __init_movies(self):
        self.movies = pd.merge(
            pd.read_csv(self.dir_path.joinpath('datasets/ml-latest/movies.csv')),
            pd.read_csv(self.dir_path.joinpath('datasets/ml-youtube.csv')),
            on=['movieId', 'title'],
            how='left')
        self.movies['genres'] = self.movies['genres'].apply(lambda x: x.split('|'))
        self.movies['thumbnailUrl'] = self.movies['youtubeId'].apply(lambda x: None if x == 'nan' else f'https://img.youtube.com/vi/{x}/hqdefault.jpg')

        movies_summary = self.ratings.groupby('movieId').agg({
            'userId': ['count'],
            'rating': ['mean', 'min', 'max']
        })
        movies_summary.columns = ['ratingUserCnt', 'ratingMean', 'ratingMin', 'ratingMax']

        movie_rating_score_dist = self.ratings.groupby(['movieId', 'rating']).count()[['userId']].unstack()
        movie_rating_score_dist.columns = [f'{c[1]}-ScoreCnt' for c in movie_rating_score_dist.columns]

        self.movies = pd.merge(self.movies, movies_summary, on='movieId')
        self.movies = pd.merge(self.movies, movie_rating_score_dist, on='movieId')

        self.movies.reset_index(inplace=True)
        self.movies = self.movies.rename(columns={'index': 'movieIdx'})

    def __init_ratings(self):
        self.ratings = pd.merge(
            self.ratings, self.users[['userId', 'userIdx']], on='userId', how='inner'
        )
        self.ratings = pd.merge(
            self.ratings, self.movies[['movieId', 'movieIdx']], on='movieId', how='inner'
        )

    def fetch_user_by_index(self, index):
        try:
            return self.users.loc[int(index)]
        except ValueError:
            return None

    def fetch_user_by_id(self, id):
        try:
            id = int(id)
        except ValueError:
            return None

        try:
            return self.users.query('userId == @id').iloc[0]
        except IndexError:
            return None

    def fetch_movie_by_index(self, index):
        try:
            return self.movies.loc[int(index)]
        except ValueError:
            return None

    def fetch_movie_by_id(self, id):
        try:
            id = int(id)
        except ValueError:
            return None

        try:
            return self.movies.query('movieId == @id').iloc[0]
        except IndexError:
            return None

    def user_rating_items(self, user_id):
        return pd.merge(
            self.ratings.query('userId == @user_id'),
            self.movies,
            on=['movieIdx', 'movieId'],
            how='inner'
        )
