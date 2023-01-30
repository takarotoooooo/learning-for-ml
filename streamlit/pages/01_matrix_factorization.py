import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_elements import elements, mui
from pathlib import Path
sys.path.append(os.path.dirname(Path().resolve()))
from modules.movie_lens import MovieLensDataset  # noqa: E402
from modules.matrix_factorization import MatrixFactorization  # noqa: E402
from components.user_id_form import user_id_form  # noqa: E402


def init_session():
    if 'movielens_dataset' not in st.session_state:
        st.session_state.movielens_dataset = MovieLensDataset()

    if 'matrix_factorization_model' not in st.session_state:
        st.session_state.matrix_factorization_model = MatrixFactorization().load_model('/workspace/datasets/matrix-factorization.h5')


def show_data():
    st.title('MatrixFactorization')
    user_id_form()

    movielens_dataset = st.session_state.movielens_dataset
    user = movielens_dataset.fetch_user_by_id(st.session_state.user_id)
    if user is None:
        return

    st.header('推薦アイテム')
    movie_idx = movielens_dataset.movies['movieIdx'].values
    user_idx = np.zeros(len(movie_idx)) + int(user.userIdx)

    movie_with_user_rating = pd.merge(
        movielens_dataset.movies,
        movielens_dataset.ratings.query('userId == @user.userId'),
        on=['movieIdx', 'movieId'],
        how='left'
    )

    result = pd.DataFrame(
        st.session_state.matrix_factorization_model.predict([user_idx, movie_idx], verbose=1),
        columns=['predictedRating']
    )
    result.reset_index(inplace=True)
    result = result.rename(columns={'index': 'movieIdx'})
    result = pd.merge(movie_with_user_rating, result, on='movieIdx', how='left').sort_values('predictedRating', ascending=False)

    with elements('contents'):
        with mui.Grid(container=True, spacing=4):
            for movie in result.query('rating != rating')[0:100].to_dict(orient='records'):
                with mui.Grid(item=True, xs=6):
                    with mui.Card:
                        mui.CardHeader(
                            avatar=mui.Avatar(movie['movieId']),
                            title=movie['title'])

                        mui.CardMedia(component='img', image=movie['thumbnailUrl'])
                        with mui.CardContent():
                            mui.Typography(round(movie['ratingMean'], 2))
                            mui.Rating(name="read-only", value=movie['ratingMean'], readOnly=True)
                            rating = round(movie['rating'], 2)
                            predicted_rating = round(movie['predictedRating'], 2)
                            mui.Typography(f'rating：{rating}')
                            mui.Typography(f'predictedRating：{predicted_rating}')


def main():
    init_session()
    show_data()


if __name__ == "__main__":
    main()
