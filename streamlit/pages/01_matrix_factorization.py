import os
import sys
import streamlit as st
import numpy as np
from tensorflow.python.keras.models import load_model
from pathlib import Path
sys.path.append(os.path.dirname(Path().resolve()))
from datasets.movie_lens import MovieLensDataset  # noqa: E402


def init_session():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = ''

    if 'movielens_dataset' not in st.session_state:
        st.session_state.movielens_dataset = MovieLensDataset()


def show_data():
    st.title('MatrixFactorization')
    st.text_input(
        'User',
        key='user_id',
        placeholder='1',
        value=st.session_state.user_id)

    movielens_dataset = st.session_state.movielens_dataset
    user = movielens_dataset.fetch_user_by_index(st.session_state.user_id)
    st.write(user)

    movie_idx = movielens_dataset.movies['movieIdx'].values
    user_idx = np.zeros(len(movie_idx))
    user_idx = user_idx + user.index

    model = load_model('/workspace/datasets/matrix-factorization.h5', compile=False)
    # st.write(model.predict([np.array([283214]), np.array([1, 2, 3])], verbose=1))
    st.write(model.predict([user_idx, movie_idx], verbose=1))


def main():
    init_session()
    show_data()


if __name__ == "__main__":
    main()
