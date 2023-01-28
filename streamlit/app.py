import streamlit as st
import math
from streamlit_elements import elements, mui
from modules.movie_lens import MovieLensDataset
from components.pagination import Pagination
from components.user_id_form import user_id_form


def init_session():
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 1

    if 'movielens_dataset' not in st.session_state:
        st.session_state.movielens_dataset = MovieLensDataset()


def move_to_page(event, page):
    st.session_state.page_number = page


def show_data():
    st.header('MovieLens')

    movielens_dataset = st.session_state.movielens_dataset

    page_per = 20
    last_page = math.ceil(len(movielens_dataset.movies) / page_per)
    start_idx = (st.session_state.page_number - 1) * page_per
    end_idx = st.session_state.page_number * page_per
    movies = movielens_dataset.movies.iloc[start_idx:end_idx]

    user_id_form()

    item_list_tab, user_info_tab = st.tabs([
        'アイテムリスト',
        'ユーザー情報'
    ])

    with item_list_tab:
        Pagination.pagination(
            key='movies-paginate-top',
            current_page=st.session_state.page_number,
            last_page=last_page,
            on_click=move_to_page)

        with elements('contents'):
            with mui.Grid(container=True, spacing=4):
                for movie in movies.to_dict(orient='records'):
                    with mui.Grid(item=True, xs=6):
                        with mui.Card:
                            mui.CardHeader(
                                avatar=mui.Avatar(movie['movieId']),
                                title=movie['title'])

                            mui.CardMedia(component='img', image=movie['thumbnailUrl'])
                            with mui.CardContent():
                                mui.Typography(round(movie['ratingMean'], 2))
                                mui.Rating(name="read-only", value=movie['ratingMean'], readOnly=True)

        Pagination.pagination(
            key='movies-paginate-bottom',
            current_page=st.session_state.page_number,
            last_page=last_page,
            on_click=move_to_page)

    with user_info_tab:
        user = movielens_dataset.fetch_user_by_id(st.session_state.user_id)
        if user is None:
            return
        st.write(user)
        st.write(
            movielens_dataset.user_rating_items(user.userId).sort_values('timestamp', ascending=False)
        )


def main():
    init_session()
    show_data()


if __name__ == "__main__":
    main()
