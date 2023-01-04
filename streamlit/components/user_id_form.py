import streamlit as st


def init_session():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = ''


def user_id_form():
    init_session()

    if st.session_state.user_id == '':
        st.text('UserIDを指定してください')

    st.text_input(
        'UserID',
        key='user_id',
        placeholder='1',
        value=st.session_state.user_id)
