import streamlit as st
from streamlit_elements import elements, mui

class Pagination:
    def pagination(key=None, current_page=1, last_page=1, on_click=None):
        with elements(key):
            mui.Pagination(
                variant='outlined',
                count=last_page,
                color='primary',
                showFirstButton=False,
                showLastButton=False,
                siblingCount=3,
                onChange=on_click)
