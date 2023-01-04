import os
import sys
import streamlit as st
from pathlib import Path
sys.path.append(os.path.dirname(Path().resolve()))
from components.user_id_form import user_id_form  # noqa: E402


def show_data():
    st.title('FactorizationMachines')
    user_id_form()


def main():
    show_data()


if __name__ == "__main__":
    main()
