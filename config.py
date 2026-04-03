import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def load_css():
    css_path = BASE_DIR / "netflix_styles.css"
    with open(css_path, "r", encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

def set_netflix_config():
    st.set_page_config(
        page_title="Netflix Churn Prediction",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )