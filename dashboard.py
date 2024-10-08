import streamlit as st
import streamlit.components.v1 as components

def dashboard_app():
    # Embed Google Looker Studio report
    st.title("Dashboard Nutrition")

    components.html(
        """
        <iframe
        src="https://lookerstudio.google.com/embed/reporting/a3ebd12e-4c2b-4524-8def-7a6754aa4c6c/page/KyLEE"
        width="100%"
        height="600"
        frameborder="0"
        allowfullscreen>
        </iframe>
        """,
        height=600,
    )
