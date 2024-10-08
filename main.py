import os
import streamlit as st
from food import food_app
from chatbot import chatbot_app
from dashboard import dashboard_app

# Configure Streamlit page settings
st.set_page_config(
    layout='wide',
    page_title='Diabetes Vision',
    page_icon='assets/icon.png',  # Replace with your image URL
    initial_sidebar_state='expanded',
)

# Main function to run the app
def run():
    working_directory = os.path.dirname(os.path.abspath(__file__))    

    # Custom Sidebar Logo
    LOGO_SIDEBAR_URL = "assets/logo2.png"  # Replace with your logo URL

    if LOGO_SIDEBAR_URL: 
        st.sidebar.image(
            LOGO_SIDEBAR_URL,             
            caption='Diabetes Vision',
            use_column_width=True
        )

    # Sidebar with navigation menu
    st.sidebar.title('Menu')
    selected_page = st.sidebar.selectbox("Choose a page", ["Food", "Chatbot", "Dashboard"])

    # Display the selected page content
    if selected_page == "Food":
        food_app()  # Call the function to display your food app
    elif selected_page == "Chatbot":
        chatbot_app()  # Call the function to display your chatbot
    elif selected_page == "Dashboard":
        dashboard_app()

# Run the app
if __name__ == '__main__':
    run()
else:
    st.error('The app failed to initialize. Please report the issue.')
