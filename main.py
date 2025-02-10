import streamlit as st
import pages
import utils

def main():
    st.set_page_config(
        page_title="Image Classifier",
    )
    if "page" not in st.session_state:
        st.session_state.page = "Home"
        
    utils.css()
    
    if st.session_state.page == "Home":
        pages.home()
    elif st.session_state.page == "Train":
        pages.train_model()
    elif st.session_state.page == "Training Results":
        pages.results()
    else:
        st.error("Page not found.")

if __name__ == '__main__':
    main()
