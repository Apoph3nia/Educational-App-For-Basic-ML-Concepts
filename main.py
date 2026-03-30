import streamlit as st
from algo_linear import show_linear_regression
from intro_page import show_intro_page
from algo_knn import show_knn
from algo_kmeans import show_kmeans
from algo_dbscan import show_dbscan
from algo_rl import show_rl

#   Page Configuration
st.set_page_config(page_title = "ML Educational App", page_icon = "🤖", layout = "wide")

#   Sidebar Navigation
st.sidebar.title("Machine Learning")
st.sidebar.title("🔍 Πλοήγηση")
st.sidebar.info("Επιλογή μιας ενότητας:")

page = st.sidebar.radio(
    "Ενότητες:",
    ["Εισαγωγή", 
     "1. Linear Regression", 
     "2. K-Nearest Neighbors", 
     "3. K-Means Clustering", 
     "4. DBSCAN", 
     "5. Reinforcement Learning"]
)

#   Page Routing
if page == "Εισαγωγή":
    show_intro_page()

elif page == "1. Linear Regression":
    show_linear_regression()

elif page == "2. K-Nearest Neighbors":
    show_knn()

elif page == "3. K-Means Clustering":
    show_kmeans()

elif page == "4. DBSCAN":
    show_dbscan()

elif page == "5. Reinforcement Learning":
    show_rl()

#   Footer in Sidebar
st.sidebar.divider()
st.sidebar.markdown("© 2026 - Πτυχιακή Εργασία")