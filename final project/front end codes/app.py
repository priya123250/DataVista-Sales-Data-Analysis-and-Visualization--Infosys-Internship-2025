from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv("data/netflix_converted.csv")

st.set_page_config(page_title="Hidden Gems", layout="wide")
st.title("🎬 Final Project: Hidden Gems Movie Recommendations")

# Sidebar
st.sidebar.header("🔎 Filter Movies & Shows")

# Filters
title_input = st.sidebar.text_input("Search by Title")
genre_filter = st.sidebar.multiselect("Genre", sorted(df['listed_in'].dropna().unique()))
country_filter = st.sidebar.multiselect("Country", sorted(df['country'].dropna().unique()))
director_filter = st.sidebar.text_input("Search by Director")
cast_filter = st.sidebar.text_input("Search by Cast")
type_filter = st.sidebar.selectbox("Type", ['All'] + sorted(df['type'].dropna().unique()))
rating_filter = st.sidebar.multiselect("Rating", sorted(df['rating'].dropna().unique()))
year_range = st.sidebar.slider("Release Year", int(df['release_year'].min()), int(df['release_year'].max()), (2010, 2020))

# Filter the dataset
filtered_df = df.copy()

if title_input:
    filtered_df = filtered_df[filtered_df['title'].str.contains(title_input, case=False, na=False)]

if genre_filter:
    filtered_df = filtered_df[filtered_df['listed_in'].apply(lambda x: any(g in x for g in genre_filter))]

if country_filter:
    filtered_df = filtered_df[filtered_df['country'].isin(country_filter)]

if director_filter:
    filtered_df = filtered_df[filtered_df['director'].str.contains(director_filter, case=False, na=False)]

if cast_filter:
    filtered_df = filtered_df[filtered_df['cast'].str.contains(cast_filter, case=False, na=False)]

if type_filter != 'All':
    filtered_df = filtered_df[filtered_df['type'] == type_filter]

if rating_filter:
    filtered_df = filtered_df[filtered_df['rating'].isin(rating_filter)]

filtered_df = filtered_df[filtered_df['release_year'].between(year_range[0], year_range[1])]

# Display results
st.subheader(f"🎯 Results: {len(filtered_df)} Hidden Gems Found")
st.dataframe(filtered_df[['title', 'listed_in', 'country', 'release_year', 'rating', 'type', 'cast', 'director', 'description']])
