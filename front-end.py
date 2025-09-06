import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from transformers import pipeline
from nltk.tokenize import PunktTokenizer
import torch

# Load data
with open('Cell_Phones_and_Accessories_5.json', 'r') as file:
    data = [json.loads(line) for line in file]

# Load models
pipe = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

# Set the page title and a wide layout
st.set_page_config(layout="wide")
st.title(" Aspect-AI: Product Insight Platform")

# Create the main layout columns
left_column, right_column = st.columns((1, 1))

# Place the file uploader in a sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a CSV of reviews")

@st.cache_data
def tokenise_sentences(text):
        if pd.isnull(text):
            return None
        return PunktTokenizer().tokenize(text.strip())

@st.cache_data
def run_full_analysis():
    norm_data = pd.json_normalize(data)
    df = pd.DataFrame(norm_data)
    ################## Sort preset product choices #####################
    df_filtered = df[df["asin"] == "B005SUHPO6"].reset_index()
    df_filtered_reviews = df_filtered["reviewText"]

    aspects = [
    'price and value',
    'protection and durability',
    'bulkiness and size',
    'fit and installation',
    'silicone/rubber outer layer quality',
    'screen protector quality',
    'port covers and flaps',
    'aesthetics and color',
    'belt clip/holster',
    'customer service/warranty'
    ]

    # Split reviews into individual sentences
    tokenised_reviews = df_filtered_reviews.apply(tokenise_sentences)
    tokenised_reviews = tokenised_reviews.explode()
    new_review_list = tokenised_reviews.tolist()
    
    # Assign each sentences a "main topic" from aspect list
    score_table = pipe(new_review_list,
        candidate_labels=aspects,
        multi_label=True
    )

    # Format score table
    score_table = pd.DataFrame(score_table)
    score_table = score_table.explode(["labels","scores"])
    score_table = score_table.pivot_table(index="sequence",
                            columns="labels",
                            values="scores",
                            aggfunc="mean")

    for col in score_table.columns:
        score_table[col] = pd.to_numeric(score_table[col], errors="coerce")
    score_table["main topic"] = score_table.idxmax(axis=1, numeric_only=True)

    score_table = score_table.reset_index()

    # Add sentiment scores to score table
    review_list = score_table["sequence"].tolist()
    sentiment = sentiment_analysis(review_list)
    sentiment_df = pd.DataFrame(sentiment).rename(columns={"label":"aspect sentiment", "score":"sentiment score"})
    score_table = score_table.reset_index().join(sentiment_df)
    score_table["sentiment score"] = np.where(score_table["aspect sentiment"] == "NEGATIVE", score_table["sentiment score"] * -1, score_table["sentiment score"])

    # Create aspect summary table
    aspect_summary = score_table.drop(columns=["index", "sequence"])
    aspect_summary = aspect_summary.groupby("main topic").agg(
        aspect_count = ("main topic", "size"),
        positive_count=("aspect sentiment", lambda x: (x == "POSITIVE").sum()),
        negative_count=("aspect sentiment", lambda x: (x == "NEGATIVE").sum()),
        average_sentiment_score=("sentiment score", "mean")
    ).reset_index()
    aspect_summary.rename(columns={"main topic":"aspect"}, inplace=True)
    aspect_summary.sort_values(by='average_sentiment_score', ascending=False, inplace=True)

    return aspect_summary

if uploaded_file:
    # Run the analysis and get the final aggregated data
    aspect_summary = run_full_analysis() #uploaded_file)
    # ... rest of the dashboard code goes here

# In the main part of your script (after the 'if uploaded_file:')
st.header("Executive Summary")
cols = st.columns(4)
with cols[0]:
    st.metric("Total Reviews Analyzed", "5,000")
with cols[1]:
    st.metric("Best Aspect", "Sound Quality")
# ... etc.

# Use Plotly Express for the stacked bar chart
import plotly.express as px
fig = px.bar(aspect_summary["aspect_count"])
st.plotly_chart(fig, use_container_width=True)

# In the main part of your script (after the 'if uploaded_file:')
st.header("Executive Summary")
cols = st.columns(4)
with cols[0]:
    st.metric("Total Reviews Analyzed", "5,000")
with cols[1]:
    st.metric("Best Aspect", "Sound Quality")
# ... etc.

# Use Plotly Express for the stacked bar chart
#import plotly.express as px
#fig = px.bar(aspect_summary, ...)
#st.plotly_chart(fig, use_container_width=True)

# Inside the 'left_column'
st.header("Aspect Breakdown")
st.plotly_chart(fig, use_container_width=True)

# In the sidebar or above the right column
selected_aspect = st.selectbox(
    "Select an aspect to deep dive into:",
    options=aspect_summary["aspect"]
)

# Inside the 'right_column'
st.header(f"Deep Dive: {selected_aspect}")

# Filter your data to get info for the selected aspect
aspect_details = aspect_summary.loc[selected_aspect]

st.subheader("Net Sentiment Score")
st.write(aspect_details['net_sentiment_score'])

st.subheader("AI-Generated Summary")
# ... your summarization logic here ...

st.subheader("Example Review Snippets")
# ... display a table of relevant sentences ...