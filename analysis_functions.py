from transformers import pipeline
from nltk.tokenize import PunktTokenizer
import requests
import streamlit as st
import numpy as np
import pandas as pd
import json
import math
from datetime import datetime

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME1 = "gemma3:4b"
MODEL_NAME2 = "gemma3:12b"
REVIEWS_PER_BATCH = 35

# Prompt Templates
EXTRACTION_PROMPT_TEMPLATE = """
You are an expert market research analyst AI. Your task is to identify the primary aspects or themes from a list of customer product reviews.
An "aspect" is a specific feature, quality, or topic that customers frequently discuss.
RULES:
1. Analyze the following list of reviews provided between the [START DATA] and [END DATA] tags.
2. Identify no more than 10 of the most frequently discussed aspects.
3. Merge semantically similar topics. For example, 'too thick' and 'heavy' should be combined into a single aspect like 'bulkiness and size'.
4. The name for each aspect should be a short, clear, and descriptive noun phrase.
5. Provide your output *only* in JSON format, as a single list of strings under the key "aspects". Do not add any other explanations or conversational text.
[START DATA]
{reviews_text}
[END DATA]
"""

CONSOLIDATION_PROMPT_TEMPLATE = """
You are an expert data analyst AI. Your task is to clean up and consolidate a list of product aspects generated from multiple batches of reviews.
RULES:
1. Analyze the following list of raw aspects provided between the [START DATA] and [END DATA] tags.
2. Merge similar or duplicate aspects into a single, canonical aspect. For example, merge 'durability', 'protection', and 'strength' into 'protection and durability'.
3. Ensure each final aspect name is a short, clear, and descriptive noun phrase.
4. Return the final, consolidated list of at least 10 of the most important aspects.
5. Provide your output *only* in JSON format, as a single list of strings under the key "aspects". Do not add any other explanations or text.
[START DATA]
{raw_aspects_list}
[END DATA]
"""

def call_ollama(prompt, model):
    """Sends a prompt to the local Ollama API and returns the parsed JSON response."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        response_content = json.loads(response.json()['response'])
        return response_content.get("aspects", [])
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print(f"Raw response: {response.text}")
        return []
    
@st.cache_data
def run_aspect_discovery(reviews):
    """Discovers key aspects from a list of reviews by calling ollama."""
    
    print(f"Processing {len(reviews)} reviews in total.")
    num_batches = math.ceil(len(reviews) / REVIEWS_PER_BATCH)
    all_raw_aspects = []

    # --- STAGE 1: BATCH PROCESSING ---
    print(f"\n--- Starting Stage 1: Extracting aspects from {num_batches} batches ---")
    for i in range(num_batches):
        start_index = i * REVIEWS_PER_BATCH
        end_index = start_index + REVIEWS_PER_BATCH
        batch_reviews = reviews[start_index:end_index]
        
        # Join reviews into a single string for the prompt
        reviews_text = "\n".join(f"- {review}" for review in batch_reviews)
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(reviews_text=reviews_text)
        
        print(f"Processing batch {i+1}/{num_batches}...")
        batch_aspects = call_ollama(prompt, MODEL_NAME1)
        
        if batch_aspects:
            print(f"  -> Found aspects: {batch_aspects}")
            all_raw_aspects.extend(batch_aspects)
        else:
            print(f"  -> No aspects found or error in batch {i+1}.")

    # --- STAGE 2: CONSOLIDATION ---
    print("\n--- Starting Stage 2: Consolidating all found aspects ---")
    if not all_raw_aspects:
        print("No raw aspects were extracted. Cannot consolidate.")
        return []

    print(f"Consolidating {len(all_raw_aspects)} raw aspects...")
    raw_aspects_text = ", ".join(f'"{aspect}"' for aspect in all_raw_aspects)
    consolidation_prompt = CONSOLIDATION_PROMPT_TEMPLATE.format(raw_aspects_list=raw_aspects_text)
    
    final_aspects = call_ollama(consolidation_prompt, MODEL_NAME2)
    print(f"Final consolidated aspects: {final_aspects}")
    return final_aspects

@st.cache_resource
def load_models():
    """Loads and caches the ml models."""
    print("--- Loading models ---")
    zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return zero_shot, sentiment

zero_shot_classifier, sentiment_analysis = load_models()

@st.cache_data
def tokenise_sentences(text):
        """Tokenizes a block of text into individual sentences."""
        if pd.isnull(text):
            return None
        return PunktTokenizer().tokenize(text.strip())

@st.cache_data
def run_full_analysis(uploaded_file):
    """Runs the full analysis pipeline on the uploaded CSV file."""
    # --- Load CSV data --- 
    start_time = datetime.now()
    df = pd.read_csv(uploaded_file)
    count = df.shape[0]
    df = df["reviewText"]
    review_list = df.tolist()

    reference = pd.DataFrame({"review": review_list})
    reference.index.name = "reviewID"
    reference.reset_index(inplace=True)

    # --- Aspect Discovery ---
    #st.spinner("Discovering aspects...")
    aspects = run_aspect_discovery(review_list)
    #aspects.append("general")
    # aspects = ['protection and durability',
    #                 'bulkiness and size',
    #                 'screen quality',
    #                 'design and aesthetics',
    #                 'fit and form factor',
    #                 'cost and value',
    #                 'material quality',
    #                 'port access',
    #                 'warranty and customer service',
    #                 'clip functionality',
    #                 'general']

    # --- Split reviews into individual sentences ---
    #st.spinner("Analyzing review topics...")
    tokenised_reviews = df.apply(tokenise_sentences)
    tokenised_reviews = tokenised_reviews.explode()
    cleaned_reviews = tokenised_reviews[tokenised_reviews.notnull()].astype(str).str.strip()
    cleaned_reviews = cleaned_reviews[cleaned_reviews != ""]  # remove empty strings
    #print(cleaned_reviews.index.tolist())
    # --- Ensure all reviews are non-empty strings ---
    #cleaned_reviews = [str(r).strip() for r in new_review_list if r is not None and str(r).strip() != ""]
    
    # --- Assign each sentences a "main topic" from aspect list ---
    print("\n--- Starting Zero-shot Classification ---")

    score_table = zero_shot_classifier(cleaned_reviews.tolist(),
        candidate_labels=aspects,
        multi_label=True
    )

    # --- Format score table ---
    score_table = pd.DataFrame(score_table)
    #score_table["reviewID"] = tokenised_reviews.index.tolist()
    score_table["reviewID"] = cleaned_reviews.index.tolist()
    score_table = score_table.explode(["labels","scores"])
    score_table = score_table.pivot_table(index=["reviewID","sequence"],
                            columns="labels",
                            values="scores",
                            aggfunc="mean")

    for col in score_table.columns:
        score_table[col] = pd.to_numeric(score_table[col], errors="coerce")
    score_table["main topic"] = score_table.idxmax(axis=1, numeric_only=True)

    # --- I want to filter out rows where the max score is below a certain threshold, say 0.7 ---
    score_table = score_table[score_table.select_dtypes(include="number").max(axis=1) >= 0.7]

    score_table = score_table.reset_index()

    # --- Add sentiment scores to score table ---
    #st.spinner("Finding sentiment...")
    print("\n--- Running Sentiment Analysis ---")
    review_list = score_table["sequence"].tolist()
    sentiment = sentiment_analysis(review_list)
    sentiment_df = pd.DataFrame(sentiment).rename(columns={"label":"aspect sentiment", "score":"sentiment score"})
    score_table = score_table.reset_index().join(sentiment_df)
    score_table["sentiment score"] = np.where(score_table["aspect sentiment"] == "NEGATIVE", score_table["sentiment score"] * -1, score_table["sentiment score"])

    # --- Create aspect summary table ---
    #st.spinner("Creating summary...")
    aspect_summary = score_table.drop(columns=["index", "sequence"])
    aspect_summary = aspect_summary.groupby("main topic").agg(
        aspect_count = ("main topic", "size"),
        positive_count=("aspect sentiment", lambda x: (x == "POSITIVE").sum()),
        negative_count=("aspect sentiment", lambda x: (x == "NEGATIVE").sum()),
        average_sentiment_score=("sentiment score", "mean")
    ).reset_index()
    aspect_summary.rename(columns={"main topic":"aspect"}, inplace=True)
    aspect_summary.sort_values(by='average_sentiment_score', ascending=False, inplace=True)

    analysis_time = datetime.now() - start_time
    return aspect_summary, count, score_table, reference, analysis_time


