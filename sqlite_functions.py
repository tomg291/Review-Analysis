import streamlit as st
import sqlite3
import pandas as pd
import hashlib
from datetime import datetime

@st.cache_resource
def get_db_connection():
    """Creates and returns a single database connection."""
    print("--- Creating DB Connection ---")
    conn = sqlite3.connect("reports.db", check_same_thread=False)
    return conn

@st.cache_data
def generate_file_hash(uploaded_file):
    """Generates a unique SHA256 hash for the content of an uploaded file."""
    uploaded_file.seek(0)
    file_contents = uploaded_file.read()
    return hashlib.sha256(file_contents).hexdigest()

def save_report(conn, file_hash, product_name, aggregated_df, sentences_df, review_count):
    """Saves both aggregated and sentence-level data to the SQLite database."""
    cursor = conn.cursor()
    
    # Insert the main report metadata
    cursor.execute(
        "INSERT INTO reports (id, product_name, created_at, review_count) VALUES (?, ?, ?, ?)",
        (file_hash, product_name, datetime.now(), review_count)
    )
    
    # Insert the aggregated aspect summary data
    for _, row in aggregated_df.iterrows():
        cursor.execute(
            """
            INSERT INTO aspects (report_id, aspect, positive_count, negative_count, count, mention_percentage, net_sentiment_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (file_hash, row["aspect"], row["positive_count"], row["negative_count"], row["positive_count"]+row["negative_count"], row["mention_percentage"], row["net_sentiment_score"])
        )
        
    # Insert the sentence data 
    for _, row in sentences_df.iterrows():
        cursor.execute(
            """
            INSERT INTO sentences (report_id, review_id, sentence, main_topic, sentiment, sentiment_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (file_hash, row["review_id"], row["sentence"], row["main_topic"], row["sentiment"], row["sentiment_score"])
        )
        
    conn.commit()
    print(f"Report {file_hash} and its sentence details saved successfully.")
    return conn

@st.cache_data
def load_report(_conn, file_hash):
    """Loads a report's aggregated aspect data from the SQL database."""
    query1 = """
        SELECT r.product_name, a.aspect, a.positive_count, a.negative_count, a.count, a.mention_percentage, a.net_sentiment_score
        FROM reports r
        JOIN aspects a ON r.id = a.report_id
        WHERE r.id = ?
    """
    query2 = """
        SELECT r.product_name, s.review_id, s.sentence, s.main_topic, s.sentiment, s.sentiment_score
        FROM reports r
        JOIN sentences s ON r.id = s.report_id
        WHERE r.id = ?
    """
    query3 = """
        SELECT created_at, review_count FROM reports WHERE id = ?
    """
    aspect_df = pd.read_sql_query(query1, _conn, params=(file_hash,))
    sentences_df = pd.read_sql_query(query2, _conn, params=(file_hash,))
    summary_df = pd.read_sql_query(query3, _conn, params=(file_hash,))

    if not aspect_df.empty and not sentences_df.empty:
        product_name = aspect_df['product_name'].iloc[0]
        aspect_df = aspect_df.drop(columns=['product_name'])
        sentences_df = sentences_df.drop(columns=['product_name'])
        return product_name, aspect_df, sentences_df, summary_df
        
    return None, None, None, None

@st.cache_data
def check_report_exists(_conn, file_hash):
    """Checks if a report with the given hash exists."""
    return pd.read_sql("SELECT 1 FROM reports WHERE id=?", _conn, params=(file_hash,)).shape[0] > 0

@st.cache_data
def get_all_reports_list(_conn):
    """Retrieves metadata for all stored reports."""
    query = "SELECT id, product_name, created_at, review_count FROM reports ORDER BY created_at DESC"
    return pd.read_sql(query, _conn)