from analysis_functions import *
from sqlite_functions import *
import plotly.express as px

if "report_loaded" not in st.session_state:
    st.session_state.report_loaded = False
if "report_data" not in st.session_state:
    st.session_state.report_data = None

# Set the page title and a wide layout
st.set_page_config(layout="wide",
                   page_title="Aspect-AI: Product Insight Platform",
                   page_icon=":bar_chart:")

st.title("Aspect-AI: Product Insight Platform")

# # Add mode selector in sidebar
# st.sidebar.header("Data Source")
# mode = st.sidebar.radio(
#     "Choose how to load data:",
#     ["Upload New CSV", "Load Existing Report"],
#     key="data_source_mode"
# )

def plot_bars(aspect_df):
    fig = px.bar(aspect_df, x="aspect", y="aspect_count", title="Aspect Mention Count")
    st.plotly_chart(fig)

def plot_pie(aspect_df):
    fig = px.pie(aspect_df, names="aspect", values="aspect_count", title="Aspect Mention Distribution", hole=0.4)
    st.plotly_chart(fig)

def plot_horizontal_bar(aspect_df):
    aspect_df['sentiment_category'] = aspect_df['average_sentiment_score'].apply(
        lambda score: 'Positive' if score >= 0 else 'Negative'
    )

    fig = px.bar(
        aspect_df.iloc[::-1],
        y="aspect", 
        x="average_sentiment_score",
        color="sentiment_category",
        orientation='h',
        color_discrete_map={'Positive': 'green', 'Negative': 'red'},
        title="Aspect Sentiment Scores"
    )
    st.plotly_chart(fig)

conn = get_db_connection()

# Place the file uploader
# uploaded_file = st.sidebar.file_uploader("Upload a CSV of reviews")

uploaded_file = st.sidebar.file_uploader("Upload a CSV of reviews")

if uploaded_file is None:
    st.info("Please upload a CSV file to get started.")
    st.stop()

if uploaded_file:
    file_hash = generate_file_hash(uploaded_file)
    # Run the analysis and get the final aggregated data
    if check_report_exists(conn, file_hash):
        st.sidebar.success("A report for this product already exists!")
        if st.sidebar.button("Load Existing Report") or st.session_state.report_loaded:
            if not st.session_state.report_loaded:
                product_name, aspect_df, sentences_df, summary_df = load_report(conn, file_hash)

                # format loaded data
                review_count = summary_df["review_count"]
                aspect_summary = aspect_df.rename(columns={"net_sentiment_score":"average_sentiment_score", "count":"aspect_count"})
                score_table = sentences_df.rename(columns={
                    "review_id":"reviewID",
                    "sentence":"sequence",
                    "main_topic":"main topic",
                    "sentiment":"aspect sentiment",
                    "sentiment_score":"sentiment score"
                }).reset_index()
                score_table = score_table.apply(pd.to_numeric, errors="ignore")
                #reference should be the raw reviews associated with the sentences
                reference = pd.DataFrame()  # Placeholder, implement as needed
                
                # store in session
                st.session_state.report_data = {
                    "product_name": product_name,
                    "aspect_summary": aspect_summary,
                    "review_count": review_count,
                    "score_table": score_table,
                    "reference": reference
                }
                st.session_state.report_loaded = True

            # unpack session data
            data = st.session_state.report_data
            product_name = data["product_name"]
            aspect_summary = data["aspect_summary"]
            review_count = data["review_count"]
            score_table = data["score_table"]
            reference = data["reference"]

            st.sidebar.success("Report loaded from database.")
        else:
            st.stop()

    else:
        aspect_summary, review_count, score_table, reference, analysis_time = run_full_analysis(uploaded_file)

        st.success(f"Analysis complete in {analysis_time}. Found {len(aspect_summary)} unique aspects from {review_count} reviews.")

        aspect_summary["mention_percentage"] = (aspect_summary["aspect_count"] / sum(aspect_summary["aspect_count"])) * 100

        ### Prep data for database 
        # Format aspect_summary to be stored in the database
        aggregated_df = aspect_summary.rename(columns={"average_sentiment_score": "net_sentiment_score"})
        
        # Create a name for the product based on the uploaded file name
        product_name = uploaded_file.name.rsplit(".", 1)[0]

        # Format score_table to be stored in the database
        sentences_df = score_table[["reviewID", "sequence", "main topic", "aspect sentiment", "sentiment score"]].rename(columns={
            "reviewID":"review_id",
            "sequence":"sentence",
            "main topic":"main_topic",
            "aspect sentiment":"sentiment",
            "sentiment score":"sentiment_score"
        })

        # create save report in database button
        if st.button("Save Report to Database"):
            save_report(conn, file_hash, product_name, aggregated_df, sentences_df, review_count)
            st.success("Report successfully saved to database!")

    with st.expander("See Aspect Summary DataFrame"):
        st.dataframe(aspect_summary, hide_index=True)
    
    st.header("Executive Summary")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Reviews Analyzed", review_count)
    with cols[1]:
        st.metric("Best Aspect", aspect_summary["aspect"].iloc[0], f"{aspect_summary['average_sentiment_score'].iloc[0]*100:.2f}%")
    with cols[2]:
        st.metric("Worst Aspect", aspect_summary["aspect"].iloc[-1], f"{aspect_summary['average_sentiment_score'].iloc[-1]*100:.2f}%")

    # Use Plotly Express for the stacked bar chart
    #plot_bars(aspect_summary) 
    plot_horizontal_bar(aspect_summary)
    plot_pie(aspect_summary)   

    # Inside the 'left_column'
    st.header("Aspect Breakdown")

    # In the sidebar or above the right column
    selected_aspect = st.selectbox(
        "Select an aspect to deep dive into:",
        options=aspect_summary["aspect"]
    )

    if selected_aspect is None:
        st.info("Please select an aspect to see more details.")
        st.stop()

    if selected_aspect:
        # Inside the 'right_column'
        st.header(f"Deep Dive: {selected_aspect}")

        # Filter your data to get info for the selected aspect

        aspect_details = aspect_summary.set_index("aspect").loc[selected_aspect]
        
        with st.expander("See Aspect Details Data"):
            st.write(aspect_details)

        #st.subheader("Net Sentiment Score")
        st.markdown(f"The aspect **{selected_aspect}** was mentioned in **{aspect_details["mention_percentage"]:.1f}%** of sentences, and carries a net sentiment score of **{aspect_details["average_sentiment_score"]*100:.1f}%**, ranking it {aspect_summary.index.get_loc(aspect_summary[aspect_summary["aspect"] == selected_aspect].index[0]) + 1} out of {len(aspect_summary)} aspects.")
        score_table["max_score"] = score_table.select_dtypes(include="number").drop(columns=["sentiment score", "index", "reviewID"]).max(axis=1)
        score_table_filtered = score_table[score_table["main topic"] == selected_aspect]
        fig = px.histogram(score_table_filtered, x="sentiment score", nbins=20, title=f"Sentiment Score Distribution for '{selected_aspect}'")
        st.plotly_chart(fig)

        st.subheader("Example Review Snippets")
        st.dataframe(score_table_filtered[["reviewID","sequence", "main topic", "aspect sentiment"]], hide_index=True)#.reset_index(drop=True))
        with st.expander("See Full Reviews for Reference"):
            st.dataframe(reference)

