import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

st.title("User Query Mapping Relevance App")

# Use online model instead of local folder for cloud deployment
MODEL_NAME = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

@st.cache_resource(show_spinner=True)
def get_model():
    # Download model from Hugging Face Hub (works in cloud)
    return SentenceTransformer(MODEL_NAME)

# Load model with progress indicator
with st.spinner("Loading AI model... (This may take a moment on first run)"):
    model = get_model()

st.success("Model loaded successfully!")

def calculate_product_score(product_name, tag_sentence, model=model):
    if not isinstance(product_name, str) or not isinstance(tag_sentence, str) or not tag_sentence.strip():
        return np.nan
    product_embedding = model.encode(product_name, convert_to_tensor=True)
    tag_embedding = model.encode(tag_sentence, convert_to_tensor=True)
    similarity = util.cos_sim(product_embedding, tag_embedding).item()
    return similarity

# Add threshold slider for better UX
threshold = 0.22
st.write(f"Current threshold: {threshold}")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Sample of uploaded data:", df.head())
        
    # Drop NA values
    original_rows = len(df)
    df = df.dropna()
    if len(df) < original_rows:
        st.info(f"Removed {original_rows - len(df)} rows with missing values")
        
    # Clean category_path (remove commas if present)
    if 'category_path' in df.columns:
        df['category_path'] = df['category_path'].str.replace(",", " ", regex=False)
    else:
        st.error("❌ Input CSV must contain 'category_path' column.")
        st.stop()
        
    if 'origin_query' not in df.columns:
        st.error("❌ Input CSV must contain 'origin_query' column.")
        st.stop()

    # Compute scores
    with st.spinner("Calculating similarity scores..."):
        df['score'] = [
            calculate_product_score(name, tags)
            for name, tags in zip(df['origin_query'], df['category_path'])
        ]

    # Predict result column (using your threshold)
    df['result'] = [0 if scr < threshold else 1 for scr in df['score']]

    st.write("Preview of prediction results:", df[['origin_query', 'category_path', 'score', 'result']].head())

    # Download button - downloads only the predictions column as CSV
    predictions_csv = df['result'].to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions CSV",
        data=predictions_csv,
        file_name="predictions.csv",
        mime="text/csv"
    )