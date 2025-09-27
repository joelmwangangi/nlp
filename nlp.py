import streamlit as st
import nltk
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from nltk.tokenize import sent_tokenize

# ------------------------------------------------------
# Ensure NLTK resources are available (for deployment)
# ------------------------------------------------------
def ensure_nltk_resources():
    resources = ["punkt", "punkt_tab"]
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res)

ensure_nltk_resources()

# ------------------------------------------------------
# 1. Extractive Summarizer (TextRank using NetworkX)
# ------------------------------------------------------
def textrank_summarizer(text, ratio=0.3):
    sentences = sent_tokenize(text)
    n = len(sentences)

    if n == 0:
        return "⚠️ No sentences found in the input text."

    # Compute sentence similarity using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    sim_matrix = (vectorizer * vectorizer.T).toarray()

    # Build graph and apply PageRank
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    # Rank sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Select top N sentences
    top_n = max(1, int(n * ratio))
    summary_sentences = [s for _, s in ranked_sentences[:top_n]]

    return " ".join(summary_sentences)

# ------------------------------------------------------
# 2. Abstractive Summarizer (BART via Hugging Face)
# ------------------------------------------------------
@st.cache_resource
def load_bart():
    return pipeline("summarization", model="facebook/bart-large-cnn")

abstractive_summarizer = load_bart()

def bart_summarizer(text, ratio=0.5, min_len=20):
    input_len = len(text.split())
    max_len = max(int(input_len * ratio), min_len + 5)

    summary = abstractive_summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )
    return summary[0]['summary_text']

# ------------------------------------------------------
# 3. Streamlit User Interface
# ------------------------------------------------------
st.set_page_config(page_title="AI Text Summarizer", layout="wide")

st.title("📝 AI Text Summarizer")
st.markdown(
    """
    Compare **two paradigms of automatic summarization**:

    - 🔹 **Extractive Summarization (TextRank)**: Picks the most important sentences.
    - 🔹 **Abstractive Summarization (BART)**: Generates a rewritten shorter version.
    """
)

text_input = st.text_area("✍️ Enter your article/text here:", height=250)

ratio = st.slider("📊 Summary Length (% of original text)", 10, 90, 40)

if st.button("🚀 Summarize"):
    if text_input.strip():
        with st.spinner("Generating summaries..."):
            extractive_summary = textrank_summarizer(text_input, ratio=ratio/100)
            abstractive_summary = bart_summarizer(text_input, ratio=ratio/100)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Extractive Summary (TextRank)")
            st.write(extractive_summary)

        with col2:
            st.subheader("📌 Abstractive Summary (BART)")
            st.write(abstractive_summary)
    else:
        st.warning("⚠️ Please enter some text first.")
