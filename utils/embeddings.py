import streamlit as st
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource
def get_embedding_model():

    return SentenceTransformer(
        MODEL_NAME
    )


def create_embeddings(texts):

    model = get_embedding_model()

    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
    )


def create_query_embedding(query):

    model = get_embedding_model()

    return model.encode(
        [query],
        convert_to_numpy=True,
    )
