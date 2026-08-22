import streamlit as st
from huggingface_hub import whoami

token = st.secrets["HF_TOKEN"]

info = whoami(token=token)

print("Successfully connected to Hugging Face!")
print("Username:", info["name"])