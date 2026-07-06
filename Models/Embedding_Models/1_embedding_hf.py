import os

# Set the Hugging Face cache directory BEFORE importing Hugging Face libraries
os.environ["HF_HOME"] = r"C:\Users\iveda\OneDrive\Desktop\gen_ai\huggingface_cache"

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Kolkata is the capital of West Bengal"

vector = embedding.embed_query(text)

print(str(vector))