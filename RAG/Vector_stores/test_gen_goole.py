# from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# print(os.getenv("GOOGLE_API_KEY"))
# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# response = client.models.generate_content(
#     model="gemini-2.5-flash-lite",
#     # model="gemini-2.0-flash",
    
#     contents="What is the caital of Europe"
# )

# print(response.text)


### Huggingface
# from langchain_huggingface import HuggingFaceEndpointEmbeddings

# embeddings = HuggingFaceEndpointEmbeddings(
#     # model="BAAI/bge-large-en-v1.5", #1024
#     model="BAAI/bge-base-en-v1.5",  #768
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# )

# vector = embeddings.embed_query("What is LangChain?")
# print(vector)