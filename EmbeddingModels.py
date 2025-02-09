import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder


model_dir_path="./models"
# Ensure the model is stored in ./models
os.environ["SENTENCE_TRANSFORMERS_HOME"] = model_dir_path

def load_hugging_face_embeddings():
    # Load model from ./models (no need for cache_dir)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Load the model
embeddings = load_hugging_face_embeddings()


path_to_normal_book_json="sparse_model/bm25_values_new.json"
path_to_ayurved_book_json="sparse_model/bm25_values_ayurved.json"
bm25_encoder_new = BM25Encoder().load(path_to_normal_book_json)
bm25_encoder_ayurved = BM25Encoder().load(path_to_ayurved_book_json)

