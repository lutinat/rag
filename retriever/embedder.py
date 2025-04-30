from sentence_transformers import SentenceTransformer

def load_embedder(model_name="intfloat/multilingual-e5-large-instruct"):
    return SentenceTransformer(model_name)