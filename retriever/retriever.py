import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

def build_faiss_index(text_chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[faiss.IndexFlatL2, np.ndarray, SentenceTransformer]:
    """
    Build a FAISS index from text chunks using sentence embeddings.
    
    Args:
        text_chunks: List of text chunks to index
        model_name: Name of the sentence transformer model to use
        
    Returns:
        Tuple containing:
        - FAISS index
        - Embeddings array
        - Sentence transformer model
    """
    # Initialize the sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings for all chunks
    embeddings = model.encode(text_chunks)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create and build the FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, embeddings, model

def retrieve_context(question: str, 
                    embedder: SentenceTransformer, 
                    text_chunks: List[str], 
                    index: faiss.IndexFlatL2,
                    k: int = 5) -> str:
    """
    Retrieve relevant context for a question using FAISS similarity search.
    
    Args:
        question: The question to find context for
        embedder: Sentence transformer model
        text_chunks: List of all text chunks
        index: FAISS index
        k: Number of most similar chunks to retrieve
        
    Returns:
        Concatenated context from the most relevant chunks
    """
    # Generate embedding for the question
    question_embedding = embedder.encode([question])
    question_embedding = np.array(question_embedding).astype('float32')
    
    # Search for similar chunks
    distances, indices = index.search(question_embedding, k)
    
    # Get the most relevant chunks
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    
    return relevant_chunks
