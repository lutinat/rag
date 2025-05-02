import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from retriever.embedder import generate_embeddings
from FlagEmbedding import FlagReranker
from typing import List, Tuple
from utils import free_model_memory


def build_faiss_index(chunks: List[dict], 
                      model_name: str, 
                      embeddings_folder: str, 
                      save_embeddings: bool = False) -> Tuple[faiss.IndexFlatIP, np.ndarray, SentenceTransformer]:
    """
    Build a FAISS index from text chunks using sentence embeddings.
    
    Args:
        text_chunks: List of chunks to be indexed
        model_name: Name of the sentence transformer model to use
        save_embeddings: Whether to save the embeddings to a file
        embeddings_file: The file path to save the embeddings if `save_embeddings` is True
        
    Returns:
        Tuple containing:
        - FAISS index
        - Embeddings array
        - Embedder model to use the same model for queries
    """
    # Generate embeddings for the chunks
    updated_embeddings, updated_chunks, embedder = generate_embeddings(embeddings_folder, 
                                                                       chunks, 
                                                                       model_name=model_name, 
                                                                       save_embeddings=save_embeddings)

    # Create and build the FAISS index
    print("Building FAISS index...")
    dimension = updated_embeddings.shape[1]
    faiss.normalize_L2(updated_embeddings)  # Normalize embeddings for cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(updated_embeddings)
    
    return index, updated_embeddings, updated_chunks, embedder


def retrieve_context(question: str, 
                     embedder: SentenceTransformer, 
                     chunks: List[dict], 
                     index: faiss.IndexFlatIP,
                     k: int = 30) -> str:
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
    query_emb = embedder.encode(question, normalize_embeddings=False)
    query_emb = np.array(query_emb).astype('float32')

    # Free up GPU memory
    free_model_memory(embedder)

    # normalize the query embedding
    query_emb = query_emb.reshape(1, -1)
    faiss.normalize_L2(query_emb)
    
    # Search for similar chunks
    distances, indices = index.search(query_emb, k)
    
    # Get the most relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    
    return relevant_chunks


def reranker(model_name: str, query, relevant_chunks, k=3):
    """
    Rerank the top chunks and return the top-k based on similarity to the query using BGE-M3 via FlagReranker.

    Parameters:
    - query (str): The input query.
    - relevant_chunks (list of dict): A list of chunks. Contains the text and the metadata.
    - k (int): The number of top chunks to return.

    Returns:
    - list of str: Top-k ranked chunks based on relevance to the query.
    """

    # Initialize the reranker
    model = FlagReranker(model_name, use_fp16=True)

    # Build context blocks with metadata
    scored_chunks = []
    for i, chunk in enumerate(relevant_chunks):
        meta = chunk.get("metadata", {})
        meta_lines = "\n".join(f"  - {key.capitalize()}: {value}" for key, value in meta.items() if value)
        formatted_block = (
            f"### Document:\n"
            f"**Metadata:**\n{meta_lines}\n"
            f"**Content:**\n{chunk['text']}"
        )
        scored_chunks.append((i, model.compute_score([[query, formatted_block]])))
        
    # Sort and select top-k
    top_k_indices = [i for i, _ in sorted(scored_chunks, key=lambda x: x[1], reverse=True)[:k]]

    return [relevant_chunks[i] for i in top_k_indices]
