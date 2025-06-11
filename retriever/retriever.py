import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from retriever.embedder import generate_embeddings
from FlagEmbedding import FlagReranker
from typing import List, Tuple
from utils import free_model_memory
from gpu_profiler import profile_function


def build_faiss_index(chunks: List[dict], 
                      model_name: str, 
                      embeddings_folder: str, 
                      save_embeddings: bool = False,
                      enable_profiling: bool = False) -> Tuple[faiss.IndexFlatIP, np.ndarray, SentenceTransformer]:
    """
    Build a FAISS index from text chunks using sentence embeddings.
    """
    # Apply conditional profiling
    @profile_function("build_faiss_index", enabled=enable_profiling)
    def _build_faiss_index():
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
    
    return _build_faiss_index()


def retrieve_context(question: str, 
                     embedder: SentenceTransformer, 
                     chunks: List[dict], 
                     index: faiss.IndexFlatIP,
                     k: int = 30,
                     enable_profiling: bool = False) -> str:
    """
    Retrieve relevant context for a question using FAISS similarity search.
    """
    # Apply conditional profiling
    @profile_function("retrieve_context", enabled=enable_profiling)
    def _retrieve_context():
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
    
    return _retrieve_context()


def reranker(model_name: str, query, relevant_chunks, k=3, enable_profiling: bool = False):
    """
    Rerank the top chunks and return the top-k based on similarity to the query using BGE-M3 via FlagReranker.
    """
    # Apply conditional profiling
    @profile_function("reranker", enabled=enable_profiling)
    def _reranker():
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
    
    return _reranker()
