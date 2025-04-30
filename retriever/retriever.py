import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Tuple


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_close_chunks(embeddings: np.ndarray, chunks: list, similarity_threshold: float = 0.98):
    """
    Find and print chunks that are considered "close" based on cosine similarity of their embeddings,
    while avoiding printing duplicate chunks.

    Args:
        embeddings: The embeddings of the text chunks.
        chunks: The list of text chunks (corresponding to the embeddings).
        similarity_threshold: The threshold above which chunks are considered duplicates.

    Prints:
        Close chunks that are above the similarity threshold, removing duplicates.
    
    Returns:
        updated_chunks: List of unique chunks that were not considered duplicates.
        updated_embeddings: Corresponding embeddings of the unique chunks.
    """
    # Calculate cosine similarity matrix
    similarities = cosine_similarity(embeddings)
    
    # A set to keep track of processed chunk indices to avoid printing duplicates
    processed_indices = set()

    for i in range(len(chunks)):
        # Skip if the chunk is already processed
        if i in processed_indices:
            continue

        for j in range(i + 1, len(chunks)):  # Avoid redundant checks (i, j) and (j, i)
            sim = similarities[i, j]
            if sim >= similarity_threshold:
                # Add index to the processed set to avoid checking them again
                processed_indices.add(j)

    return processed_indices


def find_close_chunks_faiss(embeddings: np.ndarray, chunks: list, similarity_threshold: float = 0.98):
    """
    Find and return indices of chunks considered "close" based on cosine similarity of their embeddings,
    using FAISS for efficient search.
    
    Args:
        embeddings: The embeddings of the text chunks.
        chunks: The list of text chunks (corresponding to the embeddings).
        similarity_threshold: The threshold above which chunks are considered duplicates.
    
    Returns:
        processed_indices: Set of indices of duplicate chunks.
    """
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS index (Inner Product after normalization = Cosine similarity)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Search for top 5 nearest neighbors
    distances, indices = index.search(embeddings, 10)  # (n_chunks, 5)

    processed_indices = set()

    for i in range(len(chunks)):
        if i in processed_indices:
            continue

        for j_idx, score in zip(indices[i], distances[i]):
            if i == j_idx:
                continue  # Skip self
            if score >= similarity_threshold:
                processed_indices.add(j_idx)
                print("--------------------------------")
                print(f"Chunk {i} is close to chunk {j_idx} with similarity {score}")
                print(f"Chunk {i}: {chunks[i]}")
                print(f"Chunk {j_idx}: {chunks[j_idx]}")
                print("--------------------------------")

    return processed_indices



def build_faiss_index(chunks: List[str], 
                      model_name: str = "intfloat/multilingual-e5-large-instruct", 
                      save_embeddings: bool = False, 
                      embeddings_folder: str = "/home/lucasd/code/rag/embeddings") -> Tuple[faiss.IndexFlatL2, np.ndarray, SentenceTransformer]:
    """
    Build a FAISS index from text chunks using sentence embeddings.
    
    Args:
        text_chunks: List of text chunks to index
        model_name: Name of the sentence transformer model to use
        save_embeddings: Whether to save the embeddings to a file
        embeddings_file: The file path to save the embeddings if `save_embeddings` is True
        
    Returns:
        Tuple containing:
        - FAISS index
        - Embeddings array
        - Sentence transformer model
    """
    # Initialize the sentence transformer model
    model = SentenceTransformer(model_name)

    # Prepare text list
    text_chunks = [c["text"] for c in chunks]
    
    # Generate embeddings for all chunks
    embeddings_file = os.path.join(embeddings_folder, "embeddings.npy")
    if save_embeddings or not os.path.exists(embeddings_file):
        print("Generating embeddings...")
        embeddings = model.encode(text_chunks)
        embeddings = np.array(embeddings).astype('float32')
    else:
        print("Loading embeddings...")
        embeddings = np.load(embeddings_file)

    # Remove duplicates before saving the embeddings
    if save_embeddings or not os.path.exists(embeddings_file):
        # Find and remove duplicate chunks
        duplicate_indices = find_close_chunks_faiss(embeddings, text_chunks)
        print(f"Found {len(duplicate_indices)} duplicate chunks using cosine similarity")
        
        # Collect the non-duplicate chunks and their embeddings
        updated_embeddings = embeddings[[i for i in range(len(embeddings)) if i not in duplicate_indices]]
        updated_chunks = [chunks[i] for i in range(len(chunks)) if i not in duplicate_indices]
    else:
        updated_embeddings = embeddings
        updated_chunks = chunks
        duplicate_indices = []

    # Save the embeddings
    if save_embeddings or not os.path.exists(embeddings_file):
        print(f"Saving embeddings to {embeddings_file}...")
        np.save(embeddings_file, updated_embeddings)  # Save embeddings as .npy file

    # Create and build the FAISS index
    print("Building FAISS index...")
    dimension = updated_embeddings.shape[1]
    normalized_embeddings = normalize(updated_embeddings, axis=1)
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_embeddings)
    
    return index, normalized_embeddings, updated_chunks, model


def retrieve_context(question: str, 
                     embedder: SentenceTransformer, 
                     chunks: List[str], 
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
    query_emb = embedder.encode(question)
    query_emb = np.array(query_emb).astype('float32')

    # normalize the query embedding
    query_emb = normalize(query_emb, normalize_embeddings=False)
    
    # Search for similar chunks
    distances, indices = index.search(query_emb, k)
    
    # Get the most relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    
    return relevant_chunks
