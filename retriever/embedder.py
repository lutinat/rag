import os
import numpy as np
import faiss
from typing import List, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
from utils import free_model_memory

def load_embedder(model_name="intfloat/multilingual-e5-large-instruct"):
    return SentenceTransformer(model_name)


def find_close_chunks_faiss(embeddings: np.ndarray, chunks: list[dict], similarity_threshold: float = 0.98):
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

    # Search for top 10 nearest neighbors
    distances, indices = index.search(embeddings, 20)

    processed_pairs = set()
    to_delete = set()
    for i in range(len(chunks)):
        for j_idx, score in zip(indices[i], distances[i]):
            if i == j_idx:
                continue  # Skip self

            pair = tuple(sorted((i, j_idx)))
            if pair in processed_pairs:
                continue

            processed_pairs.add(pair)

            if score >= similarity_threshold:
                to_delete.add(j_idx)  # Remove the duplicate chunk
                print("--------------------------------")
                print(f"Chunk {i} is close to chunk {j_idx} with similarity {score}")
                print(f"Chunk {i}: {chunks[i]}")
                print(f"Chunk {j_idx}: {chunks[j_idx]}")
                print("--------------------------------")

    return to_delete


def generate_embeddings(embeddings_folder: str, 
                        chunks: List[dict], 
                        model_name: str = "intfloat/multilingual-e5-large-instruct", 
                        save_embeddings: bool = False,
                        batch_size: int = 32) -> Tuple[faiss.IndexFlatIP, np.ndarray, SentenceTransformer]:
    
    # Initialize the sentence transformer model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    # Prepare text list
    text_chunks = [c["text"] for c in chunks]

    # Generate embeddings for all chunks
    embeddings_file = os.path.join(embeddings_folder, "embeddings.npy")
    if save_embeddings or not os.path.exists(embeddings_file):
        print("Generating embeddings...")
        embeddings = []
        for i in tqdm(range(0, len(text_chunks), batch_size)):
            # Process in batches
            batch = text_chunks[i:i + batch_size]
            batch_embeddings = model.encode(batch)

            embeddings.append(batch_embeddings)
            
            # Free memory for the current batch after processing
            del batch_embeddings  # Delete temporary batch embeddings
            torch.cuda.empty_cache()  # Clear GPU memory if using GPU

        embeddings = np.vstack(embeddings).astype('float32')
        
        # Free memory for the full embeddings
        del batch  # Delete the batch reference after processing
        torch.cuda.empty_cache()  # Clear GPU memory after all batches processed
    else:
        print("Loading embeddings...")
        embeddings = np.load(embeddings_file)

    # Remove duplicates before saving the embeddings
    if save_embeddings or not os.path.exists(embeddings_file):
        print("Finding duplicate chunks...")
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

    return updated_embeddings, updated_chunks, model