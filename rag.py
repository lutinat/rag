import sys
import os
from utils import save_chunks_jsonl, load_chunks_jsonl, load_model
from retriever.retriever import build_faiss_index, retrieve_context
from huggingface_hub import login
from data_processing.data_extraction.chunker import get_all_chunks
from dotenv import load_dotenv
from inference.infer import build_prompt_from_chunks, generate_answer
from retriever.retriever import reranker
from retriever.rewriter import hyDE

# Load the HF token from the .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# Paths
chunk_path = "/home/elduayen/rag/processed_data/all_chunks.jsonl"
embeddings_folder = "/home/elduayen/rag/embeddings"

# Models
phi4_model = "microsoft/Phi-4-mini-instruct"
embedder_model = "intfloat/multilingual-e5-large-instruct"
reranker_model = 'BAAI/bge-reranker-v2-m3'

def rag(question: str, recompute_embeddings: bool = False) -> str:
    """
    Process a question through the RAG pipeline and return an answer.
    
    Args:
        question (str): The question to answer
        recompute_embeddings (bool): Whether to recompute embeddings
        
    Returns:
        str: The generated answer
    """
    # Generate the hypothetical answer (HyDE)
    hyde_answer = hyDE(question, model_name=phi4_model)
    print("HyDE : ", hyde_answer)

    # Extract and save chunks from the documents
    if recompute_embeddings:
        chunks = get_all_chunks("/home/elduayen/rag/data", "/home/elduayen/rag/processed_data")
    else:
        chunks = load_chunks_jsonl(chunk_path)

    # Generate the embeddings, remove duplicates and build the FAISS index
    index, embeddings, chunks, embedder = build_faiss_index(chunks,
                                                            embedder_model,
                                                            embeddings_folder,
                                                            save_embeddings=recompute_embeddings)

    if recompute_embeddings:
        # Save all chunks to a single JSONL file
        save_chunks_jsonl(chunks, chunk_path)
        print(f"Saved all chunks to {chunk_path}")

    # Retrieve the top-20 chunks
    print("Retrieving context...")
    top_chunks = retrieve_context(hyde_answer, embedder, chunks, index, k=30)

    # Rerank to get the top-3 chunks
    print("Reranking...")
    reranked_chunks = reranker(reranker_model, question, top_chunks, k=4)

    # Generate the prompt
    print("Generating prompt...")
    prompt = build_prompt_from_chunks(question, reranked_chunks)

    # Generate the answer
    print("Generating answer...")
    answer = generate_answer(prompt, model_name=phi4_model)

    print("Prompt : ", prompt)
    print("--------------------------------")
    print("Answer : ", answer)

    # Show the sources (last top-k chunks)
    print("\nSources:")
    sources = []
    for chunk in reranked_chunks:
        sources.append(chunk['metadata']['filename'])
        print(f"- {chunk['metadata']['filename']}")
        
    return answer, sources

if __name__ == "__main__":
    # Parse the arguments
    if len(sys.argv) < 2:
        print("Usage: python rag.py <question> [-s] (optional)")
        sys.exit(1)
    question = sys.argv[1]
    # Check if the flag for saving embeddings (-s) is provided
    recompute_embeddings = '-s' in sys.argv[2:]
    
    answer, sources = rag(question, recompute_embeddings)

