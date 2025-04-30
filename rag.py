import sys
import os
from data_processing.utils import save_chunks_jsonl, load_chunks_jsonl
from retriever.retriever import build_faiss_index, retrieve_context
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from FlagEmbedding import FlagReranker
from data_processing.chunker import get_all_chunks
from dotenv import load_dotenv

# Load the HF token from the .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# Paths
chunk_path = "processed_data/all_chunks.jsonl"


def reranker(query, relevant_chunks, k=3):
    """
    Rerank the top chunks and return the top-k based on similarity to the query using BGE-M3 via FlagReranker.

    Parameters:
    - query (str): The input query.
    - relevant_chunks (list of dict): A list of chunks. Contains the text and the metadata.
    - k (int): The number of top chunks to return.

    Returns:
    - list of str: Top-k ranked chunks based on relevance to the query.
    """

    # Initialize the reranker with BGE-M3 model
    model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

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


def hyDE(question: str, pipeline) -> str:
    """
    Generate a hypothetical answer to the question to be used for retrieval.
    The answer is concise and helps improve the semantic search step in RAG.
    """

    if not question or not isinstance(question, str):
        raise ValueError("Invalid question provided.")

    generation_args = {
        "max_new_tokens": 100,  # Limite réduite pour forcer une réponse concise
        "return_full_text": False,
        "temperature": 0.1,
        "do_sample": True,
        "top_p": 0.9,  # Contrôle supplémentaire sur l'échantillonnage
    }

    prompt = (
        "You are a helpful technical assistant for a company with expertise in satellite, space sector, their teams and internal tools.\n "
        "Generate a single, concise and plausible full-sentence that could answer the following question.\n "
        "Reintroduce the question and important words and vocabulary from the question in your answer to stay on topic.\n "
        "The sentence should sound natural and informative, useful for retrieving relevant documents.\n "
        "Avoid over-speculation or unrelated details.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


    try:
        output = pipeline(prompt, **generation_args)
        answer = output[0]['generated_text'].strip()
    except Exception as e:
        raise RuntimeError(f"HyDE generation failed: {e}")

    return answer

def build_prompt_from_chunks(question: str, chunks: list[str]) -> list[dict]:
    """
    Build chat messages from chunks, embedding metadata in a structured format
    to enhance relevance and grounding for the LLM.

    Parameters:
    - question (str): The user's question.
    - chunks (list of dict): List of retrieved chunks, each with:
        - 'text' (str): The content.
        - 'metadata' (dict): Metadata like source, page, etc.

    Returns:
    - list[dict]: Messages to pass to a chat model (system + user prompts).
    """

    # Build rich context blocks with metadata
    context_blocks = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        meta_lines = "\n".join(f"  - {key.capitalize()}: {value}" for key, value in meta.items() if value)
        formatted_block = (
            f"### Context {i+1}\n"
            f"**Metadata:**\n{meta_lines if meta else '  - None'}\n"
            f"**Content:**\n{chunk['text']}"
        )
        context_blocks.append(formatted_block)

    system_prompt = {
        "role": "system",
        "content": (
            "You are an expert AI assistant for Satlantis, a company in the space sector.\n "
            "Your goal is to provide accurate, concise, and context-grounded answers strictly based on the information provided.\n "
            "Answer concisely unless more detail is needed to answer accurately.\n "
            "Do not hallucinate or make up information.\n "
            "If the information is incomplete or unclear, explain how it impacts your answer. \n "
            "In cases where an answer cannot be fully derived, explain why the full answer isn't available and what additional details would be needed.\n"
            "If the question is vague or lacks necessary context, make sure to explicitly mention the uncertainty "
            "and request additional information or clarification from the user."
        )
    }

    user_prompt = {
        "role": "user",
        "content": (
            "You are provided with context snippets from internal documents.\n"
            "Each one includes metadata and content. Use **only** the given information to answer.\n"
            "If unsure, explain what’s missing.\n\n"
            f"{'---'.join(context_blocks)}\n\n"
            f"### Question:\n{question}"
        )
    }

    return [system_prompt, user_prompt]


if __name__ == "__main__":

    # Parse the argumentsj
    if len(sys.argv) < 2:
        print("Usage: python rag.py <question> [-s] (optional)")
        sys.exit(1)
    question = sys.argv[1]
    # Check if the flag for saving embeddings (-s) is provided
    recompute_embeddings = '-s' in sys.argv[2:]

    # Load the Phi-4-mini-instruct model
    model_path = "microsoft/Phi-4-mini-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    )

    # Generate the hypothetical answer (HyDE)
    hypothetical_answer = hyDE(question, pipe)
    print("HyDE : ", hypothetical_answer)

    # Extract and save chunks
    if recompute_embeddings:
        chunks = get_all_chunks("/home/lucasd/code/rag/data", "/home/lucasd/code/rag/processed_data")
    else:
        chunks = load_chunks_jsonl(chunk_path)


    # Generate the embeddings, remove duplicates and build the FAISS index
    index, embeddings, chunks, embedder = build_faiss_index(chunks, save_embeddings=recompute_embeddings)
    if recompute_embeddings:
        # Save all chunks to a single JSONL file
        save_chunks_jsonl(chunks, chunk_path)
        print(f"Saved all chunks to {chunk_path}")

    # Retrieve the top-20 chunks
    print("Retrieving context...")
    top_chunks = retrieve_context(hypothetical_answer, embedder, chunks, index, k=20)

    # Rerank to get the top-3 chunks
    print("Reranking...")
    reranked_chunks = reranker(question, top_chunks, k=4)

    # Generate the prompt
    messages = build_prompt_from_chunks(question, reranked_chunks)

    # Generate the answer
    generation_args = { 
        "max_new_tokens": 500, 
        "return_full_text": False, 
        "temperature": 0.1,
        "do_sample": True, 
    } 
    output = pipe(messages, **generation_args)
    print(messages)
    print("--------------------------------")
    print(output[0]['generated_text'])

    # Show the source (last top-k chunks)
    # Create a set of filenames from the metadata
    print("\nSources:")
    for chunk in reranked_chunks:
        print(f"- {chunk['metadata']['filename']}")

