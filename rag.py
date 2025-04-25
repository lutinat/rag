from data_processing.chunker import extract_chunks
from data_processing.pdf_loader import extract_text_from_pdf, extract_metadata_from_pdf, load_txt
from data_processing.utils import save_chunks_jsonl, load_chunks_jsonl
from retriever.retriever import build_faiss_index, retrieve_context
from inference.infer import load_mistral_pipeline, ask_question_with_chunks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer, util
import torch
from huggingface_hub import snapshot_download, login
from pathlib import Path
import sys
import os
from glob import glob
from FlagEmbedding import FlagReranker
from data_processing.chunker import get_all_chunks
from dotenv import load_dotenv

# Load the HF token from the .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# Paths
pdf_path = "/home/lucasd/code/rag/data/proposal_polarisat.pdf"
chunk_path = "/home/lucasd/code/rag/processed_data/all_chunks.jsonl"


def download_model(model_path):
    model_path = Path(model_path)  # Assure-toi que c'est un Path
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"],
            local_dir=model_path
        )
    return model_path

# def rewrite_question(question: str, model, tokenizer) -> str:
#     prompt = (
#         f"Rewrite the following question to make it clearer and more specific, "
#         f"without adding any new information or context:\n\n{question}"
#     )
    
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     output = model.generate(
#         **inputs,
#         max_new_tokens=100,
#         temperature=0.3,
#         do_sample=False,
#         num_beams=4,
#         early_stopping=True,
#     )

#     rewritten = tokenizer.decode(output[0], skip_special_tokens=True)
#     return rewritten.strip()



def reranker(query, relevant_chunks, k=3):
    """
    Rerank the top chunks and return the top-k based on similarity to the query using BGE-M3 via FlagReranker.

    Parameters:
    - query (str): The input query.
    - relevant_chunks (list of str): A list of relevant chunks.
    - k (int): The number of top chunks to return.

    Returns:
    - list of str: Top-k ranked chunks based on relevance to the query.
    """

    # Initialize the reranker with BGE-M3 model
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    
    # Compute similarity scores between the query and each chunk
    scores = reranker.compute_score([[query, chunk] for chunk in relevant_chunks], normalize=True)
    
    # Create a list of chunks with their scores
    scored_chunks = list(zip(relevant_chunks, scores))
    
    # Sort chunks by score in descending order and return the top-k
    ranked_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
    
    # Return the top-k chunks
    return [chunk for chunk, _ in ranked_chunks[:k]]


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
        "You are a helpful technical assistant for a company with expertise in satellite, space sector, their teams and internal tools. "
        "Generate a single, concise and plausible full-sentence that could answer the following question. "
        "Reintroduce important words and vocabulary from the question in your answer to stay on topic. "
        "The sentence should sound natural and informative, useful for retrieving relevant documents. "
        "Avoid over-speculation or unrelated details.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


    try:
        output = pipeline(prompt, **generation_args)
        answer = output[0]['generated_text'].strip()
    except Exception as e:
        raise RuntimeError(f"HyDE generation failed: {e}")

    return answer

def build_chat_messages_from_chunks(question: str, chunks: list[str]) -> list[dict]:
    system_prompt = {
        "role": "system",
        "content": (
            "You are an expert AI assistant for Satlantis, a company in the space sector. "
            "Your goal is to provide accurate, concise, and context-grounded answers strictly based on the information provided. "
            "Do not hallucinate or make up information."
            "If the information is incomplete or unclear, explain how it impacts your answer. "
            "In cases where an answer cannot be fully derived, explain why the full answer isn't available and what additional details would be needed."
        )
    }

    user_prompt = { 
        "role": "user",
        "content": (
            "The following information is provided as context. "
            "Use only this information to answer the question.\n\n"
            + "\n\n".join(f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(chunks))
            + f"\n\nQuestion: {question}"
        )
    }

    return [system_prompt, user_prompt]


if __name__ == "__main__":

    # Parse the arguments
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

    # # Download the model if it's not already downloaded
    # if not os.path.exists(model_path):
    #     print(f"Downloading model to {model_path}...")
    # model_path = download_model(model_path)

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

    # Generate the hypothetical answer (HyDE)
    hypothetical_answer = hyDE(question, pipe)
    print("HyDE : ", hypothetical_answer)

    # Retrieve the top-20 chunks
    print("Retrieving context...")
    top_chunks = retrieve_context(hypothetical_answer, embedder, chunks, index, k=30)

    # Rerank to get the top-3 chunks
    print("Reranking...")
    reranked_chunks = reranker(question, top_chunks, k=3)

    # Generate the prompt
    messages = build_chat_messages_from_chunks(question, reranked_chunks)

    # Generate the answer
    generation_args = { 
        "max_new_tokens": 500, 
        "return_full_text": False, 
        "temperature": 0.2,
        "do_sample": True, 
    } 
    output = pipe(messages, **generation_args)
    print(messages)
    print("--------------------------------")
    print(output[0]['generated_text']) 

