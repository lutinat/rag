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


def build_chat_messages_from_chunks(question: str, chunks: list[str]) -> list[dict]:
    system_prompt = {
        "role": "system",
        "content": (
            "You are an expert AI assistant for Satlantis. "
            "Your goal is to provide accurate, concise, and context-grounded answers based strictly on the information provided. "
            "If the answer is not explicitly present in the context, respond with 'The answer is not available in the provided context.' "
        )
    }

    # Inject context clearly
    context_message = {
        "role": "user",
        "content": (
            "The following information is provided as context. "
            "Use only this information to answer the next question:\n\n" +
            "\n\n".join(f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks))
        )
    }

    # User question
    question_message = {
        "role": "user",
        "content": f"Question: {question}"
    }

    return [system_prompt, context_message, question_message]


if __name__ == "__main__":

    # Parse the question
    question = sys.argv[1]
    print(f"Question: {question}")

    # # Download the model if it's not already downloaded
    # if not os.path.exists(model_path):
    #     print(f"Downloading model to {model_path}...")
    # model_path = download_model(model_path)

    # # First time: extract and save
    # chunks = get_all_chunks("/home/lucasd/code/rag/data", "/home/lucasd/code/rag/processed_data")

    # Later: just load
    chunks = load_chunks_jsonl(chunk_path)

    # Prepare text list
    text_chunks = [c["text"] for c in chunks]

    # Build FAISS
    index, embeddings, embedder = build_faiss_index(text_chunks)

    top_chunks = retrieve_context(question, embedder, text_chunks, index, k=20)

    # Rerank the top-3 chunks
    reranked_chunks = reranker(question, top_chunks, k=4)

    # # Load the Mistral model
    # tokenizer, model = load_mistral_pipeline(model_path)

    model_path = "microsoft/Phi-4-mini-instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    messages = build_chat_messages_from_chunks(question, reranked_chunks)

    pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    ) 

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


    # # Ask a question
    # response, prompt = ask_question_with_chunks(question, reranked_chunks, tokenizer, model)

    # print(response)

