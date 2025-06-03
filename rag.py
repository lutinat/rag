import sys
import os
from utils import save_chunks_jsonl, load_chunks_jsonl, load_model, free_model_memory
from retriever.retriever import build_faiss_index, retrieve_context
from huggingface_hub import login
from data_processing.data_extraction.chunker import get_all_chunks
from dotenv import load_dotenv
from inference.infer import build_prompt_from_chunks, generate_answer
from retriever.retriever import reranker
from retriever.rewriter import hyDE
from gpu_profiler import profile_function, profile_block, print_gpu_memory, print_function_summary, save_profile_report, reset_profiler, save_all_plots
import torch

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

def rag(question: str, recompute_embeddings: bool = False, enable_profiling: bool = False, quantization: str = None) -> str:
    """
    Process a question through the RAG pipeline and return an answer.
    
    Args:
        question (str): The question to answer
        recompute_embeddings (bool): Whether to recompute embeddings
        enable_profiling (bool): Whether to enable GPU memory profiling
        quantization (str, optional): Type of quantization to apply to the model.
                                     Options: "4bit", "8bit", None (default: None)
        
    Returns:
        str: The generated answer
    """
    print_gpu_memory("RAG Pipeline Start", enabled=enable_profiling)
    # Reset profiler for new run
    reset_profiler(enabled=enable_profiling)
    
    # Load the Phi-4 model once for the entire pipeline
    print("🔧 Loading Phi-4 model for HyDE and answer generation...")
    with profile_block("model_loading", enabled=enable_profiling):
        model, tokenizer, pipeline_obj = load_model(phi4_model, quantization=quantization)
    
    try:
        # Generate the hypothetical answer (HyDE)
        with profile_block("HyDE_generation", enabled=enable_profiling):
            hyde_answer = hyDE(question, pipeline_obj=pipeline_obj, enable_profiling=enable_profiling)
        print("HyDE : ", hyde_answer)

        # Extract and save chunks from the documents
        with profile_block("chunk_loading", enabled=enable_profiling):
            if recompute_embeddings:
                chunks = get_all_chunks("/home/elduayen/rag/data", "/home/elduayen/rag/processed_data")
            else:
                chunks = load_chunks_jsonl(chunk_path)

        # Generate the embeddings, remove duplicates and build the FAISS index
        with profile_block("embedding_and_indexing", enabled=enable_profiling):
            index, embeddings, chunks, embedder = build_faiss_index(chunks,
                                                                    embedder_model,
                                                                    embeddings_folder,
                                                                    save_embeddings=recompute_embeddings,
                                                                    enable_profiling=enable_profiling)

        if recompute_embeddings:
            # Save all chunks to a single JSONL file
            save_chunks_jsonl(chunks, chunk_path)
            print(f"Saved all chunks to {chunk_path}")

        # Retrieve the top-20 chunks
        with profile_block("context_retrieval", enabled=enable_profiling):
            print("Retrieving context...")
            top_chunks = retrieve_context(hyde_answer, embedder, chunks, index, k=30, enable_profiling=enable_profiling)

        # Rerank to get the top-3 chunks
        with profile_block("reranking", enabled=enable_profiling):
            print("Reranking...")
            reranked_chunks = reranker(reranker_model, question, top_chunks, k=4, enable_profiling=enable_profiling)

        # Generate the prompt
        with profile_block("prompt_generation", enabled=enable_profiling):
            print("Generating prompt...")
            prompt = build_prompt_from_chunks(question, reranked_chunks, enable_profiling=enable_profiling)

        # Generate the answer
        with profile_block("answer_generation", enabled=enable_profiling):
            print("Generating answer...")
            answer = generate_answer(prompt, pipeline_obj=pipeline_obj, enable_profiling=enable_profiling)

        print("Prompt : ", prompt)
        print("--------------------------------")
        print("Answer : ", answer)

        # Show the sources (last top-k chunks)
        print("\nSources:")
        sources = []
        for chunk in reranked_chunks:
            sources.append(chunk['metadata']['filename'])
            print(f"- {chunk['metadata']['filename']}")
        
        print_gpu_memory("RAG Pipeline End", enabled=enable_profiling)
        print_function_summary(enabled=enable_profiling)
        
        return answer, sources
    
    finally:
        # Clean up the model at the end of the pipeline
        free_model_memory(pipeline_obj)
        del model, tokenizer, pipeline_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Parse the arguments
    if len(sys.argv) < 2:
        print("Usage: python rag.py <question> [-s] [-p] [-q <quantization>]")
        print("  -s: Recompute embeddings")
        print("  -p: Enable GPU profiling and save report")
        print("  -q <quantization>: Use quantization (4bit, 8bit)")
        sys.exit(1)
    question = sys.argv[1]
    # Check if the flag for saving embeddings (-s) is provided
    recompute_embeddings = '-s' in sys.argv[2:]
    enable_profiling = '-p' in sys.argv[2:]
    
    # Check for quantization flag
    quantization = None
    if '-q' in sys.argv[2:]:
        q_index = sys.argv.index('-q')
        if q_index + 1 < len(sys.argv):
            quantization = sys.argv[q_index + 1]
            if quantization not in ['4bit', '8bit']:
                print("Error: Quantization must be '4bit' or '8bit'")
                sys.exit(1)
        else:
            print("Error: -q flag requires a quantization type (4bit or 8bit)")
            sys.exit(1)
    
    print_gpu_memory("Initial State", enabled=enable_profiling)
    
    answer, sources = rag(question, recompute_embeddings, enable_profiling, quantization)
    
    # Save profiling report and plots if requested
    if enable_profiling:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        report_filename = f"rag_profile_report.json"
        report_path = save_profile_report(report_filename, timestamp, enabled=enable_profiling)
        
        # Save plots
        try:
            timeline_path, breakdown_path = save_all_plots("rag_profile", timestamp, enabled=enable_profiling)
            if timeline_path and breakdown_path:  # Only print if files were actually saved
                print(f"\n📁 All profiling files saved in folder: gpu_profiling_reports/{timestamp}/")
                print(f"📊 Files created:")
                print(f"   📄 Report: {report_path}")
                print(f"   📈 Timeline: {timeline_path}")
                print(f"   📊 Breakdown: {breakdown_path}")
        except ImportError:
            print("⚠️  matplotlib not available - plots not saved. Install with: pip install matplotlib")
        except Exception as e:
            print(f"⚠️  Error saving plots: {e}")

