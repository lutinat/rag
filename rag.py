import sys
import os
from utils import save_chunks_jsonl, load_chunks_jsonl, load_model, free_model_memory
from retriever.retriever import build_faiss_index, retrieve_context
from huggingface_hub import login
from data_processing.data_extraction.chunker import get_all_chunks
from dotenv import load_dotenv
from inference.infer import build_prompt_from_chunks, generate_answer
from retriever.retriever import reranker
from retriever.rewriter import hyDE, rewrite
from gpu_profiler import profile_function, profile_block, print_gpu_memory, print_function_summary, save_profile_report, reset_profiler, save_all_plots
import torch
from typing import Optional, Tuple, Any
from config import ProductionConfig

# Load the HF token from the .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# Paths
CHUNK_PATH = ProductionConfig.CHUNK_PATH
EMBEDDINGS_FOLDER = ProductionConfig.EMBEDDINGS_FOLDER

# Models
phi4_model = "microsoft/Phi-4-mini-instruct"
embedder_model = "intfloat/multilingual-e5-large-instruct"
reranker_model = 'BAAI/bge-reranker-v2-m3'

def rag(question: str, 
        enable_profiling: bool = False, 
        quantization: str = None,
        # New parameters for production usage with preloaded models
        preloaded_models: Optional[dict] = None,
        preloaded_embeddings: Optional[dict] = None,
        conversation_history: Optional[list] = None) -> Tuple[str, list]:
    """
    Process a question through the RAG pipeline and return an answer.
    
    Args:
        question (str): The question to answer
        enable_profiling (bool): Whether to enable GPU memory profiling
        quantization (str, optional): Type of quantization to apply to the model.
                                     Options: "4bit", "8bit", None (default: None)
        preloaded_models (dict, optional): Dictionary with preloaded models for production usage.
                                         Expected keys: 'llm', 'embedder', 'reranker'
        preloaded_embeddings (dict, optional): Dictionary with preloaded embeddings data.
                                             Expected keys: 'index', 'embedder', 'chunks'
        conversation_history (list, optional): List of dictionaries with 'user' and 'assistant' keys.
    Returns:
        Tuple[str, list]: The generated answer and list of source dictionaries with URLs.
                         Each source dict contains: filename, url, title, source_type, display_name
    """
    print_gpu_memory("RAG Pipeline Start", enabled=enable_profiling)
    # Reset profiler for new run
    reset_profiler(enabled=enable_profiling)
    print("Conversation history : ", conversation_history)
    # Determine if we're using preloaded models (production mode) or loading fresh (CLI mode)
    production_mode = preloaded_models is not None
    
    if production_mode:
        print(f"🚀 PRODUCTION MODE: Using preloaded models: {list(preloaded_models.keys()) if preloaded_models else 'None'}")
        if preloaded_embeddings:
            print(f"🚀 PRODUCTION MODE: Using preloaded embeddings: {list(preloaded_embeddings.keys())}")
    else:
        print("🔧 CLI MODE: Loading models fresh...")
    
    # Load or use preloaded Phi-4 model
    if production_mode and 'llm' in preloaded_models:
        print("✅ Using preloaded Phi-4 model...")
        pipeline_obj = preloaded_models['llm']
        model, tokenizer = None, None  # Not needed in production mode
    else:
        print("🔧 Loading Phi-4 model for HyDE and answer generation...")
        with profile_block("model_loading", enabled=enable_profiling):
            model, tokenizer, pipeline_obj = load_model(phi4_model, quantization=quantization)
    
    try:
        # # Rewrite the question to improve grammar and formality while preserving technical accuracy
        # with profile_block("question_rewriting", enabled=enable_profiling):
        #     rewritten_question = rewrite(question, pipeline_obj=pipeline_obj, enable_profiling=enable_profiling)
        # print(f"Original question: {question}")
        # print(f"Rewritten question: {rewritten_question}")
        rewritten_question = question

        # Generate the hypothetical answer (HyDE) using the rewritten question
        with profile_block("HyDE_generation", enabled=enable_profiling):
            hyde_answer = hyDE(rewritten_question, 
                               pipeline_obj=pipeline_obj, 
                               enable_profiling=enable_profiling, 
                               conversation_history=conversation_history)
        print("HyDE : ", hyde_answer)

        # Load chunks from the processed data
        with profile_block("chunk_loading", enabled=enable_profiling):
            if not os.path.exists(CHUNK_PATH):
                raise FileNotFoundError(f"No chunks found at {CHUNK_PATH}. Please run 'python process_data.py' first to process documents.")
            chunks = load_chunks_jsonl(CHUNK_PATH)

        # Handle embeddings and indexing  
        with profile_block("embedding_and_indexing", enabled=enable_profiling):
            if production_mode and preloaded_embeddings and 'index' in preloaded_embeddings:
                # Use preloaded embeddings and index
                print("🔧 Using preloaded embeddings and FAISS index...")
                index = preloaded_embeddings['index']
                embedder = preloaded_embeddings['embedder']
                chunks = preloaded_embeddings['chunks']
            elif production_mode and 'embedder' in preloaded_models:
                # Use preloaded embedder to build index
                print("🔧 Using preloaded embedder, building FAISS index...")
                index, embeddings, chunks, embedder = build_faiss_index(chunks,
                                                                        embedder_model,
                                                                        EMBEDDINGS_FOLDER,          
                                                                        enable_profiling=enable_profiling,
                                                                        pre_loaded_embedder=preloaded_models['embedder'])
            else:
                # Traditional mode: build everything from scratch
                index, embeddings, chunks, embedder = build_faiss_index(chunks,
                                                                        embedder_model,
                                                                        EMBEDDINGS_FOLDER,
                                                                        enable_profiling=enable_profiling)

        # Retrieve the top-30 chunks
        with profile_block("context_retrieval", enabled=enable_profiling):
            print("Retrieving context...")
            top_chunks = retrieve_context(hyde_answer, embedder, chunks, index, k=30, enable_profiling=enable_profiling)

        # Rerank to get the top-4 chunks
        with profile_block("reranking", enabled=enable_profiling):
            print("Reranking...")
            # Combine HyDE context and original question for better reranking
            rerank_query = f"Hypothetical Answer: {hyde_answer}\n\nQuery: {rewritten_question}"
            if production_mode and 'reranker' in preloaded_models:
                # Use preloaded reranker
                reranked_chunks = reranker(reranker_model, rerank_query, top_chunks, k=4, 
                                         enable_profiling=enable_profiling, 
                                         pre_loaded_reranker=preloaded_models['reranker'])
            else:
                # Traditional reranking
                reranked_chunks = reranker(reranker_model, rerank_query, top_chunks, k=4, enable_profiling=enable_profiling)

        # Generate the prompt
        with profile_block("prompt_generation", enabled=enable_profiling):
            print("Generating prompt...")
            prompt = build_prompt_from_chunks(rewritten_question, 
                                              reranked_chunks, 
                                              enable_profiling=enable_profiling,
                                              conversation_history=conversation_history)

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
            metadata = chunk.get('metadata', {})
            
            # Get filename and clean it if it's from web scraping
            filename = metadata.get('filename', '')
            source_type = metadata.get('source_type', 'unknown')
            
            # Remove .txt extension for web scraped files
            if source_type == 'web_scraped' and filename.endswith('.txt'):
                clean_filename = filename[:-4]  # Remove .txt
            else:
                clean_filename = filename
            
            # Create source info with URL if available
            source_info = {
                'filename': clean_filename,
                'url': metadata.get('url', ''),
                'title': metadata.get('title', ''),
                'source_type': source_type
            }
            
            # Add display name based on available info
            if source_info['url']:
                source_info['display_name'] = source_info['title'] if source_info['title'] else source_info['url']
            else:
                source_info['display_name'] = source_info['filename']
            
            sources.append(source_info)
            print(f"- {source_info['display_name']}")
            if source_info['url']:
                print(f"  URL: {source_info['url']}")
        
        print_gpu_memory("RAG Pipeline End", enabled=enable_profiling)
        print_function_summary(enabled=enable_profiling)
        
        return answer, sources
    
    finally:
        # Clean up the model only if we loaded it fresh (not in production mode)
        if not production_mode:
            free_model_memory(pipeline_obj)
            if model is not None:
                del model, tokenizer, pipeline_obj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    # Parse the arguments
    if len(sys.argv) < 2:
        print("Usage: python rag.py <question> [-p] [-q <quantization>]")
        print("  -p: Enable GPU profiling and save report")
        print("  -q <quantization>: Use quantization (4bit, 8bit)")
        print("  Note: To process documents, run 'python process_data.py' first")
        sys.exit(1)
    question = sys.argv[1]
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
    
    answer, sources = rag(question, enable_profiling, quantization)
    
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

