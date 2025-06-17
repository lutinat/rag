"""
Configuration file for Production RAG System

Adjust these settings based on your hardware and requirements.
"""

import os

class ProductionConfig:
    """Configuration class for production RAG deployment."""
    
    # GPU Memory Management
    MAX_GPU_MEMORY_GB = 16.0  # Adjust based on your GPU memory (leave 2-4GB for overhead)
    
    # Model Memory Estimates (in GB) - based on real measurements
    MODEL_MEMORY_ESTIMATES = {
        "llm_4bit": 3.3,      # Phi-4-mini-instruct with 4bit quantization
        "llm_8bit": 4.8,      # Phi-4-mini-instruct with 8bit quantization  
        "llm_full": 12.0,     # Phi-4-mini-instruct full precision (not recommended)
        "embedder": 4.4,      # E5-large-instruct
        "reranker": 2.3,      # BGE-reranker-v2-m3
    }
    
    # Models to Pre-load at Startup (conservative start, you can enable more)
    PRELOAD_MODELS = [
        "embedder",  # Always preload embedder for FAISS index
        "llm",     # Uncomment to preload LLM (uses more memory but faster first response)
        "reranker" # Uncomment to preload reranker
        # Or use: [] for no preloading (maximum memory efficiency)
    ]
    
    # File Paths (adjust to your setup)
    CHUNK_PATH = "/home/elduayen/rag/processed_data/all_chunks.jsonl"
    EMBEDDINGS_FOLDER = "/home/elduayen/rag/embeddings"
    
    # Model Names
    MODELS = {
        "llm": "microsoft/Phi-4-mini-instruct",
        "embedder": "intfloat/multilingual-e5-large-instruct", 
        "reranker": "BAAI/bge-reranker-v2-m3"
    }
    
    # Processing Settings
    DEFAULT_QUANTIZATION = "4bit"  # "4bit", "8bit", or None
    RETRIEVAL_K = 30              # Number of chunks to retrieve initially
    RERANK_K = 4                  # Number of chunks after reranking
    REQUEST_TIMEOUT = 300         # Request timeout in seconds
    
    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    MAX_WORKERS = 1               # Number of worker threads (keep at 1 for GPU)
    
    # Memory Management Strategy
    UNLOAD_STRATEGY = "lru"       # "lru" (least recently used) or "fifo"
    MEMORY_CHECK_INTERVAL = 60    # Seconds between memory checks
    
    # Logging
    LOG_LEVEL = "INFO"            # "DEBUG", "INFO", "WARNING", "ERROR"
    
    @classmethod
    def get_max_memory_for_gpu(cls, gpu_memory_gb: float) -> float:
        """Calculate max memory to use based on GPU memory."""
        # Leave 20-25% for overhead and other processes
        return gpu_memory_gb * 0.75
    
    @classmethod
    def can_fit_all_models(cls) -> bool:
        """Check if all models can fit in memory simultaneously."""
        total_needed = (
            cls.MODEL_MEMORY_ESTIMATES["llm_4bit"] + 
            cls.MODEL_MEMORY_ESTIMATES["embedder"] + 
            cls.MODEL_MEMORY_ESTIMATES["reranker"]
        )
        return total_needed <= cls.MAX_GPU_MEMORY_GB
    
    @classmethod
    def get_recommended_strategy(cls) -> str:
        """Get recommended loading strategy based on available memory."""
        if cls.can_fit_all_models():
            return "preload_all"
        elif cls.MAX_GPU_MEMORY_GB >= 12:
            return "preload_embedder_and_llm"
        else:
            return "load_on_demand"

# Memory optimization strategies
STRATEGIES = {
    "preload_all": {
        "description": "Pre-load all models at startup (requires 16GB+ GPU memory)",
        "preload": ["embedder", "llm", "reranker"],
        "pros": ["Fastest response times", "No loading delays"],
        "cons": ["High memory usage", "Requires powerful GPU"]
    },
    
    "preload_embedder_and_llm": {
        "description": "Pre-load embedder and LLM, load reranker on demand",
        "preload": ["embedder", "llm"],
        "pros": ["Fast response for most operations", "Moderate memory usage"],
        "cons": ["Small delay for reranking"]
    },
    
    "load_on_demand": {
        "description": "Load models only when needed",
        "preload": ["embedder"],  # Always need embedder for FAISS
        "pros": ["Lowest memory usage", "Works on smaller GPUs"],
        "cons": ["Slower response times", "Loading delays"]
    },

    "preload_embedder_and_reranker": {
        "description": "Pre-load embedder and reranker, load LLM on demand",
        "preload": ["embedder", "reranker"],
        "pros": ["Fast response for most operations", "Moderate memory usage"],
        "cons": ["Small delay for LLM"]
    },

    "preload_embedder": {
        "description": "Pre-load embedder only",
        "preload": ["embedder"],
        "pros": ["Low memory usage", "Works on smaller GPUs"],
        "cons": ["Slower response times", "Loading delays"]
    },

    "preload_reranker": {
        "description": "Pre-load reranker only",
        "preload": ["reranker"],
        "pros": ["Minimal memory usage", "Fast reranking step", "Good for limited memory"],
        "cons": ["Slow embedding and LLM loading", "High initial delays"]
    },

    "no_preload": {
        "description": "Load all models on-demand (no preloading)",
        "preload": [],
        "pros": ["Minimal memory usage", "Works on any GPU", "Maximum flexibility"],
        "cons": ["Slowest response times", "High loading delays", "No model reuse"]
    }
}

def print_configuration_summary():
    """Print a summary of the current configuration."""
    config = ProductionConfig()
    
    print("🚀 Production RAG Configuration Summary")
    print("=" * 50)
    print(f"Max GPU Memory: {config.MAX_GPU_MEMORY_GB}GB")
    print(f"Default Quantization: {config.DEFAULT_QUANTIZATION}")
    print(f"Can fit all models: {'✅ Yes' if config.can_fit_all_models() else '❌ No'}")
    print(f"Recommended strategy: {config.get_recommended_strategy()}")
    print()
    
    strategy = STRATEGIES[config.get_recommended_strategy()]
    print(f"📋 Recommended Strategy: {strategy['description']}")
    print(f"Models to preload: {strategy['preload']}")
    print(f"Pros: {', '.join(strategy['pros'])}")
    print(f"Cons: {', '.join(strategy['cons'])}")
    print()
    
    print("🔧 Model Memory Estimates:")
    for model, memory in config.MODEL_MEMORY_ESTIMATES.items():
        print(f"  {model}: {memory}GB")

if __name__ == "__main__":
    print_configuration_summary() 