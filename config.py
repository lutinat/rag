"""Production RAG System Configuration"""

import os

class ProductionConfig:
    """Core configuration for production RAG deployment."""
    
    # GPU Memory Management
    MAX_GPU_MEMORY_GB = 16.0
    
    # Model Memory Estimates (GB)
    MODEL_MEMORY_ESTIMATES = {
        "llm_4bit": 3.3,
        "llm_8bit": 4.8,
        "embedder": 4.4,
        "reranker": 2.3,
    }
    
    # Models to Pre-load at Startup
    PRELOAD_MODELS = [
        "embedder",
        "llm",
        "reranker"
    ]
    
    # File Paths
    PROJECT_ROOT = "/mnt/DATA2/chatlantis/rag"
    EMBEDDINGS_FOLDER = f"{PROJECT_ROOT}/data/embeddings"
    RAW_DATA_FOLDER = f"{PROJECT_ROOT}/data/raw"
    PROCESSED_DATA_FOLDER = f"{PROJECT_ROOT}/data/processed"
    CHUNK_PATH = f"{PROCESSED_DATA_FOLDER}/all_chunks.jsonl"
    
    # Model Names
    MODELS = {
        "llm": "microsoft/Phi-4-mini-instruct",
        "embedder": "intfloat/multilingual-e5-large-instruct", 
        "reranker": "BAAI/bge-reranker-v2-m3"
    }
    
    # Processing Settings
    MAX_HISTORY_TURNS = 10
    DEFAULT_QUANTIZATION = "4bit"
    RETRIEVAL_K = 30
    RERANK_K = 4
    REQUEST_TIMEOUT = 300
    
    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    MAX_WORKERS = 1
    
    @classmethod
    def can_fit_all_models(cls) -> bool:
        """Check if all models can fit in memory."""
        total_needed = (
            cls.MODEL_MEMORY_ESTIMATES["llm_4bit"] + 
            cls.MODEL_MEMORY_ESTIMATES["embedder"] + 
            cls.MODEL_MEMORY_ESTIMATES["reranker"]
        )
        return total_needed <= cls.MAX_GPU_MEMORY_GB

# Available deployment strategies
STRATEGIES = {
    "preload_all": ["embedder", "llm", "reranker"],
    "preload_embedder_llm": ["embedder", "llm"],
    "preload_embedder_reranker": ["embedder", "reranker"],
    "preload_embedder": ["embedder"],
    "no_preload": []
}

 