#!/usr/bin/env python3
"""
Production Deployment Script for RAG System

This script helps you deploy the optimized RAG system with proper configuration.
"""

import argparse
import subprocess
import sys
import os
import torch
from production_config import ProductionConfig, STRATEGIES

def check_gpu_memory():
    """Check available GPU memory."""
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected!")
        return 0.0
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_memory_gb:.1f}GB)")
    
    current_usage = torch.cuda.memory_allocated() / (1024**3)
    available = gpu_memory_gb - current_usage
    print(f"Available: {available:.1f}GB")
    
    return available

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = {
        "torch": "torch",
        "transformers": "transformers", 
        "sentence-transformers": "sentence_transformers",
        "faiss-cpu": "faiss",
        "flask": "flask",
        "flask-cors": "flask_cors",
        "FlagEmbedding": "FlagEmbedding",
        "huggingface_hub": "huggingface_hub",
        "python-dotenv": "dotenv"
    }
    
    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def validate_paths():
    """Validate that required files exist."""
    config = ProductionConfig()
    paths_to_check = [
        (config.CHUNK_PATH, "Chunks file"),
        (config.EMBEDDINGS_FOLDER, "Embeddings folder"),
    ]
    
    missing = []
    for path, description in paths_to_check:
        if not os.path.exists(path):
            missing.append((path, description))
    
    if missing:
        print("❌ Missing files/folders:")
        for path, desc in missing:
            print(f"  {desc}: {path}")
        return False
    
    print("✅ All paths valid!")
    return True

def recommend_strategy(available_memory_gb):
    """Recommend deployment strategy based on available memory."""
    estimates = ProductionConfig.MODEL_MEMORY_ESTIMATES
    llm_mem = estimates["llm_4bit"]
    embedder_mem = estimates["embedder"] 
    reranker_mem = estimates["reranker"]
    
    total_all = llm_mem + embedder_mem + reranker_mem
    total_llm_embedder = llm_mem + embedder_mem
    total_embedder_reranker = embedder_mem + reranker_mem
    
    if available_memory_gb >= total_all + 2:
        strategy = "preload_all"
        print(f"RECOMMENDED: Preload All Models (~{total_all:.1f}GB)")
    elif available_memory_gb >= total_llm_embedder + 1:
        strategy = "preload_embedder_llm"
        print(f"RECOMMENDED: Preload Embedder + LLM (~{total_llm_embedder:.1f}GB)")
    elif available_memory_gb >= total_embedder_reranker + 1:
        strategy = "preload_embedder_reranker"
        print(f"RECOMMENDED: Preload Embedder + Reranker (~{total_embedder_reranker:.1f}GB)")
    elif available_memory_gb >= embedder_mem + 0.5:
        strategy = "preload_embedder"
        print(f"RECOMMENDED: Preload Embedder Only (~{embedder_mem:.1f}GB)")
    else:
        strategy = "no_preload"
        print("RECOMMENDED: No Preload (minimal memory)")
    
    return strategy

def create_production_config(strategy, available_memory_gb):
    """Create a customized production configuration."""
    config_content = f'''"""Auto-generated Production Configuration"""

from production_config import ProductionConfig as BaseConfig

class Config(BaseConfig):
    MAX_GPU_MEMORY_GB = {available_memory_gb * 0.8:.1f}
    PRELOAD_MODELS = {STRATEGIES[strategy]}
    DEFAULT_QUANTIZATION = "4bit"
'''
    
    with open("auto_config.py", "w") as f:
        f.write(config_content)
    
    print(f"📝 Created auto_config.py with {strategy} strategy")

def run_production_server(dry_run=False):
    """Run the production server."""
    if dry_run:
        print("Would run: python production_api.py")
        return
    
    print("🚀 Starting production RAG server...")
    print("Server: http://localhost:5000")
    print("Health: http://localhost:5000/api/health")
    print("Status: http://localhost:5000/api/models/status")
    print("\nPress Ctrl+C to stop")
    
    try:
        subprocess.run([sys.executable, "production_api.py"])
    except KeyboardInterrupt:
        print("\nServer stopped")

def main():
    parser = argparse.ArgumentParser(description="Deploy Production RAG System")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Check configuration without starting server")
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()),
                       help="Force a specific deployment strategy")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip dependency and path validation")
    
    args = parser.parse_args()
    
    print("🚀 Production RAG Deployment")
    print("=" * 30)
    
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
        
        if not validate_paths():
            sys.exit(1)
    
    available_memory = check_gpu_memory()
    if available_memory < 4:
        print("⚠️ Warning: Less than 4GB GPU memory available!")
    
    if args.strategy:
        strategy = args.strategy
        print(f"Using forced strategy: {strategy}")
    else:
        strategy = recommend_strategy(available_memory)
    
    print(f"\nDeployment Plan:")
    print(f"Strategy: {strategy}")
    print(f"Preload: {STRATEGIES[strategy]}")
    print(f"Memory target: {available_memory * 0.8:.1f}GB")
    
    if not args.dry_run:
        create_production_config(strategy, available_memory)
        
        response = input("\nProceed with deployment? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Deployment cancelled")
            sys.exit(0)
    
    run_production_server(args.dry_run)

if __name__ == "__main__":
    main() 