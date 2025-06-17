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
from production_config import ProductionConfig, STRATEGIES, print_configuration_summary

def check_gpu_memory():
    """Check available GPU memory."""
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected!")
        return 0.0
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Total GPU Memory: {gpu_memory_gb:.1f}GB")
    
    # Check current usage
    current_usage = torch.cuda.memory_allocated() / (1024**3)
    available = gpu_memory_gb - current_usage
    print(f"📊 Currently Used: {current_usage:.1f}GB")
    print(f"✅ Available: {available:.1f}GB")
    
    return available

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    # Package name -> Import name mapping
    required_packages = {
        "torch": "torch",
        "transformers": "transformers", 
        "sentence-transformers": "sentence_transformers",
        "faiss-cpu": "faiss",  # faiss-cpu package imports as 'faiss'
        "flask": "flask",
        "flask-cors": "flask_cors",
        "FlagEmbedding": "FlagEmbedding",
        "huggingface_hub": "huggingface_hub",
        "python-dotenv": "dotenv"  # python-dotenv package imports as 'dotenv'
    }
    
    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name}")
            missing.append(package_name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def validate_paths():
    """Validate that required files exist."""
    print("📁 Checking file paths...")
    
    config = ProductionConfig()
    paths_to_check = [
        (config.CHUNK_PATH, "Chunks file"),
        (config.EMBEDDINGS_FOLDER, "Embeddings folder"),
    ]
    
    missing = []
    for path, description in paths_to_check:
        if os.path.exists(path):
            print(f"  ✅ {description}: {path}")
        else:
            print(f"  ❌ {description}: {path}")
            missing.append((path, description))
    
    if missing:
        print("\n❌ Missing files/folders:")
        for path, desc in missing:
            print(f"  {desc}: {path}")
        return False
    
    print("✅ All paths valid!")
    return True

def recommend_strategy(available_memory_gb):
    """Recommend deployment strategy based on available memory."""
    from production_config import ProductionConfig
    
    # Get memory estimates from config
    estimates = ProductionConfig.MODEL_MEMORY_ESTIMATES
    llm_mem = estimates["llm_4bit"]
    embedder_mem = estimates["embedder"] 
    reranker_mem = estimates["reranker"]
    
    print(f"\n🧠 Memory-based Recommendations (Available: {available_memory_gb:.1f}GB)")
    print("=" * 60)
    
    # Calculate total for all models
    total_all = llm_mem + embedder_mem + reranker_mem
    total_llm_embedder = llm_mem + embedder_mem
    total_embedder_reranker = embedder_mem + reranker_mem
    
    if available_memory_gb >= total_all + 2:  # +2GB buffer
        strategy = "preload_all"
        print("💚 RECOMMENDED: Preload All Models")
        print("   You have enough memory to keep all models loaded.")
        print("   This will give you the fastest response times.")
        print(f"   Expected usage: ~{total_all:.1f}GB ({embedder_mem:.1f}GB embedder + {llm_mem:.1f}GB LLM + {reranker_mem:.1f}GB reranker)")
    elif available_memory_gb >= total_llm_embedder + 1:  # +1GB buffer
        strategy = "preload_embedder_and_llm"
        print("💛 RECOMMENDED: Preload Embedder + LLM")
        print("   Load reranker on-demand to save memory.")
        print("   Good balance of speed and memory usage.")
        print(f"   Expected usage: ~{total_llm_embedder:.1f}GB ({embedder_mem:.1f}GB embedder + {llm_mem:.1f}GB LLM)")
    elif available_memory_gb >= total_embedder_reranker + 1:  # +1GB buffer
        strategy = "preload_embedder_and_reranker"
        print("🟡 RECOMMENDED: Preload Embedder + Reranker")
        print("   Load LLM on-demand. Good for moderate memory.")
        print("   Fast retrieval, slower answer generation.")
        print(f"   Expected usage: ~{total_embedder_reranker:.1f}GB ({embedder_mem:.1f}GB embedder + {reranker_mem:.1f}GB reranker)")
    elif available_memory_gb >= embedder_mem + 0.5:  # +0.5GB buffer
        strategy = "preload_embedder"
        print("🟠 RECOMMENDED: Preload Embedder Only")
        print("   Load LLM and reranker on demand.")
        print("   Slower but manageable memory usage.")
        print(f"   Expected usage: ~{embedder_mem:.1f}GB (embedder only)")
    elif available_memory_gb >= reranker_mem + 0.5:  # +0.5GB buffer
        strategy = "preload_reranker"
        print("🟡 RECOMMENDED: Preload Reranker Only")
        print("   Load embedder and LLM on demand.")
        print("   Very low memory, fast reranking step.")
        print(f"   Expected usage: ~{reranker_mem:.1f}GB (reranker only)")
    elif available_memory_gb >= 1:
        strategy = "no_preload"
        print("🔴 RECOMMENDED: No Preload")
        print("   Load everything on-demand for each request.")
        print("   Slowest but works on any GPU.")
        print("   Expected usage: ~0.5GB (minimal overhead)")
    else:
        strategy = "load_on_demand"
        print("⚫ RECOMMENDED: Load On-Demand")
        print("   Load models only when needed to fit in memory.")
        print("   Extremely limited memory mode.")
    
    return strategy

def create_production_config(strategy, available_memory_gb):
    """Create a customized production configuration."""
    config_content = f'''"""
Auto-generated Production Configuration
Strategy: {strategy}
Available GPU Memory: {available_memory_gb:.1f}GB
"""

from production_config import ProductionConfig as BaseConfig

class Config(BaseConfig):
    # Adjusted for your GPU
    MAX_GPU_MEMORY_GB = {available_memory_gb * 0.8:.1f}  # 80% of available
    
    # Strategy-specific preloading
    PRELOAD_MODELS = {STRATEGIES[strategy]["preload"]}
    
    # Optimized settings
    DEFAULT_QUANTIZATION = "4bit"  # Use 4bit for memory efficiency
    
    # You can override other settings here as needed
'''
    
    with open("auto_config.py", "w") as f:
        f.write(config_content)
    
    print(f"📝 Created auto_config.py with {strategy} strategy")

def run_production_server(dry_run=False):
    """Run the production server."""
    if dry_run:
        print("🏃 Would run: python production_api.py")
        return
    
    print("🚀 Starting production RAG server...")
    print("   Server will be available at http://localhost:5000")
    print("   Health check: http://localhost:5000/api/health")
    print("   Model status: http://localhost:5000/api/models/status")
    print("\n   Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, "production_api.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")

def main():
    parser = argparse.ArgumentParser(description="Deploy Production RAG System")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Check configuration without starting server")
    parser.add_argument("--strategy", choices=["preload_all", "preload_embedder_and_llm", "preload_embedder_and_reranker", "preload_embedder", "preload_reranker", "no_preload", "load_on_demand"],
                       help="Force a specific deployment strategy")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip dependency and path validation")
    
    args = parser.parse_args()
    
    print("🚀 Production RAG Deployment Tool")
    print("=" * 40)
    
    # Show current configuration
    print_configuration_summary()
    print()
    
    if not args.skip_checks:
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        print()
        
        # Validate paths
        if not validate_paths():
            sys.exit(1)
        print()
    
    # Check GPU memory
    available_memory = check_gpu_memory()
    if available_memory < 4:
        print("⚠️  Warning: Less than 4GB GPU memory available!")
        print("   Consider using CPU-only mode or a smaller model.")
    print()
    
    # Recommend strategy
    if args.strategy:
        strategy = args.strategy
        print(f"🎯 Using forced strategy: {strategy}")
    else:
        strategy = recommend_strategy(available_memory)
    
    print(f"\n📋 Deployment Plan:")
    print(f"   Strategy: {strategy}")
    print(f"   Preload: {STRATEGIES[strategy]['preload']}")
    print(f"   Memory target: {available_memory * 0.8:.1f}GB")
    
    if not args.dry_run:
        create_production_config(strategy, available_memory)
        
        # Ask for confirmation
        response = input("\n❓ Proceed with deployment? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("👋 Deployment cancelled")
            sys.exit(0)
    
    # Run the server
    run_production_server(args.dry_run)

if __name__ == "__main__":
    main() 