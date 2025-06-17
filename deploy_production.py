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
    paths_to_check = [
        ("/home/elduayen/rag/processed_data/all_chunks.jsonl", "Chunks file"),
        ("/home/elduayen/rag/embeddings", "Embeddings folder"),
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

def run_production_server():
    """Run the production server."""
    print("🚀 Starting production RAG server...")
    print("Server: http://localhost:5000")
    print("Health: http://localhost:5000/api/health")
    print("Status: http://localhost:5000/api/models/status")
    print("\nPress Ctrl+C to stop")
    
    try:
        subprocess.run([sys.executable, "api.py"])
    except KeyboardInterrupt:
        print("\nServer stopped")

def main():
    parser = argparse.ArgumentParser(description="Deploy Production RAG System")
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
    
    print("\nStarting RAG API server...")
    run_production_server()

if __name__ == "__main__":
    main() 