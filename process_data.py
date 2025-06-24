#!/usr/bin/env python3
"""
Simple Data Processing Script for RAG System

Handles document chunking and embedding generation.
"""

import os
import sys
import argparse
from typing import Optional
from dotenv import load_dotenv
from huggingface_hub import login

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ProductionConfig
from utils import save_chunks_jsonl, load_chunks_jsonl
from data_processing.data_extraction.chunker import get_all_chunks
from retriever.retriever import build_faiss_index

# Load environment variables and login
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

class DataProcessor:
    """Simple data processor for the RAG system."""
    
    def __init__(self):
        self.config = ProductionConfig
        
    def process_documents(self, data_folder=None, force_reprocess=False):
        """Process documents and extract chunks."""
        data_folder = data_folder or self.config.RAW_DATA_FOLDER
        processed_data_folder = self.config.PROCESSED_DATA_FOLDER
        
        os.makedirs(processed_data_folder, exist_ok=True)
        
        print(f"📁 Processing documents from: {data_folder}")
        
        # Check if chunks already exist
        if os.path.exists(self.config.CHUNK_PATH) and not force_reprocess:
            existing_chunks = load_chunks_jsonl(self.config.CHUNK_PATH)
            print(f"📊 Found {len(existing_chunks)} existing chunks (use --force-documents to reprocess)")
            return len(existing_chunks)
        
        # Process documents
        all_chunks = get_all_chunks(data_folder, processed_data_folder)
        
        # Save chunks
        save_chunks_jsonl(all_chunks, self.config.CHUNK_PATH)
        print(f"💾 Saved {len(all_chunks)} chunks")
            
        return len(all_chunks)
    
    def generate_embeddings(self, force_regenerate=False):
        """Generate embeddings and build FAISS index."""
        if not os.path.exists(self.config.CHUNK_PATH):
            raise FileNotFoundError(f"No chunks found. Run document processing first.")
            
        chunks = load_chunks_jsonl(self.config.CHUNK_PATH)
        print(f"📖 Loaded {len(chunks)} chunks")
        
        print(f"🔄 Generating embeddings...")
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(self.config.EMBEDDINGS_FOLDER, exist_ok=True)
        
        # Build FAISS index
        index, embeddings, chunks, embedder = build_faiss_index(
            chunks,
            self.config.MODELS["embedder"],
            self.config.EMBEDDINGS_FOLDER,
            save_embeddings=force_regenerate,
            enable_profiling=False
        )

        # Save the updated chunks (removed duplicates using embeddings)
        save_chunks_jsonl(chunks, self.config.CHUNK_PATH)

        print(f"✅ Generated embeddings for {len(chunks)} chunks")
        return index, embeddings, chunks, embedder
    
    def full_pipeline(self, data_folder=None, force_reprocess=False, force_regenerate_embeddings=False):
        """Run complete processing pipeline."""
        print("🚀 Starting data processing...")
        
        # Step 1: Process documents
        print("\n📝 Processing Documents")
        num_chunks = self.process_documents(data_folder, force_reprocess)
        
        # Step 2: Generate embeddings
        print("\n🔢 Generating Embeddings")
        index, embeddings, chunks, embedder = self.generate_embeddings(force_regenerate_embeddings)
        
        print(f"\n✅ Pipeline completed!")
        print(f"📊 Processed {num_chunks} chunks, generated {len(chunks)} embeddings, found {num_chunks - len(chunks)} duplicates")
        
        return index, embeddings, chunks, embedder


def main():
    parser = argparse.ArgumentParser(description="Process documents for RAG system")
    
    parser.add_argument("--data-folder", help="Custom data folder path")
    parser.add_argument("--force-documents", action="store_true", help="Force reprocess documents")
    parser.add_argument("--force-embeddings", action="store_true", help="Force regenerate embeddings")
    parser.add_argument("--documents-only", action="store_true", help="Only process documents")
    parser.add_argument("--embeddings-only", action="store_true", help="Only generate embeddings")
    
    args = parser.parse_args()
    processor = DataProcessor()
    
    try:
        if args.documents_only:
            print("📝 Processing documents only...")
            num_chunks = processor.process_documents(args.data_folder, args.force_documents)
            print(f"✅ Processed {num_chunks} chunks")
            
        elif args.embeddings_only:
            print("🔢 Generating embeddings only...")
            processor.generate_embeddings(args.force_embeddings)
            print("✅ Embeddings generated")
            
        else:
            processor.full_pipeline(
                args.data_folder, 
                args.force_documents, 
                args.force_embeddings
            )
                
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 