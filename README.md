# RAG
This repository contains code for a Retrieval-Augmented Generation (RAG) model for generating answers to questions based on a given context.

## How to use
1. Install the required packages by running: `pip install -r requirements.txt`
2. To generate an answer, run: `python rag.py "What is the capital of France?"`. Use the -s flag to recompute all chunks and embeddings, and save them: `python rag.py "Your question here" -s` 


## How it works
The model combines retrieval and generation steps. It first retrieves a set of relevant documents from a corpus based on the question. Then, it generates an answer using the retrieved content.

## Pipeline
The diagram below illustrates the current pipeline implementation:

![RAG pipeline](doc/pipeline.png)

## Models used
- HyDE : microsoft/Phi-4-mini-instruct
- Embedder: intfloat/multilingual-e5-large-instruct
- Retrieval : FAISS
- Reranker : BAAI/bge-reranker-v2-m3
- Generator: microsoft/Phi-4-mini-instruct
