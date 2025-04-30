# RAG
This repository contains code for a Retrieval-Augmented Generation (RAG) model for generating answers to questions based on a given context.

## How to use
1. First, you need to install the required packages. You can do this by running `pip install -r requirements.txt` in the root directory of the repository.
2. You can then use the model to generate answers to questions. For example, you can run `python rag.py "What is the capital of France?"` to generate an answer to the question. Use `python rag.py "question" -s`  to recompute all the chunks/embeddings and save them.

## How it works
The model uses a combination of retrieval and generation to generate answers to questions. First, it uses a retrieval model to retrieve a set of relevant documents from a corpus based on the question. Then, it uses a generation model to generate an answer based on the retrieved documents.

## Models used
hyDE : microsoft/Phi-4-mini-instruct
Embedder: intfloat/multilingual-e5-large-instruct
Retrieval : FAISS
reranker : BAAI/bge-reranker-v2-m3
Generator: microsoft/Phi-4-mini-instruct
