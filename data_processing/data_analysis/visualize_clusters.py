import hdbscan
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from umap import UMAP
import numpy as np

import matplotlib.pyplot as plt
import json


def cluster_text(embeddings, questions, method="hdbscan"):
    """
    Cluster the text based on their similarity.
    
    Args:
        questions (list): List of generated questions.
    
    Returns:
        list: List of clustered questions.
    """
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(embeddings)
    
    clusters = defaultdict(list)
    for label, question in zip(labels, questions):
        label = str(label)  # Convert label to string for JSON serialization
        clusters[label].append(question)
    
    return clusters


def get_umap_embeddings(embeddings, labels=None, plot=False, save_fig=False):
    """
    Generate UMAP embeddings for the given embeddings.
    
    Args:
        embeddings (list): List of question embeddings.
    
    Returns:
        np.ndarray: UMAP embeddings.
    """
    
    reducer = UMAP()
    umap_embeddings = reducer.fit_transform(embeddings)

    if plot:
        plt.figure(figsize=(10, 10))
        plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='Spectral', s=5)
        plt.title("UMAP projection of the clusters")

        # save the figure
        if save_fig:
            plt.savefig("/home/lucasd/code/rag/data_processing/data_analysis/umap_projection.png")
        plt.show()
    
    return umap_embeddings


if __name__ == "__main__":

    input_jsonl = "/home/lucasd/code/rag/processed_data/all_chunks.jsonl"

    # Load the generated questions
    with open(input_jsonl, "r") as f:
        questions = [json.loads(line)["text"] for line in f]

    # get embeddings for the questions
    print("Generating embeddings...")
    embeddings = np.load("/home/lucasd/code/rag/embeddings/embeddings.npy")

    #get umap embeddings
    umap_embeddings = get_umap_embeddings(embeddings, plot=True, save_fig=True)
    
    # Cluster the questions
    print("Clustering questions...")
    clustered_questions = cluster_text(umap_embeddings, questions)
    # Save the clustered questions to a JSON file
    with open("/home/lucasd/code/rag/data_processing/data_analysis/clustered_questions.json", "w") as f:
        json.dump(clustered_questions, f, indent=4)
    print("Clustered questions saved to clustered_questions.json")