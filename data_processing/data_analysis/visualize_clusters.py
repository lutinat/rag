import hdbscan
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from umap import UMAP
import numpy as np
import random
import os

import matplotlib.pyplot as plt
import seaborn as sns
import json

from tqdm import tqdm

from keybert import KeyBERT
from collections import defaultdict

from transformers import AutoModel

from sklearn.metrics.pairwise import cosine_similarity
import random
import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import ProductionConfig


structured_keywords = {
    "optic": [
        "optic", "optics", "lens", "mirror", "filter", "aperture", "focus", "spectrometer",
        "interferometer", "ccd", "cmos", "detector", "focal", "snr", "spectrum", "band",
        "light", "wavelength", "refraction", "reflection", "diffraction", "polarization"
    ],
    "satellite": [
        "satellite", "platform", "payload", "telemetry", "geolocation", "orbit", "bus",
        "dynamics", "deployment", "altitude", "tracking", "antenna", "uplink", "downlink",
        "communication", "gps", "navigation", "constellation", "launch", "spacecraft", 
        "rocket", "propulsion", "station", "ground", "mission control"
    ],
    "missions": [
        "mission", "nasa", "esa", "iss", "stennis", "launch", "revisit", "station", "rocket",
        "flight", "crew", "agency", "observation", "exploration", "research", "payload",
        "space", "discovery", "satellite", "launchpad", "space agency", "autonomy", "in-orbit"
    ],
    "polarimetry": [
        "polarimetry", "stokes", "brdf", "grasp", "scatter", "backscatter", "angle", "sar",
        "decomposition", "wave", "dipole", "radar", "polarization", "ellipticity", "interferometry",
        "radiometry", "light scattering", "atmospheric", "reflection", "transmission"
    ],
    "image_processing": [
        "correction", "registration", "mosaicking", "tiling", "chunking", "compression",
        "deblurring", "stitching", "contrast", "noise", "sharpening", "feature", "masking",
        "detection", "resampling", "smoothing", "enhancement", "segmentation", "classification",
        "image fusion", "normalization", "analysis", "image quality"
    ],
    "climate": [
        "climate", "albedo", "emissivity", "ozone", "tropopause", "carbon", "warming",
        "humidity", "temperature", "trend", "cloud", "energy", "carbon dioxide", "greenhouse",
        "radiation", "precipitation", "weather", "ecosystem", "sustainability", "greenhouse effect",
        "fossil fuels", "sea level rise", "global warming", "radiation balance"
    ],
    "aerosols": [
        "aerosol", "depth", "extinction", "dust", "smoke", "haze", "pollution", "layer", "column",
        "atmosphere", "particulate matter", "dust storm", "smog", "air quality", "climate change",
        "emissions", "scattering", "absorption", "particulates", "transmission", "optical depth"
    ],
    "calibration": [
        "calibration", "radiometry", "radiance", "reflectance", "lamp", "offset", "reference",
        "standard", "accuracy", "error", "response", "sensor", "geometric", "sensor calibration",
        "spectral calibration", "radiometric", "validation", "traceability", "sensor drift"
    ],
    "machine_learning": [
        "machine", "learning", "ml", "ai", "artificial", "intelligence", "algorithm", "model",
        "training", "inference", "dataset", "feature", "training", "validation", "testing",
        "accuracy", "loss", "regression", "classification", "clustering", "supervised",
        "unsupervised", "reinforcement", "transfer", "deep", "neural", "network", "architecture",
        "layer", "activation", "function", "optimizer", "gradient", "descent", "backpropagation",
        "dropout", "batch", "normalization", "regularization", "hyperparameter", "tuning",
        "overfitting", "underfitting", "cross-validation", "k-fold", "confusion", "matrix",
        "precision", "recall", "f1-score", "roc", "auc", "curve", "neural networks", "model selection"
    ],
    "sensors_instruments": [
        "sensor", "instrumentation", "spectrometer", "radiometer", "lidar", "camera", "radar", 
        "multispectral", "hyperspectral", "thermal", "infrared", "ultraviolet", "photodetector", 
        "gps receiver", "thermal imager", "laser", "multiband", "scanning", "modulation", "imaging system",
        "optical sensor", "photodiode", "microbolometer", "interferometer", "actuators"
    ],
    "geospatial_mapping": [
        "geospatial", "gis", "coordinate system", "latitude", "longitude", "altitude", "topography", 
        "georeferencing", "cartography", "map projection", "spatial analysis", 
        "digital elevation model", "3D modeling", "spatial data", "land cover", "satellite imagery", 
        "geospatial data fusion", "spatial distribution", "mapping algorithms"
    ],
    "atmospheric_studies": [
        "atmosphere", "troposphere", "stratosphere", "ozone layer", "particulates", "air quality", 
        "greenhouse gases", "weather", "meteorology", "wind", "precipitation", "cloud formation", 
        "humidity", "barometric pressure", "weather forecasting", "tropospheric ozone", "aerosol layer", 
        "temperature inversion", "storm systems", "solar radiation", "rainfall", "fog", "dew point"
    ],
    "data_acquisition_processing": [
        "data acquisition", "sampling", "raw data", "data storage", "data transmission", "data pipeline", 
        "preprocessing", "filtering", "compression", "data normalization", "data mining", "data analysis", 
        "data fusion", "big data", "parallel processing", "cloud computing", "real-time processing", 
        "signal processing", "data storage systems", "sensor data integration", "data quality assurance"
    ],
    "remote_sensing": [
        "remote sensing", "satellite imagery", "ground truth", "radiometric", "geospatial data", 
        "spectral bands", "sensing technology", "optical remote sensing", "radar remote sensing", 
        "lidar remote sensing", "active sensing", "passive sensing", "radar backscatter", "remote sensing data",
        "spatial resolution", "spectral resolution", "temporal resolution", "earth observation systems", 
        "multispectral imaging", "hyperspectral sensing"
    ],
    "earth_surface_land_use": [
        "land use", "land cover", "urbanization", "deforestation", "agriculture", "ecosystem", 
        "soil moisture", "vegetation", "forest cover", "wetlands", "coastal zones", "desertification", 
        "land degradation", "crop monitoring", "land transformation", "agricultural monitoring", 
        "forestry management", "land restoration", "land classification", "geodetic monitoring"
    ],
    "oceanography": [
        "ocean", "sea", "marine", "ocean currents", "salinity", "temperature", "sea level rise", "marine life", 
        "ocean surface", "bathymetry", "ocean circulation", "coastal zones", "tsunami monitoring", 
        "marine ecosystems", "ocean temperature", "sea surface height", "marine pollution", 
        "marine biodiversity", "ocean acidification", "fisheries", "marine conservation"
    ],
    "environmental_monitoring": [
        "environment", "biodiversity", "sustainability", "conservation", "ecosystem health", "pollution", 
        "deforestation", "land degradation", "water quality", "air pollution", "noise pollution", 
        "environmental impact", "sustainable development", "habitat loss", "wildlife monitoring", 
        "carbon footprint", "conservation efforts", "natural resources", "ecological footprint", 
        "ecosystem services", "environmental protection"
    ],
    "satellite_data_services": [
        "data services", "satellite data", "data distribution", "cloud computing", "geospatial platforms", 
        "satellite imagery access", "data subscription", "open data", "API services", "data storage", 
        "data visualization", "remote sensing services", "earth observation data", "big data services", 
        "data analytics platforms", "earth observation portals", "data as a service", "user interfaces"
    ],
    "space_policy_regulations": [
        "space policy", "space law", "satellite regulations", "space treaties", "international cooperation", 
        "satellite licensing", "space debris", "space traffic management", "space commercialization", 
        "space governance", "space safety", "regulatory compliance", "earth observation regulations", 
        "space treaties", "privacy in satellite data", "space exploration policy", "launch licensing", 
        "satellite spectrum allocation"
    ],
    "technology_innovation": [
        "innovation", "technologies", "automation", "artificial intelligence", "machine learning", 
        "blockchain", "quantum computing", "5G in space", "autonomous systems", "data analytics", 
        "sensor technologies", "new materials", "nanotechnology", "robotics", "IoT", "smart sensors", 
        "advanced algorithms", "space exploration technologies", "microgravity", "space propulsion", "AI for space"
    ],
    "methane detection": [
        "methane", "CH4", "greenhouse gas", "emissions", "detection", "monitoring", "satellite", "plume",
        "source", "atmosphere", "remote sensing", "spectroscopy", "radiative forcing", "climate change",
        "detection", "source attribution", "fugitive emissions", "natural gas", "oil and gas", "agriculture",
    ]
}




def extract_keywords_per_cluster(clustered_chunks, allowed_keywords, top_n=3, sample_size=100):
    """
    Extracts representative keywords for each cluster from allowed keywords,
    by sampling a subset of chunks per cluster.

    Args:
        clustered_chunks (dict): Cluster label → list of text chunks.
        allowed_keywords (dict): Dict of category → list of keywords.
        top_n (int): Number of top keywords to extract per cluster.
        sample_size (int): Max number of chunks to use per cluster.
    
    Returns:
        dict: Cluster label → list of top_n (keyword, category).
    """
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text

    cluster_keywords = {}

    # Flatten the allowed keywords into (keyword, category) pairs
    keyword_list = [(kw, cat) for cat, kws in allowed_keywords.items() for kw in kws]
    categories = list(set([cat for kw, cat in keyword_list]))  # Get unique categories from allowed_keywords.keys()

    for label, chunks in clustered_chunks.items():
        # Sample or use all chunks
        if len(chunks) > sample_size:
            sampled_chunks = random.sample(chunks, sample_size)
        else:
            sampled_chunks = chunks

        text = " ".join(sampled_chunks)
        text_prep = preprocess(text)

        matches = []
        sum_category = [0] * len(categories)  # Initialize sum for each category
        for keyword, category in keyword_list:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_prep):
                matches.append((keyword, category, len(keyword)))  # Store length
                sum_category[categories.index(category)] += 1

        if matches:
            # take best category
            best_category = categories[np.argmax(sum_category)]
            cluster_keywords[label] = best_category

        else:
            cluster_keywords[label] = [(None, None)] * top_n  # If no matches, return None

    return cluster_keywords



def extract_keywords_per_cluster_keybert(clustered_chunks, sample_size=100, top_n=1):
    """
    Extracts representative keywords for each cluster using KeyBERT,
    by sampling a subset of chunks per cluster.

    Args:
        clustered_chunks (dict): Cluster label → list of text chunks.
        top_n (int): Number of top keywords to return per cluster.
        sample_size (int): Max number of chunks to use per cluster.
        model_name (str): Name of embedding model for KeyBERT.

    Returns:
        dict: Cluster label → list of keywords.
    """

    french = stopwords.words('french')
    english = stopwords.words('english')
    spanish = stopwords.words('spanish')

    multi_stopwords = list(set(french + english + spanish))
    
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct")
    # Initialize KeyBERT with the multilingual model
    kw_model = KeyBERT(model)
    
    cluster_keywords = {}

    for label, chunks in tqdm(clustered_chunks.items()):
        if len(chunks) > sample_size:
            sampled_chunks = random.sample(chunks, sample_size)
        else:
            sampled_chunks = chunks

        text = " ".join(sampled_chunks)
        keywords = kw_model.extract_keywords(text,
                                             keyphrase_ngram_range=(2, 3),
                                             stop_words=multi_stopwords,
                                             top_n=top_n)
        cluster_keywords[label] = [kw[0] for kw in keywords]

    return cluster_keywords


# Function to calculate cosine similarity
def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
import random
from tqdm import tqdm
from nltk.corpus import stopwords


def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate the cosine similarity between two embeddings."""
    return cosine_similarity([embedding1], [embedding2])[0][0]


def extract_keywords_with_similarity(clustered_chunks, allowed_keywords, sample_size=1000, top_n=5):
    """
    Extracts representative keywords for each cluster using KeyBERT and matches them
    against the structured keywords using cosine similarity.
    
    Args:
        clustered_chunks (dict): Cluster label → list of text chunks.
        allowed_keywords (dict): Dict of category → list of keywords.
        sample_size (int): Max number of chunks to use per cluster.
        top_n (int): Number of top keywords to return per cluster.
    
    Returns:
        dict: Cluster label → list of keywords.
    """
    # Load stopwords only once
    french = stopwords.words('french')
    english = stopwords.words('english')
    spanish = stopwords.words('spanish')
    multi_stopwords = list(set(french + english + spanish))
    
    # Initialize models once
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    kw_model = KeyBERT(model)
    
    # Flatten allowed keywords into a list of (keyword, category) pairs
    categories = list(allowed_keywords.keys())
    
    cluster_keywords = {}

    # Iterate through the clusters and extract keywords
    for label, chunks in tqdm(clustered_chunks.items()):
        # Sample the chunks if they exceed sample_size
        sampled_chunks = random.sample(chunks, sample_size) if len(chunks) > sample_size else chunks
        text = " ".join(sampled_chunks)
        
        # Extract keywords from KeyBERT
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words=multi_stopwords, top_n=top_n)
        keybert_keywords = " ".join([kw[0] for kw in keywords])
        
        # Generate the KeyBERT embedding
        keybert_embedding = model.encode(keybert_keywords)
        
        # Store similarities for each category
        best_category = None
        best_similarity = -1  # Initialize to a very low similarity
        
        # Compare with each category's allowed keywords
        for category in categories:
            allowed_keywords_category = " ".join(allowed_keywords[category])
            allowed_embedding = model.encode(allowed_keywords_category)
            
            # Calculate cosine similarity
            similarity = calculate_cosine_similarity(keybert_embedding, allowed_embedding)
            
            # Update best category if current similarity is higher
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category
        
        # Add the best category for this cluster
        cluster_keywords[label] = best_category

    return cluster_keywords



def cluster_text(embeddings, questions, eps=0.5, min_cluster_size=5, min_samples=5):
    """
    Cluster the text based on their similarity.
    
    Args:
        questions (list): List of generated questions.
    
    Returns:
        list: List of clustered questions.
    """
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=eps,
        min_samples=min_samples,
        )
    labels = clusterer.fit_predict(embeddings)
    
    clusters = defaultdict(list)
    for label, question in zip(labels, questions):
        label = str(label)  # Convert label to string for JSON serialization
        clusters[label].append(question)
    
    return clusters, labels


def get_umap_embeddings(embeddings, output_dim=2):
    """
    Generate UMAP embeddings for the given embeddings.
    
    Args:
        embeddings (list): List of question embeddings.
    
    Returns:
        np.ndarray: UMAP embeddings.
    """
    
    reducer = UMAP(n_components=output_dim, n_neighbors=15, min_dist=0.0, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings


if __name__ == "__main__":

    input_jsonl = ProductionConfig.CHUNK_PATH
    embeddings_path = f"{ProductionConfig.EMBEDDINGS_FOLDER}/embeddings.npy"
    save_path = "/home/lucasd/code/rag/data_processing/data_analysis/"

    # Load the generated questions
    with open(input_jsonl, "r") as f:
        text = [json.loads(line)["text"] for line in f]

    # get embeddings for the questions
    print("Generating embeddings...")
    embeddings = np.load(embeddings_path)

    # get umap embeddings
    umap30D_embeddings = get_umap_embeddings(embeddings, output_dim=30)
    
    # Cluster the questions
    print("Clustering questions...")
    clustered_questions, labels = cluster_text(umap30D_embeddings, text, eps=0.2, min_cluster_size=300, min_samples=60)

    # Save the clustered questions to a JSON file
    with open(save_path + "clustered_questions.json", "w") as f:
        json.dump(clustered_questions, f, indent=4)
    print("Clustered questions saved to clustered_questions.json")

    # Get keywords for each cluster
    print("Extracting keywords...")
    # keywords = extract_keywords_with_similarity(clustered_questions, structured_keywords, sample_size=1000, top_n=3)
    keywords = extract_keywords_per_cluster_keybert(clustered_questions, sample_size=1000, top_n=1)
    # Save the keywords to a JSON file
    with open(save_path + "keywords.json", "w") as f:
        json.dump(keywords, f, indent=4)

    # Map the cluster labels to keywords
    display_labels = {}
    for label in set(labels):
        label_str = str(label)
        if label_str in keywords:
            display_labels[label] = keywords[label_str]  # e.g. "methane detection"
        else:
            display_labels.append("Not found")  # e.g. "Not found"


    print("Labels: ", labels)

    # Run UMAP on the embeddings to reduce to 2D
    umap2D_embeddings = get_umap_embeddings(embeddings, output_dim=2)
    centroids = np.array([np.mean(umap2D_embeddings[labels == label], axis=0) for label in set(labels)])

    # # if the chunk has 'UHR' in the text, set as label -5
    # for i, text in enumerate(text):
    #     if "france" in text.lower():
    #         labels[i] = -5


    # Set a clean style
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Create figure
    plt.figure(figsize=(40, 40))
    plt.title("Cluster Visualization with Centroids", fontsize=18, weight='bold')
    plt.xlabel("UMAP Dimension 1", fontsize=14)
    plt.ylabel("UMAP Dimension 2", fontsize=14)

    # Plot clusters
    unique_labels = set(labels)
    colors = sns.color_palette("hsv", len(unique_labels))  # Vibrant palette

    for idx, label in enumerate(unique_labels):
        is_noise = (label == -1)
        cluster_color = 'gray' if is_noise else colors[idx % len(colors)]
        label_name = 'Noise' if is_noise else display_labels.get(label, f"Cluster {label}")

        # Scatter points
        plt.scatter(
            umap2D_embeddings[labels == label, 0],
            umap2D_embeddings[labels == label, 1],
            color=cluster_color,
            label=label_name,
            alpha=0.6,
            edgecolors='w',
            s=60
        )

        # Plot centroid and text
        if not is_noise:
            centroid = centroids[label]
            
            # Plot the centroid with a larger, more visible marker
            plt.scatter(centroid[0], centroid[1], color='black', marker='x', s=200, linewidths=4)

            # Add text with a background color and adjusted position
            plt.text(
                centroid[0] + 0.03, centroid[1] + 0.03,  # Adjust positioning to prevent overlap
                display_labels[label],
                fontsize=18,  # Larger font size
                weight='bold',  # Bold font weight
                color='white',  # White text color
                ha='center',  # Horizontal alignment (centered)
                va='center',  # Vertical alignment (centered)
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5')  # Background color with padding
            )
    
    # Legend and layout
    plt.tight_layout()
    plt.savefig(save_path + "umap_projection_clusters.png")
    plt.show()

