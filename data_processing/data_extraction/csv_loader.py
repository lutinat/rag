import os
from typing import Dict, List, Tuple
import re
from glob import glob
from getpass import getpass
import spacy
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from pathlib import Path

from docling.document_converter import DocumentConverter


def load_csv_spacy(csv_path: str) -> str:
    # read the csv file
    df = pd.read_csv(csv_path)
    print(df)

    # extract each row to create a phrase
    for index, row in df.iterrows():
        text = row['Name']
        print(text)

def load_csv_docling(csv_path: str) -> str:
    """
    Load and extract text from a CSV file using docling.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        str: Extracted text content
    """
    try:
        # Convert CSV to Docling document
        converter = DocumentConverter()
        doc = converter.convert(csv_path)
        
        # Extract the text from the docling document
        text = doc.document.export_to_markdown()
        
        return text
        
    except Exception as e:
        print(f"Error processing CSV file {csv_path}: {e}")
        return ""
    
    
if __name__ == "__main__":
    load_csv_docling("/home/elduayen/rag/data/filter.csv")




