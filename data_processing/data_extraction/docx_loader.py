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


def load_docx_docling(docx_path: str) -> str:
    """
    Load and extract text from a DOCX file using docling.
    
    Args:
        docx_path: Path to the DOCX file
        
    Returns:
        str: Extracted text content
    """
    try:
        # Convert DOCX to Docling document
        converter = DocumentConverter()
        doc = converter.convert(docx_path)
        
        # Extract the text from the docling document
        text = doc.document.export_to_markdown()
        
        return text
        
    except Exception as e:
        print(f"Error processing DOCX file {docx_path}: {e}")
        return ""
    
    
if __name__ == "__main__":
    load_docx_docling("/home/elduayen/rag/data/strategie Satlantis France Rev1.docx")