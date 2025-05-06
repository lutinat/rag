import re
import spacy
from typing import List, Dict
from utils import save_chunks_jsonl
import os
from glob import glob
from .pdf_loader import extract_text_from_pdf, extract_metadata_from_pdf, load_pdf_spacy
from .pdf_loader import load_txt_web
from tqdm import tqdm
import subprocess
import numpy as np

# Load the spaCy model for sentence segmentation
# Note: The model "xx_ent_wiki_sm" is a multilingual model.
nlp = spacy.load("xx_ent_wiki_sm")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")


def is_author_like_chunk(text: str) -> bool:
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return False

    # Remove punctuation from words
    cleaned_words = [re.sub(r'[^\w]', '', w) for w in words]

    # Count capitalized words
    capitalized_words = sum(1 for w in cleaned_words if re.match(r'^[A-Z][a-z]+$', w))

    capital_ratio = capitalized_words / total_words
    comma_count = text.count(',')
    has_end_punctuation = text.strip().endswith(('.', '!', '?'))

    # Final decision
    if capital_ratio > 0.4 and comma_count >= 3 and not has_end_punctuation and total_words < 100:
        return True

    return False


def is_noise(text: str) -> bool:
    """Detrermine if the text is noise. If it is, return True."""
    # Noise patterns to check
    noise_patterns = [
        r'^\d+\.',  # Section numbers
        r'^PARTIE \d+',  # Part titles
        r'^\.{10,}$',  # Lines of dots
        r'^\s*\d+\s*$',  # Standalone page numbers
        r'^.{1,30}\.\.\.+\d+$',  # Titles with ellipsis and page number
        r'^\s*$',  # Empty lines
        r'^.{1,50}\.\.\.+\s*\d+$',  # Other titles with ellipsis and page number
        r'^[A-Z][a-z]*\s*\.\.\.+\s*\d+$',  # Short titles with ellipsis and page number
        r'^[IVX]+\.',  # Roman numerals
        r'^\d+\.\d+',  # Numbered subsections
        r'^[A-Z]\.',  # Single uppercase letters
        r'^[a-z]\)',  # Lowercase letters with parenthesis
        r'^\d+\)',  # Numbers with parenthesis
        r'^Table des matières',  # Table of contents headings
        r'^[A-Z]{2,}$',  # Standalone acronyms
        r'^\d{1,2}/\d{1,2}/\d{4}',  # Dates (e.g., 12/31/2023)
        r'^©',  # Copyright
        r'^http[s]?://',  # URLs
        r'^www\.',  # URLs
        r'^[A-Z][a-z]*\s*:\s*$',  # Titles followed by a colon
        r'^\d{1,2}h\d{2}',  # Time (e.g., 14h30)
        r'^\d{1,2}:\d{2}',  # Time (e.g., 14:30)
        r'^\d{1,2}/\d{1,2}',  # Short dates (e.g., 12/31)
        r'^\d{4}-\d{2}-\d{2}',  # ISO dates (e.g., 2024-04-30)
    ]
    
    # Check if the text matches any of the noise patterns
    if any(re.match(pattern, text.strip()) for pattern in noise_patterns):
        return True
        
    # Check if the text is too short
    if len(text.strip()) < 30:
        return True
        
    # Check if the text is too long
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    if len(clean_text) < len(text) * 0.4:
        return True
        
    # Check if the text contains too many uppercase letters
    if len(re.findall(r'[A-Z]', text)) > len(text) * 0.4:
        return True
    
    if is_author_like_chunk(text):
        return True
        
    return False

def clean_text(text: str) -> str:
    """Denoise and clean the text."""
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters at the beginning and end
    text = re.sub(r'^[^\w\s\.\?!]+|[^\w\s\.\?!]+$', '', text)

    # Normalize quotes
    text = re.sub(r'["\']', '"', text)
    
    # Normalize dashes
    text = re.sub(r'[-–—]', '-', text)
    
    # Remove dots page numbers
    text = re.sub(r'\.\.\.+\s*\d+$', '', text)
    
    # Remove standalone page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text)

    text = re.sub(r'(\s*\.\s*){2,}', '.', text)
    
    # Remove bullet points
    text = re.sub(r'^\s*[•\-\*]\s*', '', text)
    
    # text = re.sub(r'\s*\|\s*', ' ', text)  # Remplacer les séparateurs de colonnes
    # text = re.sub(r'\s*-\s*', ' ', text)   # Remplacer les lignes de séparation
    
    # # Ensure space after punctuation
    # text = re.sub(r'([.!?])\s*', r'\1 ', text)

    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """Divide the text into sentences."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


def create_paragraphs(text: str) -> List[str]:
    """Create paragraphs and sentences from the text.
        Input: text
        Output: paragraphs, sentences"""

    # Divide the text into paragraphs 
    paragraphs = text.split('\n\n')
    
    clean_sentences = []
    clean_paragraphs = []
    
    for i, paragraph in enumerate(paragraphs):
        # Skip empty paragraph
        if not paragraph.strip():
            continue

        # Clean the paragraph before processing
        cleaned = clean_text(paragraph)
        
        # Check if the cleaned paragraph is too short
        if len(cleaned) < 50:
            continue

        # Detect sentences using spaCy (> 1000000 to avoid spacy error)
        sentences = []
        if len(cleaned) > 1_000_000:
            for i in range(0, len(cleaned), 1_000_000):
                batch_sentences = split_into_sentences(cleaned[i:i+1_000_000])
                sentences.extend(batch_sentences)
        else:
            sentences = split_into_sentences(cleaned)

        current_clean_paragraph = ""
        for sentence in sentences:
            if is_noise(sentence):
                continue
            
            # Add a point at the end of the sentence if it doesn't have one
            if sentence and sentence[-1] not in ['.', '!', '?']:
                sentence += '.'

            # Add the sentence to the current paragraph and clean sentence
            current_clean_paragraph += sentence + ' '
            if not is_similar(sentence, clean_sentences[-1] if clean_sentences else "", threshold=0.8):
                clean_sentences.append(sentence)

        # Add the cleaned paragraph to the list
        if current_clean_paragraph:
            if current_clean_paragraph not in clean_paragraphs:
                clean_paragraphs.append(current_clean_paragraph)
    
    return clean_paragraphs, clean_sentences

def create_chunks_from_sentences(sentences: List[str], 
                                overlap: int = 0.2,
                                min_chunk_tokens: int = 300,
                                max_chunk_tokens: int = 500, 
                                source_name: str = "document.pdf",
                                metadata: Dict = {}) -> List[Dict]:
    """Create chunks from sentences with overlap."""

    # Helper function to split a paragraph into words
    def _split_into_words(paragraph: str):
        return paragraph.split()
    
    chunks = []
    current_chunk_id = 0
    current_chunk = []
    current_token_count = 0
    current_sentences_token_count = []
    

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        current_chunk.append(sentence)
        n_tokens = len(_split_into_words(sentence))
        current_token_count += n_tokens
        current_sentences_token_count.append(current_token_count)

        # If the chunk is too small, continue adding sentences
        if current_token_count < min_chunk_tokens:
            i += 1
            continue

        else:
            # If the chunk is too large, we need to split it
            if current_token_count > max_chunk_tokens:
                # TODO : check this code to see if it works
                # We cannot split by sentences and have tokens count above min and below max
                # Split in to smaller chunks in a brute way with overlap

                # Find the ratio to split
                split_ratio = current_token_count / max_chunk_tokens
                split_ratio = np.ceil(split_ratio)
                split_ratio = int(split_ratio)
                split_ratio = max(split_ratio, 1)
                size = len(current_chunk) // split_ratio

                # Split the chunk into smaller chunks
                for j in range(split_ratio):
                    chunk = current_chunk[j*size:(j+1)*size]
                    chunk_text = ' '.join(chunk)
                    chunk_data = {
                        "id": f"{source_name}_c{current_chunk_id}",
                        "text": chunk_text,
                        "metadata": metadata
                    }
                    chunks.append(chunk_data)
                    current_chunk_id += 1
                

            # If the chunks has enough tokens, save it
            elif current_token_count >= min_chunk_tokens:
                chunk_text = ' '.join(current_chunk)
                chunk_data = {
                    "id": f"{source_name}_c{current_chunk_id}",
                    "text": chunk_text,
                    "metadata": metadata
                }
                chunks.append(chunk_data)
                current_chunk_id += 1
        
            else:
                raise ValueError("Should not happen")

            # Overlap for the next chunk
            next_sentence_idx = i+1
            overlap_size = int(current_token_count * overlap)
            for j, value in enumerate(current_sentences_token_count[::-1]):
                difference = current_token_count - value
                if difference > overlap_size:
                    next_sentence_idx = i - j + 1
                    break
            
            # Initialize new chunk
            i = next_sentence_idx
            current_chunk = []
            current_token_count = 0
            current_sentences_token_count = []
  
    return chunks

def extract_chunks(text: str, source_name: str, metadata: Dict) -> List[Dict]:
    """Extract chunks from the text."""
    # Get the sentences from the text
    _, sentences = create_paragraphs(text)
    
    # Create chunks from the sentences
    chunks = create_chunks_from_sentences(
        sentences,
        overlap=0.2,     # 20% overlap
        min_chunk_tokens=200,
        max_chunk_tokens=500, 
        source_name=source_name,
        metadata=metadata
    )
    
    return chunks

def is_similar(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Check if two texts are similar based on common words."""
    # Convert texts to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Compute the similarity based on common words
    common_words = words1.intersection(words2)
    similarity = len(common_words) / max(len(words1), len(words2))
    
    return similarity > threshold


# Function to download the PDF to a temporary directory
def download_pdf_to_temp(remote_path, temp_dir=None):
    # Create a temporary directory (automatically cleaned up)
    # Construct the local path for the downloaded PDF
    local_pdf_path = os.path.join(temp_dir, os.path.basename(remote_path))
    
    # Check if the file already exists
    if os.path.exists(local_pdf_path):
        raise Exception(f"File already exists: {local_pdf_path}")
    
    # Use rclone to download the PDF to the temp folder
    subprocess.run(['rclone', 'copyto', remote_path, local_pdf_path], check=True)
    
    # Check if the path is a directory, not a file
    if os.path.isdir(local_pdf_path):
        print(f"Error: {local_pdf_path} is a directory, not a file.")
        return None
    
    # Verify the file actually exists and is a valid file
    if not os.path.isfile(local_pdf_path):
        print(f"Error: {local_pdf_path} does not exist or is not a valid file.")
        return None
    
    return local_pdf_path


def get_all_chunks(folder_path: str, chunk_folder_path: str):
    """
    Process multiple PDFs, extract text and metadata, chunk the text, and save chunks to JSONL.
    
    Args:
        folder_path (str): The directory containing PDFs to process.
        chunk_folder_path (str): The folder where JSONL files will be saved.
    """
    # Get all PDF files in the directory
    pdf_paths = glob(os.path.join(folder_path, "*.pdf"))
    txt_paths = sorted(glob(os.path.join(folder_path, "*.txt")))
    csv_paths = glob(os.path.join(folder_path, "*.csv"))
    all_chunks = []

    # PDF FILES
    print(f"Processing {len(pdf_paths)} PDFs")
    for pdf_path in tqdm(pdf_paths):
        print(f"Processing {pdf_path}")
        # Extract text and metadata
        try:
            text = load_pdf_spacy(pdf_path)
        except Exception as e:
            print(f"Failed to extract from {pdf_path}: {e}")
            continue

        metadata = extract_metadata_from_pdf(pdf_path)
        
        # Extract chunks from the text
        chunks = extract_chunks(text, source_name=pdf_path, metadata=metadata)
        all_chunks.extend(chunks)
        
        # Generate a filename for the JSONL chunk file based on the PDF name
        chunk_filename = os.path.basename(pdf_path).replace(".pdf", "_chunks.jsonl")
        chunk_path = os.path.join(chunk_folder_path, chunk_filename)
        
        # Save the chunks to JSONL
        save_chunks_jsonl(chunks, chunk_path)


    # TXT FILES
    print(f"Processing {len(txt_paths)} TXT files")
    for txt_path in tqdm(txt_paths):
        # print(f"Processing {txt_path}")
        text = load_txt_web(txt_path)
        metadata = {'filename': os.path.basename(txt_path)}
        chunks = extract_chunks(text, source_name=txt_path, metadata=metadata)
        all_chunks.extend(chunks)

        # Generate a filename for the JSONL chunk file based on the TXT name
        chunk_filename = os.path.basename(txt_path).replace(".txt", "_chunks.jsonl")
        chunk_path = os.path.join(chunk_folder_path, chunk_filename)

        # Save the chunks to JSONL
        save_chunks_jsonl(chunks, chunk_path)
    
    # CSV FILES
    # TODO: Implement CSV files
    
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    
    folder_path = "/home/lucasd/code/rag/data/"
    chunk_folder_path = "/home/lucasd/code/rag/processed_data/"
    get_all_chunks(folder_path, chunk_folder_path)