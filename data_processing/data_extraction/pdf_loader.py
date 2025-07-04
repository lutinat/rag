import pdfplumber
import os
from typing import Dict, List, Tuple
import re
from glob import glob
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.user_credential import UserCredential
from getpass import getpass
import spacy
import pandas as pd
import concurrent.futures
from tqdm import tqdm

from spacy_layout import spaCyLayout


def load_pdf_spacy(pdf_path: str) -> str:
    # Sample function to create a custom display of the table
    def display_table(df: pd.DataFrame) -> str:
        return f"Table with columns: {', '.join(df.columns.tolist())}"

    nlp = spacy.load("xx_ent_wiki_sm")
    layout = spaCyLayout(nlp)

    # Process a document and create a spaCy Doc object
    doc = layout(pdf_path)

    # The text-based contents of the document
    # print(doc.text)

    # # Document layout including pages and page sizes
    # print(doc._.layout)

    # Tables in the document and their extracted data
    # print(doc._.tables)

    #Iterate through the tables in the document
    #Markdown representation of the document
    # print(doc._.markdown)

    return doc.text


def load_pdfs_spacy(pdf_path_list: List[str]) -> List[str]:
    # TODO: Not working
    nlp = spacy.load("xx_ent_wiki_sm")
    layout = spaCyLayout(nlp)

    # Function to process PDF paths
    def process_pdfs(pdf_paths):
        return list(layout.pipe(pdf_paths))

    # Number of processes
    n_processes = 12

    # Split the list of PDFs into chunks for parallel processing
    chunk_size = len(pdf_path_list) // n_processes
    chunks = [pdf_path_list[i:i + chunk_size] for i in range(0, len(pdf_path_list), chunk_size)]

    # Now use tqdm for individual PDFs, not chunks
    all_docs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_processes) as executor:
        # Use tqdm to show the progress of processing PDFs
        for chunk_result in tqdm(executor.map(process_pdfs, chunks), total=len(pdf_path_list), desc="Processing PDFs", unit="pdf"):
            all_docs.extend(chunk_result)  # Flatten the result from each chunk

    return all_docs



def load_txt_web(txt_path: str) -> str:
    """Loads a text file and returns its content as a string."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()


def clean_txt_file(input_path, output_path=None):
    unwanted_lines = {
        "About Us", "What we offer", "Missions",
        "Applications", "Work With Us", "News", "Contact"
    }

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = [line for line in lines if line.strip() not in unwanted_lines]

    # If no output path is given, overwrite the input file
    if output_path is None:
        output_path = input_path

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    print(f"Cleaned file saved to: {output_path}")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from pdf using pdfplumber
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text
    """
    text_blocks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract words with their positions (top, bottom, left, right)
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=True,
                horizontal_ltr=True,
                vertical_ttb=True,
                extra_attrs=["fontname", "size"]
            )
            
            # Group words by line
            lines = {}
            for word in words:
                y = round(word['top'], 1)  # Group close words
                if y not in lines:
                    lines[y] = []
                lines[y].append(word)
            
            # Sort lines by their vertical position
            sorted_lines = sorted(lines.items())
            
            # Reconstruct text from lines
            for y, words in sorted_lines:
                # Sort words from left to right
                words.sort(key=lambda w: w['x0'])
                line_text = ' '.join(w['text'] for w in words)
                
                # Detect titles based on font size and position
                is_title = any(
                    w['size'] > 12 for w in words
                ) or y < 100
                
                if is_title:
                    # Mark the title with a specific format
                    text_blocks.append(f"\n## {line_text}\n")
                else:
                    text_blocks.append(line_text)
            
            text_blocks.append("\n")  # 
    
    # Retyrn the text as a single string
    raw_text = "\n".join(text_blocks)
    clean_text = " ".join(raw_text.split())
    return clean_text


def extract_metadata_from_pdf(pdf_path: str) -> Dict:
    """
    Extract metadata from pdf using pdfplumber
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dict: Extracted metadata
    """
    metadata = {
        "title_from_pdf": None,
        "title_from_metadata": None,
        "filename": None,
        # "authors": None,
        "creation_date": None,
        "subject": None,
        "keywords": None
    }
    
    # Get filename and title from pdf
    try:
        title_from_pdf = get_title_from_pdf(pdf_path)
    except Exception as e:
        print(f"Error extracting title from PDF: {e}")
        title_from_pdf = None
    filename = os.path.basename(pdf_path)
    metadata["title_from_pdf"] = title_from_pdf
    metadata["filename"] = filename

    # Get metadata from pdf
    metadata_from_pdf = get_metadata_plumberpdf(pdf_path)
    metadata["title_from_metadata"] = metadata_from_pdf["title"]
    # metadata["authors"] = metadata_from_pdf["author"]
    metadata["creation_date"] = metadata_from_pdf["creation_date"]
    metadata["subject"] = metadata_from_pdf["subject"]
    metadata["keywords"] = metadata_from_pdf["keywords"]
    
    return metadata



def get_title_from_pdf(pdf_path: str) -> str:
    min_font_size = 15
    with pdfplumber.open(pdf_path) as pdf:
        # Get the first page
        first_page = pdf.pages[0]
        
        # Extract words with their positions (top, bottom, left, right)
        layout_objects = first_page.extract_words()

        # Initialize variables for sentence segmentation and font size tracking
        sentences = []
        current_sentence = []
        current_font_sizes = []
        last_font_size = None
        
        for obj in layout_objects:
            word = obj['text']
            font_size = obj['bottom'] - obj['top']  # Font size as the difference between top and bottom
            
            # If the font size is significantly different, consider it a new sentence
            if last_font_size and abs(last_font_size - font_size) > 1:  # Change threshold can be adjusted
                if current_sentence:
                    sentences.append({
                        'sentence': ' '.join(current_sentence),
                        'avg_font_size': sum(current_font_sizes) / len(current_font_sizes) if current_font_sizes else 0
                    })
                    current_sentence = []
                    current_font_sizes = []
            
            # Add word and its font size to the current sentence
            current_sentence.append(word)
            current_font_sizes.append(font_size)
            last_font_size = font_size
        

        # If there's any leftover sentence, append it
        if current_sentence:
            sentences.append({
                'sentence': ' '.join(current_sentence),
                'avg_font_size': sum(current_font_sizes) / len(current_font_sizes) if current_font_sizes else 0
            })

        # Remove sentences with average font size less than min_font_size
        sentences = [s for s in sentences if s['avg_font_size'] > min_font_size]
        
        # Find the sentence with the largest average font size (likely the title)
        if sentences:
            title = max(sentences, key=lambda x: x['avg_font_size'])['sentence']
        else:
            title = None

    return title


def get_metadata_plumberpdf(pdf_path: str) -> Dict:
    metadata = {
        "title": None,
        "author": None,
        "subject": None,
        "keywords": None,
        "creation_date": None
    }
    with pdfplumber.open(pdf_path) as pdf:
        if pdf.metadata:
            metadata.update({
                k.lower(): v for k, v in pdf.metadata.items()
                if k.lower() in metadata
            })
    return metadata


if __name__ == "__main__":
    from office365.sharepoint.client_context import ClientContext
    from office365.runtime.auth.user_credential import UserCredential

    base_url = "https://satlantis.sharepoint.com"

    # SharePoint site URL
    sharepoint_sites = ["/sites/ImgProcss", "/sites/intranet", "/sites/SATLANTISFrance"]

    # Document library names for each site
    document_libraries = {
        "/sites/ImgProcss": "Documents",
        "/sites/intranet": "Documentos%20compartidos", 
        "/sites/SATLANTISFrance": "Documents"
    }

    sharepoint_site = sharepoint_sites[0]
    site_url = base_url + sharepoint_site

    # Your credentials
    username = ""
    password = ""

    ctx = ClientContext(site_url).with_credentials(UserCredential(username, password))

    def list_pdfs_recursively(folder_relative_url):
        folder = ctx.web.get_folder_by_server_relative_url(folder_relative_url)
        ctx.load(folder)
        ctx.execute_query()

        # Lister les fichiers dans le dossier
        files = folder.files
        ctx.load(files)
        ctx.execute_query()
        for file in files:
            if file.properties["Name"].lower().endswith(".pdf"):
                # Get the file's server relative URL
                file_relative_url = file.properties["ServerRelativeUrl"]
                # Construct the full download URL
                file_download_url = f"{base_url}{file_relative_url}"
                
                # Construct SharePoint viewer URL (opens in SharePoint's PDF viewer)
                file_viewer_url = f"{base_url}{sharepoint_site}/_layouts/15/WopiFrame.aspx?sourcedoc={file_relative_url}&action=default"
                
                print(f"PDF: {file.properties['Name']}")
                print(f"Direct Link: {file_download_url}")
                print(f"SharePoint Viewer: {file_viewer_url}")
                print(f"Page 5 Link (PDF viewer): {file_download_url}#page=5")
                print(f"Page 10 Link (PDF viewer): {file_download_url}#page=10")
                print(f"Path: {folder_relative_url}/{file.properties['Name']}")
                print("---")

        # Lister les sous-dossiers
        folders = folder.folders
        ctx.load(folders)
        ctx.execute_query()
        for subfolder in folders:
            subfolder_url = subfolder.properties["ServerRelativeUrl"]
            list_pdfs_recursively(subfolder_url)

    # Get the appropriate document library for the selected site
    document_library = document_libraries.get(sharepoint_site, "Documents")
    
    # Démarrer la recherche à la racine de la bibliothèque Documents
    list_pdfs_recursively(sharepoint_site + "/" + document_library)