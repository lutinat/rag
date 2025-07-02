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
from dotenv import load_dotenv
import tempfile
import requests
from pathlib import Path

from spacy_layout import spaCyLayout

# Import other loaders
from .docx_loader import load_docx_docling
from .csv_loader import load_csv_docling


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


def process_sharepoint_documents_to_chunks(sharepoint_site: str = None, 
                                          document_library: str = None,
                                          output_folder: str = "processed_data") -> List[Dict]:
    """
    Process all documents from SharePoint and convert them to chunks with metadata.
    Uses the existing get_all_chunks function for processing.
    
    Args:
        sharepoint_site: SharePoint site to process (if None, uses first site)
        document_library: Document library to process (if None, uses default)
        output_folder: Output folder for chunks
        
    Returns:
        List[Dict]: List of chunks with metadata
    """
    # Load environment variables
    load_dotenv()
    
    base_url = "https://satlantis.sharepoint.com"
    
    # SharePoint site URL
    sharepoint_sites = ["/sites/SATLANTISFrance"]
    
    # Document library names for each site
    document_libraries = {
        "/sites/SATLANTISFrance": "Documents/Projects/2024_FUSE-POLARISAT"
    }
    
    if sharepoint_site is None:
        sharepoint_site = sharepoint_sites[0]
    
    if document_library is None:
        document_library = document_libraries.get(sharepoint_site, "Documents")
    
    site_url = base_url + sharepoint_site
    
    # Get credentials from environment variables
    username = os.getenv("SHAREPOINT_USERNAME")
    password = os.getenv("SHAREPOINT_PASSWORD")
    
    if not username or not password:
        raise ValueError("SHAREPOINT_USERNAME and SHAREPOINT_PASSWORD must be set in .env file")
    
    # Create temporary folder for downloaded files
    temp_folder = f"temp_sharepoint_{sharepoint_site.replace('/', '_')}"
    os.makedirs(temp_folder, exist_ok=True)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    def create_sharepoint_context():
        """Create a new SharePoint context."""
        return ClientContext(site_url).with_credentials(UserCredential(username, password))
    
    def download_document(file_info, folder_relative_url):
        """Download a document and add SharePoint metadata."""
        try:
            # Create a new context for each download to avoid closed file issues
            ctx = create_sharepoint_context()
            
            file_name = file_info["Name"]
            file_relative_url = file_info["ServerRelativeUrl"]
            file_download_url = f"{base_url}{file_relative_url}"
            file_viewer_url = f"{base_url}{sharepoint_site}/_layouts/15/WopiFrame.aspx?sourcedoc={file_relative_url}&action=default"
            
            print(f"🔄 Téléchargement de : {file_name}")
            print(f"   URL relative : {file_relative_url}")
            print(f"   Taille SharePoint : {file_info.get('Length', 'inconnue')} bytes")
            
            # Create local file path
            local_path = os.path.join(temp_folder, file_name)
            
            # Get the file object
            file = ctx.web.get_file_by_server_relative_url(file_relative_url)
            ctx.load(file)
            ctx.execute_query()
            
            # Download the file
            try:
                with open(local_path, 'wb') as local_file:
                    file.download(local_file).execute_query()
                
                # Vérifier que le fichier n'est pas vide
                if os.path.getsize(local_path) == 0:
                    print(f"⚠️ Fichier vide téléchargé : {file_name}")
                    os.remove(local_path)  # Supprimer le fichier vide
                    return None
                else:
                    print(f"✅ Fichier téléchargé avec succès : {file_name} ({os.path.getsize(local_path)} bytes)")
                    
            except Exception as download_error:
                print(f"❌ Error downloading file {file_name}: {download_error}")
                return None
            
            # Add SharePoint metadata to the file
            metadata_file = local_path + ".meta"
            metadata = {
                "source": "sharepoint",
                "sharepoint_site": sharepoint_site,
                "document_library": document_library,
                "folder_path": folder_relative_url,
                "file_name": file_name,
                "file_relative_url": file_relative_url,
                "download_url": file_download_url,
                "viewer_url": file_viewer_url,
                "file_size": file_info.get("Length", 0),
                "created_date": str(file_info.get("TimeCreated", "")) if file_info.get("TimeCreated") else "",
                "modified_date": str(file_info.get("TimeLastModified", "")) if file_info.get("TimeLastModified") else "",
                "author": file_info.get("Author", ""),
                "file_type": Path(file_name).suffix.lower()
            }
            
            try:
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"📝 Métadonnées sauvegardées : {metadata_file}")
            except Exception as meta_error:
                print(f"⚠️ Error writing metadata for {file_name}: {meta_error}")
            
            return local_path
            
        except Exception as e:
            print(f"❌ Error downloading file {file_info.get('Name', 'unknown')}: {e}")
            return None
    
    def list_and_download_documents_recursively(folder_relative_url):
        """List and download all documents in a folder recursively."""
        try:
            # Create a new context for each folder operation
            ctx = create_sharepoint_context()
            
            folder = ctx.web.get_folder_by_server_relative_url(folder_relative_url)
            ctx.load(folder)
            ctx.execute_query()
            
            # Process files in current folder
            try:
                files = folder.files
                ctx.load(files)
                ctx.execute_query()
                
                # Get file information first
                file_infos = []
                for file in files:
                    file_infos.append({
                        "Name": file.properties["Name"],
                        "ServerRelativeUrl": file.properties["ServerRelativeUrl"],
                        "Length": file.properties.get("Length", 0),
                        "TimeCreated": file.properties.get("TimeCreated"),
                        "TimeLastModified": file.properties.get("TimeLastModified"),
                        "Author": file.properties.get("Author", "")
                    })
                
                # Now download each file
                for file_info in file_infos:
                    file_name = file_info["Name"]
                    file_extension = Path(file_name).suffix.lower()
                    
                    # Only process supported file types
                    if file_extension in ['.pdf', '.docx', '.doc', '.csv', '.txt']:
                        print(f"Downloading: {file_name}")
                        local_path = download_document(file_info, folder_relative_url)
                        if local_path:
                            print(f"Downloaded: {local_path}")
                            
            except Exception as files_error:
                print(f"Error processing files in folder {folder_relative_url}: {files_error}")
            
            # Process subfolders
            try:
                folders = folder.folders
                ctx.load(folders)
                ctx.execute_query()
                
                for subfolder in folders:
                    subfolder_url = subfolder.properties["ServerRelativeUrl"]
                    list_and_download_documents_recursively(subfolder_url)
            except Exception as folders_error:
                print(f"Error processing subfolders in {folder_relative_url}: {folders_error}")
                
        except Exception as e:
            print(f"Error processing folder {folder_relative_url}: {e}")
    
    # Start downloading from the document library root
    print(f"Starting to download documents from {sharepoint_site}/{document_library}")
    list_and_download_documents_recursively(sharepoint_site + "/" + document_library)
    
    # Use the existing get_all_chunks function to process the downloaded files
    print(f"Processing downloaded files with get_all_chunks...")
    # Import locally to avoid circular import
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from data_processing.data_extraction.chunker import get_all_chunks
    all_chunks = get_all_chunks(temp_folder, output_folder)
    
    # Clean up temporary folder
    import shutil
    shutil.rmtree(temp_folder)
    
    print(f"Processed {len(all_chunks)} chunks from SharePoint documents")
    return all_chunks


if __name__ == "__main__":
    # Process all SharePoint documents and convert to chunks
    print("Starting SharePoint document processing...")
    
    # You can specify a specific site and library, or use defaults
    # chunks = process_sharepoint_documents_to_chunks(
    #     sharepoint_site="/sites/intranet",
    #     document_library="Documentos%20compartidos",
    #     output_file="intranet_chunks.jsonl"
    # )
    
    # Or process all sites
    all_chunks = []
    sharepoint_sites = ["/sites/SATLANTISFrance"]
    
    for site in sharepoint_sites:
        print(f"\nProcessing site: {site}")
        try:
            chunks = process_sharepoint_documents_to_chunks(
                sharepoint_site=site,
                output_file=f"{site.replace('/', '_')}_chunks.jsonl"
            )
            all_chunks.extend(chunks)
            print(f"Processed {len(chunks)} chunks from {site}")
        except Exception as e:
            print(f"Error processing site {site}: {e}")
    
    print(f"\nTotal chunks processed: {len(all_chunks)}")