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

    # Iterate through the tables in the document
    # Markdown representation of the document
    # print(doc._.markdown)

    return doc.text


def load_pdfs_spacy(pdf_path_list: List[str]) -> List[str]:
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


    # Process docs as needed
    for doc in all_docs:
        print(doc.text)



def load_txt(txt_path: str) -> str:
    """Charge un fichier texte et retourne son contenu textuel."""
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
    Extrait le texte d'un PDF en conservant la structure et les titres, optimisé pour un traitement ultérieur avec SpaCy.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        
    Returns:
        str: Texte extrait avec structure préservée pour traitement ultérieur.
    """
    text_blocks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extraire les mots avec les coordonnées
            words = page.extract_words(
                x_tolerance=1,
                y_tolerance=1,
                keep_blank_chars=False,
                use_text_flow=True,
                horizontal_ltr=True,
                vertical_ttb=True,
                extra_attrs=["fontname", "size"]
            )
            
            # Grouper les mots par ligne
            lines = {}
            for word in words:
                y = round(word['top'], 1)  # Grouper les mots proches
                if y not in lines:
                    lines[y] = []
                lines[y].append(word)
            
            # Trier les lignes par position verticale
            sorted_lines = sorted(lines.items())
            
            # Reconstruire le texte avec la structure
            for y, words in sorted_lines:
                # Trier les mots de gauche à droite
                words.sort(key=lambda w: w['x0'])
                line_text = ' '.join(w['text'] for w in words)
                
                # Détecter les titres (plus précis avec un seuil de taille et une position)
                is_title = any(
                    w['size'] > 12 for w in words  # Taille de police plus grande
                ) or y < 100  # Position en haut de page
                
                if is_title:
                    # Marquer les titres de manière explicite
                    text_blocks.append(f"\n## {line_text}\n")
                else:
                    text_blocks.append(line_text)
            
            text_blocks.append("\n")  # Saut de page
    
    # Retourner le texte brut, normalisé
    raw_text = "\n".join(text_blocks)
    clean_text = " ".join(raw_text.split())  # Normaliser les espaces excessifs
    return clean_text


def extract_metadata_from_pdf(pdf_path: str) -> Dict:
    """
    Extrait les métadonnées du PDF.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        
    Returns:
        Dict: Métadonnées extraites
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
    title_from_pdf = get_title_from_pdf(pdf_path)
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
    from msal import ConfidentialClientApplication
    import requests

    client_id = ""
    tenant_id = ""
    client_secret = ""
    authority = ""
    scopes = ""

    app = ConfidentialClientApplication(client_id=client_id, authority=authority, client_credential=client_secret)

  # Acquire token using client credentials flow (no need for user interaction)
    result = app.acquire_token_for_client(scopes=scopes)

    if "access_token" in result:
        # Successfully obtained access token, now you can make requests
        headers = {
            "Authorization": f"Bearer {result['access_token']}"
        }
        
        # Make API call to get files from your SharePoint folder
        site_url = "https://satlantis.sharepoint.com/sites/ImgProcss"
        api = f"{site_url}/_api/web/GetFolderByServerRelativeUrl('Documents')/Files"

        resp = requests.get(api, headers=headers)
        if resp.ok:
            files = resp.json()["value"]
            for file in files:
                print(file["Name"])
        else:
            print("Error:", resp.status_code, resp.text)
    else:
        print("Failed to acquire token:", result.get("error_description"))

    # Print PDF files only
    for file in files:
        if file.properties["Name"].lower().endswith(".pdf"):
            print(file.properties["Name"])

    
    # s = 0
    # for pdf_file in pdf_files:
    #     print(pdf_file)
    #     title = get_title_from_pdf(pdf_file)
    #     print(f"\n\nTitle: {title}")
    #     if not title:
    #         s += 1

    #     metadata = get_metadata_plumberpdf(pdf_file)
    #     print(f"Metadata: {metadata}")
    # print(f"Number of files with no title: {s}")