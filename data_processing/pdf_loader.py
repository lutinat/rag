import pdfplumber
import os
from typing import Dict, List, Tuple
import re
import glob
def load_pdf(pdf_path: str) -> str:
    """Charge un fichier PDF et retourne son contenu textuel."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

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
    Extrait le texte d'un PDF en conservant la structure et les titres.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        
    Returns:
        str: Texte extrait avec structure préservée
    """
    text_blocks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extraire le texte avec les coordonnées
            words = page.extract_words(
                x_tolerance=3,  # Tolérance horizontale pour regrouper les mots
                y_tolerance=3,  # Tolérance verticale pour regrouper les mots
                keep_blank_chars=False,
                use_text_flow=True,
                horizontal_ltr=True,
                vertical_ttb=True,
                extra_attrs=["fontname", "size"]
            )
            
            # Grouper les mots par ligne
            lines = {}
            for word in words:
                y = round(word['top'], 1)  # Arrondir pour grouper les lignes proches
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
                
                # Détecter les titres (basé sur la taille de police et la position)
                is_title = any(
                    w['size'] > 12 for w in words  # Taille de police plus grande
                ) or y < 100  # Position en haut de page
                
                if is_title:
                    text_blocks.append(f"\n# {line_text}\n")
                else:
                    text_blocks.append(line_text)
            
            text_blocks.append("\n")  # Saut de page
    
    return "\n".join(text_blocks)

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
    pdf_files = glob.glob("/home/lucasd/code/rag/data/*.pdf")
    s = 0
    for pdf_file in pdf_files:
        print(pdf_file)
        title = get_title_from_pdf(pdf_file)
        print(f"\n\nTitle: {title}")
        if not title:
            s += 1

        metadata = get_metadata_plumberpdf(pdf_file)
        print(f"Metadata: {metadata}")
    print(f"Number of files with no title: {s}")