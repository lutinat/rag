import re
import spacy
from typing import List, Dict
from data_processing.utils import save_chunks_jsonl
import os
from glob import glob
from .pdf_loader import extract_text_from_pdf, extract_metadata_from_pdf
from .pdf_loader import load_txt
from tqdm import tqdm


# Charger le modèle spaCy multilingue et ajouter le sentencizer
nlp = spacy.load("xx_ent_wiki_sm")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

def is_noise(text: str) -> bool:
    """Détermine si un texte est du bruit à filtrer."""
    # Patterns de bruit à filtrer
    noise_patterns = [
        r'^\d+\.',  # Numéros de section
        r'^PARTIE \d+',  # Titres de partie
        r'^\.{10,}$',  # Lignes de points
        r'^\s*\d+\s*$',  # Numéros de page seuls
        r'^.{1,30}\.\.\.+\d+$',  # Titres avec points de suite
        r'^\s*$',  # Lignes vides
        r'^.{1,50}\.\.\.+\s*\d+$',  # Autres titres avec points de suite
        r'^[A-Z][a-z]*\s*\.\.\.+\s*\d+$',  # Titres courts avec points de suite
        r'^[IVX]+\.',  # Numéros romains
        r'^\d+\.\d+',  # Sous-sections numérotées
        r'^[A-Z]\.',  # Lettres majuscules seules
        r'^[a-z]\)',  # Lettres minuscules avec parenthèse
        r'^\d+\)',  # Nombres avec parenthèse
        r'^Table des matières',  # Titres de table des matières
        r'^Annexe',  # Titres d'annexe
        r'^Figure \d+',  # Légendes de figure
        r'^Tableau \d+',  # Légendes de tableau
        r'^[A-Z]{2,}$',  # Acronymes seuls
        r'^\d{1,2}/\d{1,2}/\d{4}',  # Dates
        r'^©',  # Copyright
        r'^http[s]?://',  # URLs
        r'^www\.',  # URLs
        r'^[A-Z][a-z]*\s*:\s*$',  # Titres avec deux points
        r'^\d{1,2}h\d{2}',  # Heures
        r'^\d{1,2}:\d{2}',  # Heures
        r'^\d{1,2}/\d{1,2}',  # Dates courtes
        r'^\d{4}-\d{2}-\d{2}',  # Dates ISO
    ]
    
    # Vérifier les patterns de bruit
    if any(re.match(pattern, text.strip()) for pattern in noise_patterns):
        return True
        
    # Vérifier si le texte est trop court
    if len(text.strip()) < 30:
        return True
        
    # Vérifier le ratio de caractères spéciaux
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    if len(clean_text) < len(text) * 0.4:
        return True
        
    # Vérifier le ratio de majuscules
    if len(re.findall(r'[A-Z]', text)) > len(text) * 0.4:
        return True
        
    return False

def clean_text(text: str) -> str:
    """Nettoie le texte en supprimant les éléments indésirables."""
    # # Remplacer les retours à la ligne par des espaces
    # text = text.replace('\n', ' ')
    
    # # Supprimer les espaces multiples
    # text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les caractères spéciaux en début/fin de ligne
    text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)
    
    # Normaliser les guillemets
    text = re.sub(r'["\']', '"', text)
    
    # Normaliser les tirets
    text = re.sub(r'[-–—]', '-', text)
    
    # Supprimer les points de suite avec numéros de page
    text = re.sub(r'\.\.\.+\s*\d+$', '', text)
    
    # Supprimer les numéros de page seuls
    text = re.sub(r'^\s*\d+\s*$', '', text)

    text = re.sub(r'(\s*\.\s*){2,}', '.', text)
    
    # Gérer les listes à puces
    text = re.sub(r'^\s*[•\-\*]\s*', '', text)
    
    # Gérer les tableaux
    text = re.sub(r'\s*\|\s*', ' ', text)  # Remplacer les séparateurs de colonnes
    text = re.sub(r'\s*-\s*', ' ', text)   # Remplacer les lignes de séparation
    
    # # S'assurer qu'il y a un espace après les points, points d'exclamation et points d'interrogation
    # text = re.sub(r'([.!?])\s*', r'\1 ', text)


    
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """Divise le texte en phrases en utilisant le sentencizer de spaCy."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]  # Filter out short sentences
    return sentences


def create_paragraphs(text: str) -> List[str]:
    """Crée des paragraphes à partir du texte."""
    # Nettoyer le texte
    text = clean_text(text)
    
    # Diviser en phrases
    sentences = split_into_sentences(text)
    
    # Créer des paragraphes
    paragraphs = []
    current_paragraph = []
    current_length = 0
    
    for sentence in sentences:
        # Nettoyer la phrase
        cleaned = clean_text(sentence)
        if is_noise(cleaned):
            continue
            
        # Si la phrase est trop longue, la diviser
        if len(cleaned) > 150:  # Réduit de 200 à 150
            words = cleaned.split()
            temp_sentence = []
            temp_length = 0
            
            for word in words:
                if temp_length + len(word) + 1 <= 150:
                    temp_sentence.append(word)
                    temp_length += len(word) + 1
                else:
                    if temp_sentence:
                        current_paragraph.append(' '.join(temp_sentence))
                        current_length += temp_length
                    temp_sentence = [word]
                    temp_length = len(word)
            
            if temp_sentence:
                current_paragraph.append(' '.join(temp_sentence))
                current_length += temp_length
        else:
            current_paragraph.append(cleaned)
            current_length += len(cleaned) + 1
        
        # Si le paragraphe est assez long, le sauvegarder
        if current_length > 150:  # Réduit de 200 à 150
            paragraph_text = ' '.join(current_paragraph)
            if not is_noise(paragraph_text):
                paragraphs.append(paragraph_text)
            current_paragraph = []
            current_length = 0
    
    # Ajouter le dernier paragraphe
    if current_paragraph:
        paragraph_text = ' '.join(current_paragraph)
        if not is_noise(paragraph_text):
            paragraphs.append(paragraph_text)
    
    return paragraphs

def create_chunks(paragraphs: List[str], 
                  chunk_size: int = 2, 
                  overlap: int = 1, 
                  max_chunk_length: int = 300, 
                  source_name: str = "document.pdf",
                  metadata: Dict = None) -> List[Dict]:
    """Crée des chunks à partir des paragraphes avec overlap."""
    chunks = []
    chunk_id = 0
    
    # Créer des chunks avec overlap
    for i in range(0, len(paragraphs), chunk_size - overlap):
        # Créer un chunk avec les paragraphes
        chunk_paragraphs = paragraphs[i:i + chunk_size]
        if len(chunk_paragraphs) >= 1:  # Minimum 1 paragraphe par chunk
            # Joindre les paragraphes en respectant la ponctuation
            chunk_text = ' '.join(chunk_paragraphs)
            
            if len(chunk_text) >= 50:  # Minimum 50 caractères
                # Vérifier si le chunk est trop similaire au précédent
                if not chunks or not is_similar(chunk_text, chunks[-1]["text"], threshold=0.7):
                    chunk_data = {
                        "id": f"{source_name}_c{chunk_id}",
                        "text": chunk_text,
                        "metadata": metadata
                    }
                    chunks.append(chunk_data)
                    chunk_id += 1
    
    return chunks

def extract_chunks(text: str, source_name: str, metadata: Dict) -> List[Dict]:
    """Extrait des chunks de haute qualité à partir d'un texte brut."""
    # Créer des paragraphes
    paragraphs = create_paragraphs(text)
    
    # Créer des chunks avec overlap
    chunks = create_chunks(
        paragraphs,
        chunk_size=2,  # 2 paragraphes par chunk
        overlap=1,     # 1 paragraphe d'overlap
        max_chunk_length=300,  # Maximum 300 caractères par chunk
        source_name=source_name,
        metadata=metadata
    )
    
    return chunks

def is_similar(text1: str, text2: str, threshold: float = 0.7) -> bool:
    """Vérifie si deux textes sont trop similaires."""
    # Convertir en ensembles de mots
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculer le ratio de mots communs
    common_words = words1.intersection(words2)
    similarity = len(common_words) / max(len(words1), len(words2))
    
    return similarity > threshold


def get_all_chunks(folder_path: str, chunk_folder_path: str):
    """
    Process multiple PDFs, extract text and metadata, chunk the text, and save chunks to JSONL.
    
    Args:
        folder_path (str): The directory containing PDFs to process.
        chunk_folder_path (str): The folder where JSONL files will be saved.
    """
    # Get all PDF files in the directory
    pdf_paths = glob(os.path.join(folder_path, "*.pdf"))
    txt_paths = glob(os.path.join(folder_path, "*.txt"))
    csv_paths = glob(os.path.join(folder_path, "*.csv"))
    all_chunks = []
    
    # PDF FILES
    print(f"Processing {len(pdf_paths)} PDFs")
    for pdf_path in tqdm(pdf_paths):
        # print(f"Processing {pdf_path}")
        # Extract text and metadata
        text = extract_text_from_pdf(pdf_path)
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
        text = load_txt(txt_path)
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