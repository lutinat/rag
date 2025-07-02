#!/usr/bin/env python3
"""
Script principal pour traiter tous les documents SharePoint et les convertir en chunks.
Ce script utilise les loaders PDF, DOCX, et CSV pour extraire le contenu
et le chunker avec les métadonnées appropriées.
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent))

from data_processing.data_extraction.pdf_loader import process_sharepoint_documents_to_chunks

def main():
    """
    Fonction principale pour traiter les documents SharePoint.
    """
    print("=== Traitement des documents SharePoint ===")
    print("Ce script va :")
    print("1. Se connecter à SharePoint avec les credentials du fichier .env")
    print("2. Parcourir tous les sites configurés")
    print("3. Extraire le contenu des fichiers PDF, DOCX, et CSV")
    print("4. Créer des chunks avec métadonnées")
    print("5. Sauvegarder les chunks dans des fichiers JSONL")
    print()
    
    # Vérifier que le fichier .env existe
    if not os.path.exists('.env'):
        print("❌ Erreur: Le fichier .env n'existe pas.")
        print("   Veuillez copier env_example.txt vers .env et remplir vos credentials SharePoint.")
        return
    
    # Demander confirmation à l'utilisateur
    response = input("Voulez-vous continuer ? (y/N): ")
    if response.lower() != 'y':
        print("Opération annulée.")
        return
    
    try:
        # Traiter tous les sites SharePoint
        all_chunks = []
        sharepoint_sites = ["/sites/SATLANTISFrance"]
        
        for site in sharepoint_sites:
            print(f"\n🔄 Traitement du site: {site}")
            try:
                chunks = process_sharepoint_documents_to_chunks(
                    sharepoint_site=site,
                    output_folder="processed_data"
                )
                all_chunks.extend(chunks)
                print(f"✅ {len(chunks)} chunks traités pour {site}")
            except Exception as e:
                print(f"❌ Erreur lors du traitement du site {site}: {e}")
        
        print(f"\n🎉 Traitement terminé !")
        print(f"📊 Total des chunks générés: {len(all_chunks)}")
        print(f"📁 Les fichiers de chunks ont été sauvegardés dans le dossier 'processed_data/'")
        
        # Afficher un résumé des fichiers créés
        processed_files = list(Path("processed_data").glob("*_chunks.jsonl"))
        if processed_files:
            print("\n📋 Fichiers créés:")
            for file in processed_files:
                print(f"   - {file.name}")
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        return

if __name__ == "__main__":
    main() 