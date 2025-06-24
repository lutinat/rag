import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import ProductionConfig

RAW_DATA_FOLDER = ProductionConfig.RAW_DATA_FOLDER

# List of common user-agents to rotate through
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0"
]

visited = set()
total_files_saved = 0

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def is_internal_link(base_url, link):
    return urlparse(link).netloc == urlparse(base_url).netloc

def save_page_to_file(url, paragraphs, title, base_output_path):
    """Save a single page to a txt file immediately"""
    global total_files_saved
    
    if not paragraphs:
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(base_output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a safe filename from the URL
    parsed_url = urlparse(url)
    path = parsed_url.path.strip('/')
    if not path:
        path = 'home'
    
    # Replace problematic characters
    safe_filename = path.replace('/', '_').replace('?', '_').replace('&', '_').replace('=', '_')
    if len(safe_filename) > 150:  # Limit filename length
        safe_filename = safe_filename[:150]
    
    # Add URL hash to filename for uniqueness
    url_hash = str(hash(url))[-8:]
    safe_filename = f"{safe_filename}_{url_hash}"
    
    # Create the full file path
    file_path = os.path.join(output_dir, f"{safe_filename}.txt")
    
    # Write the page content to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f'# URL: {url}\n')
        f.write(f'# Title: {title or "No title"}\n\n')
        
        for paragraph in paragraphs:
            text = paragraph['text'] if isinstance(paragraph, dict) else paragraph
            f.write(text + '\n\n')
    
    print(f"Saved page to: {os.path.basename(file_path)}")
    total_files_saved += 1

def scrape(url, base_url, base_output_path):
    if url in visited:
        return
    visited.add(url)
    print(f"Scraping: {url}")

    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract paragraph texts
        page_paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text and len(text) > 50:  # Only keep meaningful paragraphs
                page_paragraphs.append({
                    'text': text,
                    'url': url,
                    'title': soup.title.string if soup.title else None
                })
        
        # Save page immediately if it has content
        if page_paragraphs:
            save_page_to_file(url, page_paragraphs, soup.title.string if soup.title else None, base_output_path)

        # Find and process all internal links
        for a_tag in soup.find_all('a', href=True):
            link = urljoin(url, a_tag['href'])
            if is_internal_link(base_url, link):
                scrape(link, base_url, base_output_path)

        # Pause between requests (random delay between 1 and 3 seconds)
        time.sleep(random.uniform(1, 3))

    except requests.RequestException as e:
        print(f"Failed to retrieve {url}: {e}")


# Starting point
base_url = 'https://www.satlantis.com'
base_output_path = f'{RAW_DATA_FOLDER}/{base_url.split("/")[-1]}'

print(f"Starting web scraping of {base_url}")
print(f"Files will be saved to: {base_output_path}")

scrape(base_url, base_url, base_output_path)

print(f"Scraping completed!")
print(f"Total pages visited: {len(visited)}")
print(f"Total files saved: {total_files_saved}")
