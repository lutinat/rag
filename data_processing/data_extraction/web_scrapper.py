import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
import os
import sys
import hashlib
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

def normalize_url(url):
    """Remove fragment (hash) from URL to avoid scraping the same page multiple times"""
    parsed = urlparse(url)
    # Reconstruct URL without fragment
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}{('?' + parsed.query) if parsed.query else ''}"

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
    sub_path = parsed_url.path
    main_path = parsed_url.netloc.replace('www.', '')
    path = f'{main_path}/{sub_path}'.strip('/')
    
    # Replace problematic characters
    safe_filename = path.replace('/', '_').replace('?', '_').replace('&', '_').replace('=', '_').replace('.', '_')
    if len(safe_filename) > 150:  # Limit filename length
        safe_filename = safe_filename[:150]
    
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
    # Normalize URL to remove fragments for deduplication
    normalized_url = normalize_url(url)
    
    if normalized_url in visited:
        return
    visited.add(normalized_url)
    print(f"Scraping: {url}")

    try:
        headers = {'User-Agent': get_random_user_agent()}
        # Use the original URL for the actual request (fragments are ignored by servers anyway)
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


def scrape_websites(output_folder=None, urls=None):
    """
    Main function to scrape websites
    
    Args:
        output_folder: Base folder to save scraped data
        urls: Can be a single URL string or a list of URLs
    """
    global visited, total_files_saved
    
    # Reset global counters for each run
    visited = set()
    total_files_saved = 0
    
    if not urls:
        raise ValueError("No URLs provided")
    
    # Handle both single URL string and list of URLs
    if isinstance(urls, str):
        urls = [urls]
    
    # Process each URL
    for url in urls:
        base_url = url
        base_output_path = f'{output_folder}/{base_url.split("/")[-1]}'
        print(f"Starting web scraping of {base_url}")
        print(f"Files will be saved to: {base_output_path}")
        scrape(base_url, base_url, base_output_path)
    
    print(f"Scraping completed!")
    print(f"Total pages visited: {len(visited)}")
    print(f"Total files saved: {total_files_saved}")


if __name__ == "__main__":
    urls = ['https://www.satlantis.com']
    scrape_websites(output_folder=RAW_DATA_FOLDER, urls=urls)