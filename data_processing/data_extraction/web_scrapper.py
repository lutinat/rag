import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from fpdf import FPDF
import json
import time
import random
import os

# List of common user-agents to rotate through
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0"
]

visited = set()
scraped_data = []  # Changed from paragraphs to scraped_data to include metadata

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def is_internal_link(base_url, link):
    return urlparse(link).netloc == urlparse(base_url).netloc

def scrape(url, base_url):
    if url in visited:
        return
    visited.add(url)
    print(f"Scraping: {url}")

    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract and save paragraph texts with metadata
        page_paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text and len(text) > 50:  # Only keep meaningful paragraphs
                page_paragraphs.append({
                    'text': text,
                    'url': url,
                    'title': soup.title.string if soup.title else None
                })
        
        # Add page data with metadata
        if page_paragraphs:
            page_data = {
                'url': url,
                'paragraphs': page_paragraphs,
                'title': soup.title.string if soup.title else None
            }
            scraped_data.append(page_data)

        # Find and process all internal links
        for a_tag in soup.find_all('a', href=True):
            link = urljoin(url, a_tag['href'])
            if is_internal_link(base_url, link):
                scrape(link, base_url)

        # Pause between requests (random delay between 1 and 3 seconds)
        time.sleep(random.uniform(1, 3))

    except requests.RequestException as e:
        print(f"Failed to retrieve {url}: {e}")


def save_to_txt_with_urls(scraped_data, base_output_path):
    """
    Save scraped data to separate TXT files, one for each page.
    Each file will contain only the paragraphs from that specific page.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(base_output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    total_files = 0
    
    for page in scraped_data:
        url = page['url']
        paragraphs = page['paragraphs']
        
        if not paragraphs:
            continue
        
        # Create a safe filename from the URL that includes the URL for easy extraction
        parsed_url = urlparse(url)
        
        # Use the path part of the URL to create filename
        path = parsed_url.path.strip('/')
        if not path:
            path = 'home'
        
        # Replace problematic characters
        safe_filename = path.replace('/', '_').replace('?', '_').replace('&', '_').replace('=', '_')
        if len(safe_filename) > 150:  # Limit filename length
            safe_filename = safe_filename[:150]
        
        # Add URL hash to filename for uniqueness and easy URL extraction
        url_hash = str(hash(url))[-8:]  # Last 8 characters of hash
        safe_filename = f"{safe_filename}_{url_hash}"
        
        # Create the full file path
        file_path = os.path.join(output_dir, f"{safe_filename}.txt")
        
        # Write the page content to its own file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f'# URL: {url}\n')
            f.write(f'# Title: {page.get("title", "No title")}\n\n')
            
            for paragraph in paragraphs:
                text = paragraph['text'] if isinstance(paragraph, dict) else paragraph
                f.write(text + '\n\n')
        
        print(f"Saved page to: {os.path.basename(file_path)}")
        total_files += 1
    
    print(f"Created {total_files} separate TXT files")
    return total_files


# Starting point
base_url = 'https://www.satlantis.com'
scrape(base_url, base_url)

# Save scraped data to separate TXT files
base_output_path = '/home/elduayen/rag/data/' + base_url.split('/')[-1]
total_files = save_to_txt_with_urls(scraped_data, base_output_path)

print(f"Total pages scraped: {len(scraped_data)}")
print(f"Total paragraphs scraped: {sum(len(page['paragraphs']) for page in scraped_data)}")
print(f"Total files created: {total_files}")
