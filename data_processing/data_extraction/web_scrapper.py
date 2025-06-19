import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from fpdf import FPDF
import json
import time
import random

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


def save_to_txt_with_urls(scraped_data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for page in scraped_data:
            url = page['url']
            for paragraph in page['paragraphs']:
                # paragraph est un dict, on prend le texte
                text = paragraph['text'] if isinstance(paragraph, dict) else paragraph
                f.write(f'# URL: {url}\n')
                f.write(text + '\n\n')
    print(f"Saved TXT with URLs to {file_path}")


# Starting point
base_url = 'https://www.satlantis.com'
scrape(base_url, base_url)

# Save scraped data to TXT (with URLs)
output_txt_path = '/home/elduayen/rag/data/' + base_url.split('/')[-1] + '.txt'
save_to_txt_with_urls(scraped_data, output_txt_path)

print(f"Total pages scraped: {len(scraped_data)}")
print(f"Total paragraphs scraped: {sum(len(page['paragraphs']) for page in scraped_data)}")
