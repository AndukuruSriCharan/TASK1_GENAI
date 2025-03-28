import os
import re
import requests
import time
import csv
import pandas as pd
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# List of company URLs
URLS = [
    "https://www.apple.com",
    "https://www.toyota-global.com",
    "https://www.jpmorganchase.com",
    "https://www.pfizer.com",
    "https://www.thewaltdisneycompany.com",
    "https://www.shell.com",
    "https://www.siemens.com",
    "https://www.samsung.com/in/",
    "https://www.nike.com"
]

# Keywords to find additional links
COMPANY_KEYWORDS = {
    "https://www.apple.com": ['business', 'newsroom', 'compliance'],
    "https://www.toyota-global.com": ['executives', 'global-vision', 'company', 'financial-results', 'profile'],
    "https://www.jpmorganchase.com": ['leadership', 'awards-and-recognition', 'our-history', 'suppliers', 'business-principles'],
    "https://www.pfizer.com": ['product-list', 'executives', 'history', 'purpose', 'global-impact'],
    "https://www.thewaltdisneycompany.com": ['our-businesses', 'news', 'social-impact', 'about'],
    "https://www.shell.com": ['who-we-are', 'our-values', 'our-history', 'news-and-insights'],
    "https://www.siemens.com": ['products.html', 'management.html', 'telegraphy-and-telex.html', 'system-and-method-for-robotic-picking.html', 'technology-to-transform-the-everyday.html'],
    "https://www.samsung.com/in/": ['company-info', 'business-area', 'brand-identity', 'environment'],
    "https://www.nike.com": ['about', 'investors', 'sustainability']
}

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY, transport="rest")


def fetch_content(url):
    """Fetches webpage content and extracts JavaScript-based links."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return None, None

    soup = BeautifulSoup(response.text, 'html.parser')
    js_links = re.findall(r'https?://[^\s"\'<>]+', response.text)
    return soup, js_links


def clean_text(text):
    """Cleans text by removing extra spaces, quotes, and newlines for CSV compatibility."""
    return re.sub(r'\s+', ' ', text.replace('"', "'").replace('\n', ' ')).strip()


def extract_text(soup):
    """Extracts and cleans visible text from a webpage."""
    for tag in soup(['script', 'style']):
        tag.decompose()

    # Append link URLs next to their text
    for a in soup.find_all('a', href=True):
        a.insert_after(f" ({a['href']})")

    return soup.get_text(separator=' ', strip=True)


def call_gemini_api(cleaned_text):
    """Calls the Gemini API to extract structured company details."""
    prompt = f"""
    Extract the following detailed company information from the provided text:
    
    1. What is the company's mission statement or core values?
    2. What products or services does the company offer?
    3. When was the company founded, and who were the founders?
    4. Where is the company's headquarters located?
    5. Who are the key executives or leadership team members?
    6. Has the company received any notable awards or recognitions?
    
    Provide the output in a CSV-compatible format, ensuring each question is clearly answered.
    
    Text: {cleaned_text}
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text if response else ""
    except Exception:
        return ""


def extract_relevant_links(soup, js_links, base_url, url):
    """Finds additional relevant links based on company keywords."""
    keywords = COMPANY_KEYWORDS.get(url, [])
    links = {urljoin(base_url, a["href"]) for a in soup.find_all(
        "a", href=True) if any(k in a["href"].lower() for k in keywords)}
    links.update({link for link in js_links if any(
        k in link.lower() for k in keywords)})
    return list(links)


def get_company_information(url):
    """Scrapes and extracts structured company details."""
    soup, js_links = fetch_content(url)
    if not soup:
        return "Failed to fetch content."

    cleaned_text = extract_text(soup)
    extracted_info = call_gemini_api(cleaned_text)

    if "Information Not Available" in extracted_info:
        for link in extract_relevant_links(soup, js_links, url, url):
            additional_soup, _ = fetch_content(link)
            if additional_soup:
                cleaned_text += "\n" + extract_text(additional_soup)
                extracted_info = call_gemini_api(cleaned_text)
                if "Information Not Available" not in extracted_info:
                    break  # Stop once sufficient info is found

    return extracted_info


def save_to_csv(data, filename="company_details1.csv"):
    """Saves extracted data to a CSV file."""
    try:
        pd.DataFrame(data).to_csv(
            filename, index=False, quoting=csv.QUOTE_MINIMAL)
    except Exception:
        pass  # Silently handle CSV writing errors


def process_urls(urls):
    """Scrapes multiple company websites and saves details to a CSV."""
    output_data = [{'Company URL': url, 'Extracted Information': clean_text(
        get_company_information(url))} for url in urls]
    save_to_csv(output_data)


if __name__ == "__main__":
    process_urls(URLS)
