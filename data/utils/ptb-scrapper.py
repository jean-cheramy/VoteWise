from pathlib import Path
from urllib.parse import urlparse
from typing import List
import requests
from bs4 import BeautifulSoup
import re

BASE_DIR: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = BASE_DIR / "raw" / "fr" / "ptb_programme"
LINKS_FILE: Path = BASE_DIR / "utils" / "ptb-links.txt"


def read_links(file_path: Path) -> List[str]:
    """
    Read URLs from a text file, one per line, ignoring empty lines.

    Args:
        file_path (Path): Path to the text file containing URLs.

    Returns:
        List[str]: List of URLs as strings.
    """
    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fetch_clean_text(url: str) -> str:
    """
    Download a webpage and extract cleaned text content.

    Keeps only the content between "Home\nProgramme" and either
    "Notes de bas de page" or "Partager\nPartagez cette page".

    Args:
        url (str): The URL of the page to scrape.

    Returns:
        str: Cleaned text content from the page.
    """
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove scripts, styles, and noscript tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    match = re.search(
        r"(?s)Home\s*Programme(.*?)(?:Notes de bas de page|Partager\s*Partagez cette page|$)",
        clean,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else clean


def save_page_text(url: str, text: str, out_dir: Path) -> None:
    """
    Save the cleaned text content to a file.

    The filename is derived from the last path segment of the URL,
    replacing "/" with "_" and defaulting to "index.txt".

    Args:
        url (str): URL of the page.
        text (str): Cleaned text content.
        out_dir (Path): Directory where the file will be saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
    file_path = out_dir / f"{slug}.txt"
    file_path.write_text(text, encoding="utf-8")


def main() -> None:
    """
    Main function to scrape all URLs from the links file and save their text.
    """
    urls: List[str] = read_links(LINKS_FILE)
    print(f"Found {len(urls)} URLs to scrape.")

    for url in urls:
        print(f"Scraping {url}")
        text = fetch_clean_text(url)
        save_page_text(url, text, OUTPUT_DIR)

    print(f"All pages saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
