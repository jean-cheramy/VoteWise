from pathlib import Path
from typing import List, Dict
import re
import pandas as pd
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Chunker:
    """Class to load, normalize, chunk files per party, and save as Parquet."""
    
    def __init__(self, base_dir: Path, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            base_dir (Path): Root folder containing party subfolders.
            chunk_size (int): Chunk size in characters.
            chunk_overlap (int): Overlap between chunks in characters.
        """
        self.base_dir = base_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for efficient chunking."""
        text = text.strip()
        text = re.sub(r"[ \t]+", " ", text)                 
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       
        text = re.sub(r"\n{3,}", "\n\n", text)            
        text = re.sub(r"●", "", text)
        return text

    @staticmethod
    def pdf_to_text(pdf_path: Path) -> str:
        """Extract text from a PDF and normalize it using pdfplumber."""
        raw_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    raw_text += page_text + "\n"
        return Chunker.normalize_text(raw_text)

    @staticmethod
    def txt_to_text(txt_path: Path) -> str:
        """Read and normalize a .txt file."""
        with open(txt_path, "r", encoding="utf-8") as f:
            return Chunker.normalize_text(f.read())

    def load_and_chunk_party(self, party_dir: Path) -> List[Dict]:
        """Load and chunk all files in a single party folder."""
        party_name = party_dir.name
        party_chunks = []

        for file_path in party_dir.glob("*"):
            if file_path.suffix.lower() == ".pdf":
                text = self.pdf_to_text(file_path)
            elif file_path.suffix.lower() == ".txt":
                text = self.txt_to_text(file_path)
            else:
                continue  

            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                party_chunks.append({
                    "party": party_name,
                    "language": "fr",
                    "chunk": chunk,
                    "source": file_path.name
                })
        return party_chunks

    def process_all_parties(self) -> List[Dict]:
        """Process all party folders under the base directory."""
        all_chunks = []
        for party_dir in self.base_dir.iterdir():
            if party_dir.is_dir():
                party_chunks = self.load_and_chunk_party(party_dir)
                all_chunks.extend(party_chunks)
        return all_chunks

    def save_as_parquet(self, output_path: Path) -> None:
        """Process all parties and save chunks as a Parquet file."""
        all_chunks = self.process_all_parties()
        if all_chunks:
            df = pd.DataFrame(all_chunks)
            df.to_parquet(output_path, index=False)
            print(f"Saved {len(all_chunks)} chunks to {output_path}")
        else:
            print("No chunks found to save.")


if __name__ == "__main__":
    OUTPUT_FILE = Path("chunks/party_chunks.parquet")

    BASE_DIR: Path = Path(__file__).resolve().parent
    RAW_DIR: Path = BASE_DIR / "data" / "fr"

    chunker = Chunker(base_dir=RAW_DIR, chunk_size=1000, chunk_overlap=200)
    chunker.save_as_parquet(OUTPUT_FILE)
