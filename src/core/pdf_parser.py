import hashlib
from pathlib import Path
import fitz  # PyMuPDF
from typing import List
from src.models.chunk import ParagraphChunk

class PDFParser:
    def __init__(self):
        self.current_document_id = None
        self.current_checksum = None

    def compute_checksum(self, file_path: Path) -> str:
        """Compute MD5 checksum of a file."""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def parse_pdf(self, file_path: Path) -> List[ParagraphChunk]:
        """Parse a PDF file and return a list of chunks."""
        self.current_document_id = file_path.name
        self.current_checksum = self.compute_checksum(file_path)
        chunks = []

        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc, 1):
            # Extract text
            text = page.get_text()
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            # Create chunks for text paragraphs
            for idx, paragraph in enumerate(paragraphs):
                chunk = ParagraphChunk(
                    document_id=self.current_document_id,
                    checksum=self.current_checksum,
                    is_chart=False,
                    page_number=page_num,
                    paragraph_or_chart_index=f"p{idx}",
                    text_content=paragraph,
                    embedding_model="text-embedding-ada-002"
                )
                chunks.append(chunk)

            # Extract images (basic implementation)
            image_list = page.get_images()
            for img_idx, img in enumerate(image_list):
                chunk = ParagraphChunk(
                    document_id=self.current_document_id,
                    checksum=self.current_checksum,
                    is_chart=True,
                    page_number=page_num,
                    paragraph_or_chart_index=f"chart-{img_idx}",
                    text_content=f"Chart or figure found on page {page_num}",
                    embedding_model="text-embedding-ada-002"
                )
                chunks.append(chunk)

        doc.close()
        return chunks 