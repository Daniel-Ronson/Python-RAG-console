import hashlib
from pathlib import Path
import fitz  # PyMuPDF
from typing import List, Dict, Any
from src.models.chunk import ParagraphChunk
from src.config.settings import EMBEDDING_MODEL, PDF_LOADER_TYPE
import re as regex
from .pdf_loaders.factory import PDFLoaderFactory, PDFLoaderType

class PDFParser:
    def __init__(self):
        self.current_document_id = None
        self.current_checksum = None
        # Lazy load the appropriate loader only when needed
        self._loader = None

    @property
    def loader(self):
        """Lazy loader property that initializes the loader only when first accessed."""
        if self._loader is None:
            self._loader = PDFLoaderFactory.create(PDF_LOADER_TYPE)
        return self._loader

    def compute_checksum(self, file_path: Path) -> str:
        """Compute MD5 checksum of a file."""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def parse_pdf(self, file_path: Path, document_checksum: str) -> List[ParagraphChunk]:
        """
        Parse a PDF file using the configured loader.
        
        Args:
            file_path: Path to the PDF file
            document_checksum: Checksum of the document
            
        Returns:
            List of ParagraphChunks
        """
        self.current_document_id = file_path.name
        self.current_checksum = document_checksum
        
        # Get the document data from the loader
        doc_data = self.loader.load(str(file_path))
        chunks = []
        
        # Get the loader type directly from the factory
        loader_type = PDFLoaderFactory.get_loader_type()
        
        # Process all chunks (now in a single "page")
        page = doc_data['pages'][0]  # Only one "page" containing all chunks
        for chunk_data in page.get('chunks', []):
            chunk = ParagraphChunk(
                title=self.current_document_id,
                documentChecksum=self.current_checksum,
                is_chart=(chunk_data.get('type') == 'table' or chunk_data.get('type') == 'image'),
                page_number=chunk_data.get('offset', 0),  # Use offset instead of page number
                paragraph_or_chart_index=str(chunk_data.get('chunk_index')),
                text_content=chunk_data.get('content'),
                embedding_model=EMBEDDING_MODEL,
                pdf_loader=loader_type
            )
            chunks.append(chunk)
        
        return chunks


    def parse_pdf_old(self, file_path: Path, document_checksum: str) -> List[ParagraphChunk]:
        """Parse a PDF file and return a list of chunks."""
        self.current_document_id = file_path.name
        self.current_checksum = document_checksum
        chunks = []

        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc, 1):
            # Extract text with better formatting preservation
            text = page.get_text("text")  # Use "text" format instead of default
            
            # Split into paragraphs more robustly
            paragraphs = []
            current_paragraph = []
            
            for line in text.split('\n'):
                line = line.strip()
                if line:  # If line is not empty
                    current_paragraph.append(line)
                elif current_paragraph:  # Empty line and we have content
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            
            # Don't forget the last paragraph
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
            
            # Filter out any empty paragraphs
            paragraphs = [p for p in paragraphs if p.strip()]
            
            # Create chunks for text paragraphs
            for idx, paragraph in enumerate(paragraphs):
                chunk = ParagraphChunk(
                    title=self.current_document_id,
                    documentChecksum=self.current_checksum,
                    is_chart=False,
                    page_number=page_num,
                    paragraph_or_chart_index=f"p{idx}",
                    text_content=paragraph,
                    embedding_model=EMBEDDING_MODEL
                )
                chunks.append(chunk)

            # Extract images (basic implementation)
            image_list = page.get_images()
            for img_idx, img in enumerate(image_list):
                chunk = ParagraphChunk(
                    title=self.current_document_id,
                    documentChecksum=self.current_checksum,
                    is_chart=True,
                    page_number=page_num,
                    paragraph_or_chart_index=f"chart-{img_idx}",
                    text_content=f"Chart or figure found on page {page_num}",
                    embedding_model=EMBEDDING_MODEL
                )
                chunks.append(chunk)

        doc.close()
        return chunks 