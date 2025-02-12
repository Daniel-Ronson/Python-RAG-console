from typing import Dict, Any
from docling.document_converter import DocumentConverter
from .base_loader import BasePDFLoader

class DoclingPDFLoader(BasePDFLoader):
    """Docling-based PDF loader implementation."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        converter = DocumentConverter()
        doc = converter.convert_pdf_to_markdown(file_path)
        
        # Process the Docling output into our standardized schema
        pages = []
        for page_num, page_content in enumerate(doc.pages):
            pages.append({
                'number': page_num + 1,
                'text': page_content.text,
                'images': [
                    {
                        'index': idx,
                        'width': img.width,
                        'height': img.height,
                        'format': img.format
                    } for idx, img in enumerate(page_content.images)
                ],
                'tables': page_content.tables,
                'metadata': page_content.metadata
            })
        
        return {
            'pages': pages,
            'metadata': doc.metadata
        } 