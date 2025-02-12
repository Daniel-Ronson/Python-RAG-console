import fitz
from typing import Dict, Any
from .base_loader import BasePDFLoader

class FitzPDFLoader(BasePDFLoader):
    """PyMuPDF-based PDF loader implementation."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        doc = fitz.open(file_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text
            text = page.get_text()
            # Extract images
            images = []
            for img_index, img in enumerate(page.get_images()):
                xref = img[0]
                base_image = doc.extract_image(xref)
                if base_image:
                    images.append({
                        'index': img_index,
                        'width': base_image['width'],
                        'height': base_image['height'],
                        'format': base_image['ext']
                    })
            
            pages.append({
                'number': page_num + 1,
                'text': text,
                'images': images,
                'tables': [],  # PyMuPDF doesn't extract tables by default
                'metadata': page.metadata if hasattr(page, 'metadata') else {}
            })
        
        return {
            'pages': pages,
            'metadata': doc.metadata
        } 