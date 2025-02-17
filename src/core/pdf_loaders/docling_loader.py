from typing import Dict, Any, List
from docling.document_converter import DocumentConverter
from .base_loader import BasePDFLoader
import os
class DoclingPDFLoader(BasePDFLoader):
    """Docling-based PDF loader implementation."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        markdown_text = self.convert_to_markdown(file_path)
        
        # Create markdown directory if it doesn't exist
        os.makedirs("markdown", exist_ok=True)
        
        # Use relative path instead of absolute
        filename = os.path.basename(file_path)
        with open(f"markdown/{filename}-markdown.md", "w") as f:
            f.write(markdown_text)
        return markdown_text
        
    def convert_to_markdown(self, file_path: str) -> str:
        converter = DocumentConverter()
        doc = converter.convert(file_path)
        markdown_text = doc.document.export_to_markdown()
        return markdown_text
    
    def convert_each_page_to_markdown(self, file_path: str) -> List[str]:
        converter = DocumentConverter()
        doc = converter.convert(file_path)
        return [page.text for page in doc.pages]


        #converter = DocumentConverter()
        # doc = converter.convert(file_path)
        # Process the Docling output into our standardized schema
        # pages = []
        # for page_num, page_content in enumerate(doc.pages):
        #     pages.append({
        #         'number': page_num + 1,
        #         'text': page_content.text,
        #         'images': [
        #             {
        #                 'index': idx,
        #                 'width': img.width,
        #                 'height': img.height,
        #                 'format': img.format
        #             } for idx, img in enumerate(page_content.images)
        #         ],
        #         'tables': page_content.tables,
        #         'metadata': page_content.metadata
        #     })
        
        # return {
        #     'pages': pages,
        #     'metadata': doc.metadata
        # } 