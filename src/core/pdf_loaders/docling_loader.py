from typing import Dict, Any, List
from docling.document_converter import DocumentConverter
from .base_loader import BasePDFLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, MarkdownTextSplitter
import re
import logging

logger = logging.getLogger(__name__)

class DoclingPDFLoader(BasePDFLoader):
    """Enhanced Docling-based PDF loader with header-based metadata and table extraction."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load the PDF and convert it to structured markdown chunks."""
        # Step 1: Convert PDF to Markdown
        markdown_text = self.convert_to_markdown(file_path)
        
        # Step 2: Split into pages
        pages = self._split_into_pages(markdown_text)
        
        # Step 3 & 4: Process each page into chunks
        processed_pages = []
        for page_num, page_content in enumerate(pages, 1):
            chunks = self._process_page_content(page_content, page_num)
            processed_pages.append({
                'number': page_num,
                'text': page_content,
                'chunks': chunks
            })
        
        return {
            'pages': processed_pages,
            'metadata': {'source': file_path}
        }

    def _split_into_pages(self, markdown_text: str) -> List[str]:
        """Split markdown text into pages based on page break markers."""
        # Assuming page breaks are marked with horizontal rules (---)
        pages = re.split(r'\n-{3,}\n', markdown_text)
        return [page.strip() for page in pages if page.strip()]

    def _process_page_content(self, content: str, page_num: int) -> List[Dict]:
        """Process page content into semantic chunks with headers as metadata and table extraction."""
        try:
            # Extract tables separately
            tables = self._extract_tables(content)
            
            # Remove tables from content before text chunking
            cleaned_content = self._remove_extracted_tables(content, tables)

            chunks = []
            
            # First split by headers
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3")
                ]
            )
            # This returns List[Document] where each Document has page_content and metadata
            header_splits = header_splitter.split_text(cleaned_content)

            # Create text splitter for large chunks
            text_splitter = MarkdownTextSplitter(
                chunk_size=2000,  # Smaller chunk size to stay well under token limit
                chunk_overlap=200
            )

            # Process each Document object
            for doc_idx, doc in enumerate(header_splits):
                if len(doc.page_content) > 2000:
                    # Split large content into smaller chunks
                    sub_chunks = text_splitter.split_text(doc.page_content)
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        chunk = {
                            'type': 'text',
                            'content': sub_chunk,
                            'page_number': page_num,
                            'chunk_index': f"{doc_idx}-{sub_idx}",
                            'header_1': doc.metadata.get('Header 1', ''),
                            'header_2': doc.metadata.get('Header 2', ''),
                            'header_3': doc.metadata.get('Header 3', '')
                        }
                        chunks.append(chunk)
                else:
                    # Use original chunk if it's small enough
                    chunk = {
                        'type': 'text',
                        'content': doc.page_content,
                        'page_number': page_num,
                        'chunk_index': doc_idx,
                        'header_1': doc.metadata.get('Header 1', ''),
                        'header_2': doc.metadata.get('Header 2', ''),
                        'header_3': doc.metadata.get('Header 3', '')
                    }
                    chunks.append(chunk)

            logger.debug(f"Split page {page_num} into {len(chunks)} chunks")

            # Add extracted tables as separate chunks
            for idx, table in enumerate(tables):
                chunks.append({
                    'type': 'table',
                    'content': table['table_text'],
                    'table_title': table['title'],
                    'page_number': page_num,
                    'chunk_index': f"table-{idx}"
                })

            return chunks
            
        except Exception as e:
            logger.error(f"Error in DoclingPDFLoader._process_page_content: {str(e)}")
            # If header splitting fails, try fallback method
            try:
                split_texts = self._fallback_split(content, max_length=2000)
                chunks = [{
                    'type': 'text',
                    'content': text,
                    'page_number': page_num,
                    'chunk_index': idx
                } for idx, text in enumerate(split_texts)]
                return chunks
            except Exception as inner_e:
                logger.error(f"Fallback splitting failed: {str(inner_e)}")
                raise e

    def _fallback_split(self, text: str, max_length: int) -> List[str]:
        """Fallback method to split text into chunks if header splitting fails."""
        chunks = []
        current_chunk = ""
        
        for paragraph in text.split('\n\n'):
            if len(current_chunk) + len(paragraph) < max_length:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _extract_tables(self, content: str) -> List[Dict]:
        """Extract tables from markdown content along with their titles."""
        tables = []
        table_pattern = re.compile(r'(Table\s\d+:.*?)\n\n(\|.*?\|(?:\n\|[-:]+.*?)+\n(?:\|.*?\|.*?\n)+)', re.DOTALL)

        for match in table_pattern.finditer(content):
            table_title = match.group(1).strip()
            table_text = match.group(2).strip()
            tables.append({"title": table_title, "table_text": table_text})

        return tables

    def _remove_extracted_tables(self, content: str, tables: List[Dict]) -> str:
        """Remove extracted tables from content to prevent duplication in text chunks."""
        for table in tables:
            content = content.replace(table["table_text"], "")
        return content

    def convert_to_markdown(self, file_path: str) -> str:
        """Convert PDF to Markdown using Docling."""
        converter = DocumentConverter()
        doc = converter.convert(file_path)
        return doc.document.export_to_markdown()
