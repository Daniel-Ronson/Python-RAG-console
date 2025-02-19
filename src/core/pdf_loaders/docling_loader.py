from typing import Dict, Any, List
from docling.document_converter import DocumentConverter
from .base_loader import BasePDFLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, MarkdownTextSplitter
import re
import logging

logger = logging.getLogger(__name__)

class DoclingPDFLoader(BasePDFLoader):
    """Enhanced Docling-based PDF loader that parses the entire markdown document into ordered chunks,
    tagging chunks as either 'text' or 'table'. All content is preserved.
    
    The block splitting logic uses a lookahead: when a line matches a table title pattern (e.g. "Table 1: ..."),
    we only treat it as a table block if the next line (or subsequent lines) are table rows (i.e. start and end with "|").
    Otherwise, the line is included in a text block.
    """
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load the PDF and convert it to structured markdown chunks."""
        # Step 1: Convert PDF to Markdown
        markdown_text = self.convert_to_markdown(file_path)
        
        # Process the entire document as one piece
        chunks = self._process_content(markdown_text)
        
        return {
            'pages': [{
                'number': 1,  # Single "page" containing all chunks
                'text': markdown_text,
                'chunks': chunks
            }],
            'metadata': {'source': file_path}
        }

    def _process_content(self, content: str) -> List[Dict]:
        """Process content into ordered chunks.
        
        The method first splits the document into blocks (of type "table" or "text") using a line-by-line approach.
        For each block:
          - Table blocks are emitted as a single chunk.
          - Text blocks are further split using header splitting (via MarkdownHeaderTextSplitter)
            and then, if necessary, further split into sub-chunks (via MarkdownTextSplitter).
        """
        blocks = self._split_into_blocks(content)
        chunks = []
        cumulative_offset = 0  # Track character offset (approximate)
        
        for block in blocks:
            block_content = block["content"]
            block_type = block["type"]
            if block_type == "table":
                # Extract table title from the first line, if present
                lines = block_content.splitlines()
                table_title = lines[0].strip() if lines and re.match(r'^Table\s+\d+[:\.]', lines[0].strip(), re.IGNORECASE) else ""
                chunk = {
                    'type': 'table',
                    'content': block_content,
                    'table_title': table_title,
                    'offset': cumulative_offset,
                    'is_chart': True,
                    'chunk_index': f"{len(chunks)}"
                }
                chunks.append(chunk)
                cumulative_offset += len(block_content)
            else:
                # Process text block using header splitting first
                header_splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3")
                    ]
                )
                header_splits = header_splitter.split_text(block_content)
                
                # Secondary splitter for overly large chunks
                text_splitter = MarkdownTextSplitter(
                    chunk_size=2000,  # Adjust as needed
                    chunk_overlap=200
                )
                
                for doc in header_splits:
                    if len(doc.page_content) > 2000:
                        sub_chunks = text_splitter.split_text(doc.page_content)
                        for sub_chunk in sub_chunks:
                            chunk = {
                                'type': 'text',
                                'content': sub_chunk,
                                'offset': cumulative_offset,
                                'chunk_index': f"{len(chunks)}",
                                'header_1': doc.metadata.get("Header 1", ""),
                                'header_2': doc.metadata.get("Header 2", ""),
                                'header_3': doc.metadata.get("Header 3", "")
                            }
                            chunks.append(chunk)
                            cumulative_offset += len(sub_chunk)
                    else:
                        chunk = {
                            'type': 'text',
                            'content': doc.page_content,
                            'offset': cumulative_offset,
                            'chunk_index': f"{len(chunks)}",
                            'header_1': doc.metadata.get("Header 1", ""),
                            'header_2': doc.metadata.get("Header 2", ""),
                            'header_3': doc.metadata.get("Header 3", "")
                        }
                        chunks.append(chunk)
                        cumulative_offset += len(doc.page_content)
        
        logger.debug(f"Processed document into {len(chunks)} chunks")
        return chunks

    def _split_into_blocks(self, content: str) -> List[Dict]:
        """
        Split the markdown document into ordered blocks.
        
        Blocks are determined as follows:
         - If a line matches a table title pattern (e.g. "Table 1:"), look ahead.
         - If the immediately following line (or subsequent lines) are table rows (i.e. lines starting and ending with "|"),
           then treat the candidate title plus those table rows as a table block.
         - Otherwise, treat the candidate line as part of a regular text block.
        
        Returns a list of dictionaries, each with:
          - "type": either "table" or "text"
          - "content": the block's content as a string.
        """
        lines = content.splitlines()
        blocks = []
        text_accum = []
        i = 0
        table_title_re = re.compile(r'^Table\s+\d+[:\.]', re.IGNORECASE)
        table_row_re = re.compile(r'^\|.*\|$')
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            # Check if this line could be a table title candidate
            if table_title_re.match(stripped):
                # Look ahead: if the next line exists and is a table row, then we have a table block.
                if i + 1 < len(lines) and table_row_re.match(lines[i + 1].strip()):
                    # Flush any accumulated text as a text block.
                    if text_accum:
                        blocks.append({"type": "text", "content": "\n".join(text_accum)})
                        text_accum = []
                    # Start accumulating table block: include the title line
                    table_block = [line]
                    i += 1
                    # Accumulate all subsequent lines that are table rows.
                    while i < len(lines) and table_row_re.match(lines[i].strip()):
                        table_block.append(lines[i])
                        i += 1
                    blocks.append({"type": "table", "content": "\n".join(table_block)})
                    continue  # Skip the rest of the loop
                else:
                    # Not followed by a table row, so treat as regular text.
                    text_accum.append(line)
                    i += 1
            else:
                # Regular line; add to text accumulator.
                text_accum.append(line)
                i += 1
        
        # Flush any remaining text lines.
        if text_accum:
            blocks.append({"type": "text", "content": "\n".join(text_accum)})
        return blocks

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

    def convert_to_markdown(self, file_path: str) -> str:
        """Convert PDF to Markdown using Docling."""
        converter = DocumentConverter()
        doc = converter.convert(file_path)
        return doc.document.export_to_markdown()
