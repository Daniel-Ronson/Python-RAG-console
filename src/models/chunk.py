from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ParagraphChunk:
    document_id: str
    checksum: str
    is_chart: bool
    page_number: int
    paragraph_or_chart_index: str
    text_content: str
    embedding_model: str
    embedding: Optional[List[float]] = None 