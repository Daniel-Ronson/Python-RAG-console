from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ParagraphChunk:
    title: str
    documentChecksum: str
    is_chart: bool
    page_number: int
    paragraph_or_chart_index: str
    text_content: str
    embedding_model: str
    pdf_loader: str
    embedding: Optional[List[float]] = None 