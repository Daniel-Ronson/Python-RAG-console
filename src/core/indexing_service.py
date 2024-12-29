from typing import List
from opensearchpy import OpenSearch
from ..config.settings import (
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_USER,
    OPENSEARCH_PASSWORD,
    INDEX_NAME,
    VECTOR_DIMENSION
)
from ..models.chunk import ParagraphChunk

class IndexingService:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
            http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
            use_ssl=False
        )
        self.ensure_index()

    def ensure_index(self):
        """Create the index if it doesn't exist."""
        if not self.client.indices.exists(INDEX_NAME):
            mapping = {
                "mappings": {
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "checksum": {"type": "keyword"},
                        "is_chart": {"type": "boolean"},
                        "page_number": {"type": "integer"},
                        "paragraph_or_chart_index": {"type": "keyword"},
                        "text_content": {"type": "text"},
                        "embedding_model": {"type": "keyword"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": VECTOR_DIMENSION
                        }
                    }
                }
            }
            self.client.indices.create(INDEX_NAME, body=mapping)

    def index_chunks(self, chunks: List[ParagraphChunk]):
        """Index a list of chunks into OpenSearch."""
        bulk_data = []
        for chunk in chunks:
            bulk_data.extend([
                {"index": {"_index": INDEX_NAME}},
                {
                    "document_id": chunk.document_id,
                    "checksum": chunk.checksum,
                    "is_chart": chunk.is_chart,
                    "page_number": chunk.page_number,
                    "paragraph_or_chart_index": chunk.paragraph_or_chart_index,
                    "text_content": chunk.text_content,
                    "embedding_model": chunk.embedding_model,
                    "embedding": chunk.embedding
                }
            ])
        
        if bulk_data:
            self.client.bulk(body=bulk_data) 