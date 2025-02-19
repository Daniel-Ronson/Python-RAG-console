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
import logging

logger = logging.getLogger(__name__)

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
                        "title": {"type": "keyword"},
                        "documentChecksum": {"type": "keyword"},
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
        try:
            bulk_data = []
            for chunk in chunks:
                # Each chunk should only create one entry
                bulk_data.extend([
                    # This is the operation metadata
                    {"index": {"_index": INDEX_NAME}},
                    # This is the actual document data
                    {
                        "title": chunk.title,
                        "documentChecksum": chunk.documentChecksum,
                        "is_chart": chunk.is_chart,
                        "page_number": chunk.page_number,
                        "paragraph_or_chart_index": chunk.paragraph_or_chart_index,
                        "text_content": chunk.text_content,
                        "embedding_model": chunk.embedding_model,
                        "embedding": chunk.embedding,
                        "pdf_loader": chunk.pdf_loader
                    }
                ])
            
            if bulk_data:
                # Add debug logging
                logger.debug(f"Indexing {len(bulk_data)//2} chunks")  # Divide by 2 because each chunk has 2 entries
                response = self.client.bulk(body=bulk_data)
                
                # Check for errors
                if response.get('errors'):
                    error_items = [item for item in response['items'] if item.get('index', {}).get('error')]
                    logger.error(f"Bulk indexing had {len(error_items)} errors: {error_items}")
                    
                return {
                    'indexed': len(bulk_data)//2,
                    'errors': len(error_items) if response.get('errors') else 0
                }
                
        except Exception as e:
            logger.error(f"Error during bulk indexing: {str(e)}")
            raise

    def get_index_stats(self) -> dict:
        """Get statistics about the index."""
        try:
            stats = self.client.indices.stats(index=INDEX_NAME)
            total = stats['indices'][INDEX_NAME]['total']
            return {
                'doc_count': total['docs']['count'],
                'store_size': total['store']['size_in_bytes']
            }
        except Exception as e:
            raise Exception(f"Failed to get index stats: {str(e)}")

    def get_sample_documents(self, size: int = 5) -> list:
        """Get a sample of documents from the index."""
        try:
            response = self.client.search(
                index=INDEX_NAME,
                body={
                    "query": {"match_all": {}},
                    "size": size,
                    "sort": [{"_doc": "desc"}]  # Random sort
                }
            )
            return response['hits']['hits']
        except Exception as e:
            raise Exception(f"Failed to get sample documents: {str(e)}") 

    def check_existing_checksums(self, checksums: List[str]) -> set:
        """Check which checksums from the provided list already exist in the index."""
        try:
            response = self.client.search(
                index=INDEX_NAME,
                body={
                    "size": 0,
                    "query": {
                        "terms": {
                            "documentChecksum": checksums
                        }
                    },
                    "aggs": {
                        "existing_checksums": {
                            "terms": {
                                "field": "documentChecksum",
                                "size": len(checksums)
                            }
                        }
                    }
                }
            )
            return {bucket['key'] for bucket in response['aggregations']['existing_checksums']['buckets']}
        except Exception as e:
            print(f"Warning: Could not check checksums: {str(e)}")
            return set() 

    def delete_by_document_ids(self, document_ids: List[str]) -> dict:
        """Delete all chunks associated with given document IDs."""
        try:
            response = self.client.delete_by_query(
                index=INDEX_NAME,
                body={
                    "query": {
                        "terms": {
                            "document_id": document_ids
                        }
                    }
                },
                refresh=True  # Ensure deletion is immediately visible
            )
            
            return {
                'total_deleted': response['deleted'],
                'total_failed': response['failures']
            }
        except Exception as e:
            raise Exception(f"Failed to delete documents: {str(e)}") 

    def delete_all_documents(self) -> dict:
        """Delete all documents from the index."""
        try:
            response = self.client.delete_by_query(
                index=INDEX_NAME,
                body={
                    "query": {
                        "match_all": {}
                    }
                },
                refresh=True  # Ensure deletion is immediately visible
            )
            
            return {
                'total_deleted': response['deleted'],
                'total_failed': len(response.get('failures', []))
            }
        except Exception as e:
            raise Exception(f"Failed to delete all documents: {str(e)}") 