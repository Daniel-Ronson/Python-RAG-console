from typing import List
from openai import OpenAI
from ..config.settings import OPENAI_API_KEY, EMBEDDING_MODEL
from ..models.chunk import ParagraphChunk

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using OpenAI's API."""
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def embed_chunks(self, chunks: List[ParagraphChunk]) -> List[ParagraphChunk]:
        """Embed a list of chunks and return them with their embeddings."""
        for chunk in chunks:
            chunk.embedding = self.get_embedding(chunk.text_content)
        return chunks 