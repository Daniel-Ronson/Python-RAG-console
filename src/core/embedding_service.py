import asyncio
from openai import AsyncOpenAI
from typing import List
import logging
import time
from ..models.chunk import ParagraphChunk
from ..config.settings import OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.total_embeddings = 0

    async def _get_embedding_async(self, chunk: ParagraphChunk) -> ParagraphChunk:
        """Get embedding for a single chunk asynchronously."""
        try:
            response = await self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=chunk.text_content
            )
            chunk.embedding = response.data[0].embedding
            return chunk
        except Exception as e:
            logger.error(f"Error getting embedding for chunk {chunk.paragraph_or_chart_index}: {str(e)}")
            raise

    async def _get_embeddings_batch(self, chunks: List[ParagraphChunk]) -> List[ParagraphChunk]:
        """Process multiple chunks in parallel."""
        tasks = [self._get_embedding_async(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)

    def embed_chunks(self, chunks: List[ParagraphChunk]) -> List[ParagraphChunk]:
        """Get embeddings for a list of chunks using parallel processing."""
        if not chunks:
            return chunks

        try:
            start_time = time.time()
            chunk_count = len(chunks)
            self.total_embeddings = 0  # Reset at start
            logger.info(f"Starting embedding generation for {chunk_count} chunks")

            # Create a new event loop for this operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                embedded_chunks = loop.run_until_complete(self._get_embeddings_batch(chunks))
            finally:
                loop.close()

            self.total_embeddings += chunk_count
            elapsed_time = time.time() - start_time
            rate = chunk_count / elapsed_time
            
            logger.info(
                f"Completed {chunk_count} embeddings in {elapsed_time:.2f} seconds "
                f"({rate:.2f} embeddings/second). "
                f"Total embeddings created: {self.total_embeddings}"
            )
            
            return embedded_chunks
        except Exception as e:
            logger.error(f"Error during batch embedding: {str(e)}")
            self.total_embeddings = 0  # Reset on error
            raise
