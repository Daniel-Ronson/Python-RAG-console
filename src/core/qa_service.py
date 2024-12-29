from typing import List
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from opensearchpy import OpenSearch

from src.config.settings import (
    OPENAI_API_KEY,
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_USER,
    OPENSEARCH_PASSWORD,
    INDEX_NAME,
    COMPLETION_MODEL,
    MAX_CHUNKS_PER_QUERY,
    EMBEDDING_MODEL
)

class QAService:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
            http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
            use_ssl=False
        )
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.llm = ChatOpenAI(
            model_name=COMPLETION_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )

    def _search_similar_chunks(self, question: str) -> List[dict]:
        """Search for similar chunks using the question embedding."""
        # Get embedding for the question
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=question
        )
        question_embedding = response.data[0].embedding

        # Search in OpenSearch using KNN query instead of script query
        knn_query = {
            "knn": {
                "embedding": {
                    "vector": question_embedding,
                    "k": MAX_CHUNKS_PER_QUERY
                }
            }
        }

        response = self.client.search(
            index=INDEX_NAME,
            body={
                "query": knn_query,
                "_source": ["text_content", "document_id", "page_number"]
            }
        )

        return response['hits']['hits']

    def answer_question(self, question: str) -> str:
        """Answer a question using the indexed papers."""
        # Get relevant chunks
        similar_chunks = self._search_similar_chunks(question)
        
        if not similar_chunks:
            return "I couldn't find any relevant information to answer your question."

        # Prepare context from chunks
        context = "\n\n".join([
            f"[Ref{idx+1}] {hit['_source']['text_content']}"
            for idx, hit in enumerate(similar_chunks)
        ])

        # Prepare prompt
        prompt_template = """
        Answer the question based on the following context. Use the reference numbers [Ref1], [Ref2], etc. 
        when citing information from the context. If you cannot answer the question based on the context, 
        say so.

        Context:
        {context}

        Question: {question}

        Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Get answer from LLM
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            return_source_documents=True
        )

        response = self.llm.predict(
            prompt.format(
                context=context,
                question=question
            )
        )

        # Add reference legend
        reference_legend = "\n\nReferences:"
        for idx, hit in enumerate(similar_chunks):
            source = hit['_source']
            reference_legend += f"\n[Ref{idx+1}] Document: {source['document_id']}, Page: {source['page_number']}"

        return response + reference_legend 