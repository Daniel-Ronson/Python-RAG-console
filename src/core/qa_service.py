from typing import List
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from opensearchpy import OpenSearch
from colorama import Fore, Style

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
        """Search for similar chunks using hybrid search (KNN + text similarity)."""
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=question
        )
        question_embedding = response.data[0].embedding

        # Optimized Hybrid Search Query
        hybrid_query = {
            "bool": {
                "should": [
                    {
                        "match": {
                            "text_content": {
                                "query": question,
                                "fuzziness": "1",  # Reduced fuzziness for stricter matching
                                "operator": "AND",  # Require all terms to be present
                                "boost": 0.3
                            }
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": question_embedding,
                                "k": MAX_CHUNKS_PER_QUERY,
                                "boost": 0.7
                            }
                        }
                    }
                ],
                "minimum_should_match": 1  # Ensures at least one strong match
            }
        }

        response = self.client.search(
            index=INDEX_NAME,
            body={
                "query": hybrid_query,
                "size": MAX_CHUNKS_PER_QUERY,
                "_source": ["text_content", "document_id", "page_number"],
                "min_score": 0.7  # Filters out low-relevance results
            }
        )

        # Filter out weak matches from KNN results
        filtered_results = [
            hit for hit in response['hits']['hits']
            if hit['_score'] >= 0.7  # Ensures we only return highly relevant matches
        ]

        return filtered_results


    def answer_question(self, question: str) -> str:
        """Answer a question using the indexed papers."""
        # Get relevant chunks
        similar_chunks = self._search_similar_chunks(question)
        
        if not similar_chunks:
            # Fallback to general knowledge with a disclaimer
            prompt_template = """
            The user asked a question but no relevant documents were found in the knowledge base.
            Please provide a brief, general answer based on your knowledge. Let the user know that no personal documents were used to help answer their question

            Question: {question}

            Answer:"""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["question"]
            )

            response = self.llm.invoke(
                prompt.format(question=question)
            ).content

            return f"{Fore.YELLOW}Note: No relevant documents found in the index. Providing a general answer:{Style.RESET_ALL}\n\n{response}"

        # Prepare context from chunks
        context = "\n\n".join([
            f"[Ref{idx+1}] {hit['_source']['text_content']}"
            for idx, hit in enumerate(similar_chunks)
        ])

        # Prepare prompt
        prompt_template = """
        Answer the question based on the following context. Use the reference numbers [Ref1], [Ref2], etc. 
        when citing information from the context or if the reference has the same information. If you cannot answer the question based on the context, 
        say so and provide a general answer based on your knowledge.  Rememeber Always cite sources at the time you use them. provide an answer that is betwee n1 and 3 paragraphs.
        Consider all the given sources, your own internal knowledge, and think critically about the information provided and the question before answering.


        -------------------------
        Context:
        {context}
        -------------------------
        Question: {question}
        -------------------------
        Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Get answer from LLM using invoke instead of predict
        response = self.llm.invoke(
            prompt.format(
                context=context,
                question=question
            )
        ).content  # Add .content to get the string response

        # Add reference legend
        reference_legend = "\n\nReferences:"
        for idx, hit in enumerate(similar_chunks):
            source = hit['_source']
            reference_legend += f"\n[Ref{idx+1}] Document: {source['document_id']}, Page: {source['page_number']}"

        return response + reference_legend 