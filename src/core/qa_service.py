from typing import List
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from opensearchpy import OpenSearch
from colorama import Fore, Style
import re

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
        # Add colors list for cycling through reference colors
        self.ref_colors = [Fore.CYAN, Fore.GREEN, Fore.YELLOW, Fore.MAGENTA, Fore.BLUE]

    def _search_similar_chunks(self, question: str) -> List[dict]:
        """Search for similar chunks using hybrid search (KNN + text similarity)."""
        # Get the embedding for the input question
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=question
        )
        question_embedding = response.data[0].embedding

        # Build the hybrid query combining text match and KNN search
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

        # Run the search query with min_score set to filter out low-relevance hits
        response = self.client.search(
            index=INDEX_NAME,
            body={
                "query": hybrid_query,
                "size": MAX_CHUNKS_PER_QUERY,
                "_source": ["text_content", "document_id", "page_number"],
                "min_score": 0.7  # OpenSearch returns only hits with _score >= 0.7
            }
        )

        results = response['hits']['hits']

        # Print out the first 50 characters of the text content for each result
        # print("\nRelevant text chunks found:")
        # for idx, hit in enumerate(results):
        #     text_preview = hit['_source']['text_content'][:50] + "..."
        #     print(f"\n[Chunk {idx+1}] Score: {hit['_score']:.2f}")
        #     print(text_preview)
        # print("\n")

        return results

    def _highlight_references(self, text: str) -> str:
        """Highlight reference tags with cycling colors."""
        # Find all unique reference numbers
        pattern = r'\[Ref(\d+)\]'
        matches = re.finditer(pattern, text)
        
        # Create a mapping of reference numbers to colors
        ref_color_map = {}
        for match in matches:
            ref_num = int(match.group(1))
            if ref_num not in ref_color_map:
                ref_color_map[ref_num] = self.ref_colors[(ref_num - 1) % len(self.ref_colors)]
        
        # Apply colors to all references
        result = text
        for ref_num, color in sorted(ref_color_map.items(), reverse=True):
            ref_tag = f'[Ref{ref_num}]'
            colored_ref = f'{color}{ref_tag}{Style.RESET_ALL}'
            result = result.replace(ref_tag, colored_ref)
        
        return result

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

        response_with_refs = response + reference_legend
        return self._highlight_references(response_with_refs) 