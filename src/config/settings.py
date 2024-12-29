import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment settings
ENV = os.getenv('APP_ENV', 'dev')
IS_DEV = ENV == 'dev'

# OpenSearch settings
OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT = int(os.getenv('OPENSEARCH_PORT', '9200'))
OPENSEARCH_USER = os.getenv('OPENSEARCH_USER', 'admin')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', 'admin')
INDEX_NAME = os.getenv('OPENSEARCH_INDEX', 'papers-index')

# OpenAI settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
COMPLETION_MODEL = os.getenv('COMPLETION_MODEL', 'gpt-3.5-turbo')

# Vector settings
VECTOR_DIMENSION = 1536  # For Ada-002 embeddings
MAX_CHUNKS_PER_QUERY = int(os.getenv('MAX_CHUNKS_PER_QUERY', '5')) 