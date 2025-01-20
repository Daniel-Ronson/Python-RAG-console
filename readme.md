# Scientific Paper Parser 

A console application that ingests scientific papers (PDFs) and allows you to ask questions to an LLM with the relevant context from the papers you uploaded. This app indexes pdf text, metadata, and vector embeddings from a pdf and provides question-answering in the console.

## Features

- Ingest PDF's and chunk text into vector embeddings (no image extraction yet)
- Vector embeddings using OpenAI's API
- Storage and Semantic search using OpenSearch
- Ask an LLM a question - using OpenAi api

## Prerequisites

- **Python 3.11+**
- Docker (needed to run OpenSearch locally)
- OpenAI API key

## Installation

1. Clone the repository:
    git clone https://github.com/Daniel-Ronson/Python-RAG-console.git
2. Create virtual env and Install dependencies:
    `python -m venv myenv`
    `source myenv/bin/activate`
    `pip install -r requirements.txt`
3. Create and update `.env` which is a new file (excluded from version control) and populate it like so:
   ```ini
   OPENAI_API_KEY=<your-openai-api-key>
   OPENSEARCH_HOST=localhost
   OPENSEARCH_PORT=9200
   OPENSEARCH_USERNAME=admin
   OPENSEARCH_PASSWORD=admin
   EMBEDDING_MODEL=text-embedding-ada-002
   COMPLETION_MODEL=gpt-3.5-turbo
   ```

4. Build the Docker image for OpenSearch:
 - docker pull opensearchproject/opensearch:2
 - docker pull opensearchproject/opensearch-dashboards:2
 - ```docker-compose up --build```
 #### You can verify opensearch is running with this command: ```curl http://localhost:9200```
 #### To access OpenSearch Dashboards: Open a web browser and navigate to: ```http://localhost:5601/app/home```



## Usage
1. Start the OpenSearch container:
    docker-compose up -d
2. Run the application:
    python3 run.py
3. Basic commands:
    - `ingest <folder-path>`: Parse and index PDFs from a folder (example: `ingest /path/to/pdf/folder)
    - `ask <question>`: Ask a question about the indexed papers (example: `ask What is a vector?`)
    - `help`: Show available commands
    - `status`: Show status of OpenSearch database documents
    - `reload`: Hot reload Python code for local development
    - `invalidate <folder> *or* <filepath>`: Delete all chunks associated with a document
    - `exit`: Exit the application  
   **FYI:** You will `ingest` pdfs before you can `ask` questions about them
