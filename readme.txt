# Scientific Paper Parser

A Python application that parses scientific papers (PDFs), indexes both text and chart/figure metadata, and provides question-answering capabilities with color-coded references in the console.

## Features

- PDF parsing with text and image extraction
- Vector embeddings using OpenAI's API
- Semantic search using OpenSearch
- Question-answering capabilities using LangChain
- Color-coded reference system in console output

## Prerequisites

- Python 3.9+
- Docker and Docker Compose (for OpenSearch)
- OpenAI API key

## Installation

1. Clone the repository:
    git clone https://github.com/yourusername/scientific-paper-parser.git
2. Create virtual env and Install dependencies:
    `python -m venv venv`
    `source venv/bin/activate`
    `pip install -r requirements.txt`
3. Create and update `.env` which is a new file (excluded from version control) and populate it like so:
   ```ini
   OPENAI_API_KEY=<your-openai-api-key>
   OPENSEARCH_HOST=localhost
   OPENSEARCH_PORT=9200
   OPENSEARCH_USERNAME=admin
   OPENSEARCH_PASSWORD=admin
   EMBEDDING_MODEL=""
   COMPLETION_MODEL=""
   ```

4. Build the Docker image for OpenSearch:
 - docker pull opensearchproject/opensearch:2
 - docker pull opensearchproject/opensearch-dashboards:2
 - ```docker-compose up --build```
 ### You can verify opensearch is running with this command: ```curl http://localhost:9200```
 ### Access OpenSearch Dashboards: Open a web browser and navigate to: ```http://localhost:5601/app/home```

5. Run the application:
    python main.py


## Usage
1. Start the OpenSearch container:
    docker-compose up -d
2. Run the application:
    python main.py
3. Basic commands:
    `ingest <folder>`: Parse and index PDFs from a folder
    `ask <question>`: Ask a question about the indexed papers
    `help`: Show available commands
    `exit`: Exit the application
