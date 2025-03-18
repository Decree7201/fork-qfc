# Agentic RAG with Vertex AI Search

This agent enhances the Gen AI App Starter Pack with a production-ready data ingestion pipeline, enriching your Retrieval Augmented Generation (RAG) applications. Using Vertex AI Search's state-of-the-art search capabilities, you can ingest, process, and embed custom data, improving the relevance and context of your generated responses.

The agent provides the infrastructure to create a Vertex AI Pipeline with your custom code. Because it's built on Vertex AI Pipelines, you benefit from features like scheduled runs, recurring executions, and on-demand triggers. For processing terabyte-scale data, we recommend combining Vertex AI Pipelines with data analytics tools like BigQuery or Dataflow.

![search agent demo](https://storage.googleapis.com/github-repo/generative-ai/sample-apps/e2e-gen-ai-app-starter-pack/starter-pack-search-pattern.gif)

## Architecture

The agent implements the following architecture:

![architecture diagram](https://storage.googleapis.com/github-repo/generative-ai/sample-apps/e2e-gen-ai-app-starter-pack/agentic_rag_vertex_ai_search_architecture.png)

### Key Features

- **Vertex AI Search Integration:** Utilizes Vertex AI Search for efficient data storage and retrieval.
- **Automated Data Ingestion Pipeline:** Automates the process of ingesting data from input sources.
- **Custom Embeddings:** Generates embeddings using Vertex AI Embeddings and incorporates them into your data for enhanced semantic search.
- **Terraform Deployment:** Ingestion pipeline is instantiated with Terraform alongside the rest of the infrastructure of the starter pack.
- **Cloud Build Integration:** Deployment of ingestion pipelines is added to the CD pipelines of the starter pack.
- **Customizable Code:** Easily adapt and customize the code to fit your specific application needs and data sources.
